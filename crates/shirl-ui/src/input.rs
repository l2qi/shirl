// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Single-line input editor state and key->action mapping.
//!
//! Pure state plus a free `on_key` function so the editor logic is testable
//! without a terminal. The viewport renderer reads `current()`/`cursor()` to
//! draw the prompt; nothing in this module touches I/O.

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::completion;

const DOUBLE_PRESS_WINDOW: Duration = Duration::from_millis(800);

/// Mutable state for the single-line editor.
#[derive(Debug, Default)]
pub(super) struct InputState {
    input: String,
    cursor: usize,
    history_idx: Option<usize>,
    saved_draft: String,
    /// When navigating Up/Down across visual lines of different lengths,
    /// remember the column the user was on so it's restored on longer lines.
    /// Cleared on any non-Up/Down edit or cursor action.
    desired_col: Option<usize>,
    /// Cursor affinity at a soft-wrap boundary, where one char index is both
    /// the end of one visual row and the start of the next. `true` renders the
    /// cursor at the end of the row it closes (set by `Ctrl+E`); `false`
    /// renders it at the start of the next row (the default for typing,
    /// `Ctrl+A`, arrows, and Up/Down). Reset by every other action.
    prefer_row_end: bool,
}

impl InputState {
    pub(super) fn current(&self) -> &str {
        &self.input
    }

    pub(super) fn cursor(&self) -> usize {
        self.cursor
    }

    /// Whether the cursor should render at the end of the visual row it closes
    /// (rather than the start of the next) when it sits on a soft-wrap
    /// boundary. See [`prefer_row_end`](Self::prefer_row_end).
    pub(super) fn prefer_row_end(&self) -> bool {
        self.prefer_row_end
    }

    pub(super) fn is_empty(&self) -> bool {
        self.input.is_empty()
    }

    /// Replace the input contents wholesale. Used by the file picker
    /// to splice a selected path into the buffer.
    pub(super) fn set(&mut self, s: &str) {
        self.input = s.to_string();
        self.history_idx = None;
        self.desired_col = None;
        self.prefer_row_end = false;
    }

    /// Set the cursor position (char index). Clamps to the end of input.
    pub(super) fn set_cursor(&mut self, pos: usize) {
        let max = self.input.chars().count();
        self.cursor = pos.min(max);
        self.desired_col = None;
        self.prefer_row_end = false;
    }

    /// Take the current line, leaving the editor cleared.
    pub(super) fn take(&mut self) -> String {
        let line = std::mem::take(&mut self.input);
        self.cursor = 0;
        self.history_idx = None;
        self.saved_draft.clear();
        self.desired_col = None;
        self.prefer_row_end = false;
        line
    }

    /// Discard the current line.
    pub(super) fn clear(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.saved_draft.clear();
        self.desired_col = None;
        self.prefer_row_end = false;
    }

    pub(super) fn insert_char(&mut self, c: char) {
        let bp = byte_pos(&self.input, self.cursor);
        self.input.insert(bp, c);
        self.cursor += 1;
        self.history_idx = None;
        self.desired_col = None;
    }

    /// Insert a string at the cursor position in one operation.
    /// Equivalent to calling `insert_char` per character but avoids O(n^2)
    /// for large pastes.
    pub(super) fn insert_str(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let bp = byte_pos(&self.input, self.cursor);
        self.input.insert_str(bp, s);
        self.cursor += s.chars().count();
        self.history_idx = None;
        self.desired_col = None;
    }

    fn backspace(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        let bp = byte_pos(&self.input, self.cursor);
        self.input.remove(bp);
        self.desired_col = None;
        true
    }

    fn delete(&mut self) -> bool {
        if self.cursor >= self.input.chars().count() {
            return false;
        }
        let bp = byte_pos(&self.input, self.cursor);
        self.input.remove(bp);
        self.desired_col = None;
        true
    }

    fn cursor_left(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        self.desired_col = None;
        true
    }

    fn cursor_right(&mut self) -> bool {
        if self.cursor >= self.input.chars().count() {
            return false;
        }
        self.cursor += 1;
        self.desired_col = None;
        true
    }

    fn home(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor = 0;
        self.desired_col = None;
        true
    }

    fn end(&mut self) -> bool {
        let n = self.input.chars().count();
        if self.cursor == n {
            return false;
        }
        self.cursor = n;
        self.desired_col = None;
        true
    }

    /// Move cursor to the start of the current visual line.
    /// For single-line or unwrapped text this is equivalent to `home()`.
    /// When already at the start of a visual line, steps back to the
    /// previous visual line's start so repeated presses chain backwards
    /// - the mirror of how Ctrl+E chains forwards.
    fn ctrl_a(&mut self, width: usize) -> bool {
        let info = visual_line_info(&self.input, self.cursor, width);
        let target = info.line_starts[info.vis_line];
        if self.cursor == target {
            // Already at the start of this visual line. Step back to the
            // previous visual line's start so repeated Ctrl+A presses chain
            // backwards - the mirror of how Ctrl+E chains forwards.
            if info.vis_line == 0 {
                return false;
            }
            self.cursor = info.line_starts[info.vis_line - 1];
        } else {
            self.cursor = target;
        }
        self.desired_col = None;
        true
    }

    /// Move cursor to the end of the current visual line.
    /// For single-line or unwrapped text this is equivalent to `end()`.
    /// When already at the end of a visual line, steps forward to the next
    /// visual line's end so repeated presses chain forwards - the mirror of
    /// how Ctrl+A chains backwards.
    fn ctrl_e(&mut self, width: usize) -> bool {
        let info = visual_line_info(&self.input, self.cursor, width);
        let end = visual_line_end(&self.input, &info.line_starts, info.vis_line);
        if self.cursor == end {
            // Already at the end of this visual line. Step to the next
            // visual line's end so repeated Ctrl+E presses chain forwards.
            if info.vis_line + 1 >= info.line_starts.len() {
                return false;
            }
            self.cursor = visual_line_end(&self.input, &info.line_starts, info.vis_line + 1);
        } else {
            self.cursor = end;
        }
        self.desired_col = None;
        // Render at the end of the closed row, not the start of the next.
        self.prefer_row_end = true;
        true
    }

    fn history_prev(&mut self, history: &[String]) -> bool {
        if history.is_empty() {
            return false;
        }
        let new_idx = match self.history_idx {
            None => {
                self.saved_draft = self.input.clone();
                history.len() - 1
            }
            Some(idx) => idx.saturating_sub(1),
        };
        self.history_idx = Some(new_idx);
        self.input = history[new_idx].clone();
        self.cursor = self.input.chars().count();
        self.desired_col = None;
        true
    }

    fn history_next(&mut self, history: &[String]) -> bool {
        match self.history_idx {
            Some(idx) if idx + 1 < history.len() => {
                self.history_idx = Some(idx + 1);
                self.input = history[idx + 1].clone();
                self.cursor = self.input.chars().count();
                self.desired_col = None;
                true
            }
            Some(_) => {
                self.history_idx = None;
                self.input = std::mem::take(&mut self.saved_draft);
                self.cursor = self.input.chars().count();
                self.desired_col = None;
                true
            }
            None => false,
        }
    }

    /// Navigate to the previous visual line. If already on the first visual
    /// line, fall through to history navigation instead (matching Claude Code
    /// behavior).
    fn cursor_up(&mut self, history: &[String], width: usize) -> bool {
        let info = visual_line_info(&self.input, self.cursor, width);
        if info.vis_line == 0 {
            return self.history_prev(history);
        }
        let col = self.desired_col.unwrap_or(info.vis_col);
        self.desired_col = Some(col);
        let target_line = info.vis_line - 1;
        self.cursor = char_index_at_visual_pos(&self.input, target_line, col, &info.line_starts);
        true
    }

    /// Navigate to the next visual line. If already on the last visual line,
    /// fall through to history navigation instead (matching Claude Code
    /// behavior).
    fn cursor_down(&mut self, history: &[String], width: usize) -> bool {
        let info = visual_line_info(&self.input, self.cursor, width);
        let last_line = info.line_starts.len() - 1;
        if info.vis_line >= last_line {
            return self.history_next(history);
        }
        let col = self.desired_col.unwrap_or(info.vis_col);
        self.desired_col = Some(col);
        let target_line = info.vis_line + 1;
        self.cursor = char_index_at_visual_pos(&self.input, target_line, col, &info.line_starts);
        true
    }
}

// ---------------------------------------------------------------------------
// Visual-line layout helpers
// ---------------------------------------------------------------------------

/// Summary of where the cursor sits within the visual-line layout.
struct VisualLineInfo {
    /// 0-based index of the visual line the cursor is on.
    vis_line: usize,
    /// Column offset within that visual line.
    vis_col: usize,
    /// Char-index where each visual line starts.
    /// `line_starts[i]` is the char index of the first character on visual
    /// line `i`. Length equals the number of visual lines.
    line_starts: Vec<usize>,
}

/// Determine which visual line and column the cursor is on. Columns are
/// terminal display cells (a CJK glyph or emoji is two cells), derived from
/// the same [`visual_line_starts`](super::visual_line_starts) layout used to
/// render and wrap text, so navigation agrees with what is drawn. Returns the
/// full layout (all line starts) so callers like `ctrl_e` can find the end of
/// the current visual line.
fn visual_line_info(text: &str, cursor: usize, width: usize) -> VisualLineInfo {
    let line_starts = super::visual_line_starts(text, width);
    let vis_line = line_starts.iter().rposition(|&s| s <= cursor).unwrap_or(0);
    let vis_col = super::span_width(text, line_starts[vis_line], cursor);
    VisualLineInfo {
        vis_line,
        vis_col,
        line_starts,
    }
}

/// Given a target visual line and display column, compute the char index where
/// the cursor should be placed. Walks the line accumulating display widths and
/// stops at the first character boundary at or past `target_col`, clamping to
/// the visual line's end.
fn char_index_at_visual_pos(
    text: &str,
    target_line: usize,
    target_col: usize,
    line_starts: &[usize],
) -> usize {
    let start = line_starts[target_line];
    let end = if target_line + 1 < line_starts.len() {
        line_starts[target_line + 1]
    } else {
        text.chars().count()
    };

    let mut char_idx = start;
    let mut col = 0;
    for ch in text.chars().skip(start).take(end - start) {
        if ch == '\n' || col >= target_col {
            break;
        }
        col += super::char_width(ch);
        char_idx += 1;
    }
    char_idx
}

/// Char index of the end of visual line `target_line` - the position just
/// past its last visible character. For a line terminated by a hard newline
/// that is the `\n`'s index; for a soft-wrapped line it is the next line's
/// start; for the final visual line it is the end of the text.
fn visual_line_end(text: &str, line_starts: &[usize], target_line: usize) -> usize {
    if target_line + 1 < line_starts.len() {
        let next_start = line_starts[target_line + 1];
        // A hard newline ends the visual line one char before the next
        // line's start; a soft wrap has no separator char to skip over.
        if text.chars().nth(next_start - 1) == Some('\n') {
            next_start - 1
        } else {
            next_start
        }
    } else {
        text.chars().count()
    }
}

// ---------------------------------------------------------------------------
// Outcome types and chord tracker
// ---------------------------------------------------------------------------

/// What the input thread should do after a key event.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum InputOutcome {
    /// User pressed Enter - line ready to submit. State is already cleared.
    Submit(String),
    /// State changed; redraw needed.
    Redraw,
    /// User asked to cancel the current model turn (Ctrl+C while working,
    /// Ctrl+D on empty line while working).
    Cancel,
    /// User asked to exit (double Ctrl+C, double Ctrl+D on empty line).
    Exit,
    /// Shift+Tab - cycle permission mode.
    CycleMode,
    /// Ctrl+O - toggle transcript view.
    ToggleTranscript,
    /// Ctrl+V / Alt+V - read an image from the system clipboard and splice
    /// `@"path"` into the input buffer.
    PasteImage,
    /// Nothing to do.
    None,
}

/// Tracks repeated Ctrl+C / Ctrl+D presses for the double-press exit.
#[derive(Debug, Default)]
pub(super) struct ChordTracker {
    last_ctrl_c: Option<Instant>,
    last_ctrl_d: Option<Instant>,
}

impl ChordTracker {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Returns true if this Ctrl+C completes a double-press chord.
    fn ctrl_c(&mut self) -> bool {
        if let Some(last) = self.last_ctrl_c {
            if last.elapsed() < DOUBLE_PRESS_WINDOW {
                return true;
            }
        }
        self.last_ctrl_c = Some(Instant::now());
        false
    }

    /// Returns true if this Ctrl+D completes a double-press chord.
    fn ctrl_d(&mut self) -> bool {
        if let Some(last) = self.last_ctrl_d {
            if last.elapsed() < DOUBLE_PRESS_WINDOW {
                return true;
            }
        }
        self.last_ctrl_d = Some(Instant::now());
        false
    }
}

/// Translate a key press into an [`InputOutcome`], mutating `state` in place.
///
/// `content_width` is the terminal width available for input text (total
/// width minus the prompt indicator width). It drives soft-wrap-aware
/// navigation (Up/Down, Ctrl+A/E).
///
/// `commands` is the full list of slash commands (built-in + custom) used
/// for ghost-text completion.
pub(super) fn on_key(
    state: &mut InputState,
    history: &[String],
    chord: &mut ChordTracker,
    working: bool,
    key: KeyEvent,
    content_width: usize,
    commands: &[completion::CommandInfo],
) -> InputOutcome {
    // Soft-wrap cursor affinity only survives the keypress that sets it
    // (`Ctrl+E`). Clear it up front so every other key falls back to the
    // default (render at the start of the next row); the `Ctrl+E` arm below
    // re-sets it during its own handling.
    state.prefer_row_end = false;

    // Shift+Tab cycles the permission mode.
    if key.code == KeyCode::BackTab {
        return InputOutcome::CycleMode;
    }

    // Insert a literal newline instead of submitting.
    //
    // - Shift+Enter / Alt+Enter: work on some terminals, but NOT macOS
    //   Terminal.app (Option is used for special chars and Shift+Enter is
    //   indistinguishable from Enter).
    // - Ctrl+J: sends the actual ASCII LF byte (0x0A). Works on every
    //   terminal universally and is the reliable way to type a newline on
    //   macOS. Trade-off: a quick brush of Ctrl while typing 'j' will
    //   insert a spurious newline - accepted because there's no other
    //   universal chord available.
    if (matches!(key.code, KeyCode::Enter)
        && (key.modifiers.contains(KeyModifiers::ALT)
            || key.modifiers.contains(KeyModifiers::SHIFT)))
        || (key.code == KeyCode::Char('j') && key.modifiers.contains(KeyModifiers::CONTROL))
    {
        state.insert_char('\n');
        return InputOutcome::Redraw;
    }

    if matches!(key.code, KeyCode::Enter) {
        let line = state.take();
        if line.trim().is_empty() {
            return InputOutcome::Redraw;
        }
        return InputOutcome::Submit(line);
    }

    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if chord.ctrl_c() {
            return InputOutcome::Exit;
        }
        return if working {
            InputOutcome::Cancel
        } else {
            state.clear();
            InputOutcome::Redraw
        };
    }

    if key.code == KeyCode::Char('d')
        && key.modifiers.contains(KeyModifiers::CONTROL)
        && state.is_empty()
    {
        if chord.ctrl_d() {
            return InputOutcome::Exit;
        }
        return if working {
            InputOutcome::Cancel
        } else {
            InputOutcome::None
        };
    }

    if key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return InputOutcome::ToggleTranscript;
    }

    // Ctrl+V (universal) or Alt+V (where Option/Alt sends Meta) - paste an
    // image from the system clipboard. Text paste goes through bracketed
    // paste (Cmd+V on macOS, Ctrl+Shift+V on Linux) as Event::Paste, so the
    // raw KeyEvent reaching here is unused on every supported terminal and
    // free for image use.
    //
    // SHIFT is explicitly excluded: most Linux terminals turn Ctrl+Shift+V
    // into bracketed paste before we see it, but configurations exist where
    // it leaks through as a raw KeyEvent with CONTROL | SHIFT. Treating
    // that as a paste-image attempt would produce a confusing "no image in
    // clipboard" warning on every text paste in those setups.
    //
    // Alt+V is reliable on Linux (terminals default Alt to Meta) and on
    // macOS terminals that send Option as Meta (iTerm2, Ghostty, Alacritty,
    // kitty). macOS Terminal.app's default Option mode types `√` for
    // Option+V, which arrives as Char('√') without a modifier and is
    // inserted as text - no conflict.
    if matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V'))
        && !key.modifiers.contains(KeyModifiers::SHIFT)
        && (key.modifiers.contains(KeyModifiers::CONTROL)
            || key.modifiers.contains(KeyModifiers::ALT))
    {
        return InputOutcome::PasteImage;
    }

    let changed = match key {
        // Ctrl+A: move to start of current visual line (readline convention).
        KeyEvent {
            code: KeyCode::Char('a'),
            modifiers,
            ..
        } if modifiers.contains(KeyModifiers::CONTROL)
            && !modifiers.contains(KeyModifiers::ALT) =>
        {
            state.ctrl_a(content_width)
        }

        // Ctrl+E: move to end of current visual line (readline convention).
        KeyEvent {
            code: KeyCode::Char('e'),
            modifiers,
            ..
        } if modifiers.contains(KeyModifiers::CONTROL)
            && !modifiers.contains(KeyModifiers::ALT) =>
        {
            state.ctrl_e(content_width)
        }

        KeyEvent {
            code: KeyCode::Tab, ..
        } => accept_completion(state, commands),
        KeyEvent {
            code: KeyCode::Char(c),
            modifiers,
            ..
        } if !modifiers.contains(KeyModifiers::CONTROL)
            && !modifiers.contains(KeyModifiers::ALT) =>
        {
            state.insert_char(c);
            true
        }
        KeyEvent {
            code: KeyCode::Backspace,
            ..
        } => state.backspace(),
        KeyEvent {
            code: KeyCode::Delete,
            ..
        } => state.delete(),
        KeyEvent {
            code: KeyCode::Left,
            ..
        } => state.cursor_left(),
        KeyEvent {
            code: KeyCode::Right,
            ..
        } => {
            if state.cursor() < state.current().chars().count() {
                state.cursor_right()
            } else {
                accept_completion(state, commands)
            }
        }
        KeyEvent {
            code: KeyCode::Home,
            ..
        } => state.home(),
        KeyEvent {
            code: KeyCode::End, ..
        } => state.end(),
        // Up/Down: navigate between visual lines; fall through to history
        // navigation when already on the top/bottom edge.
        KeyEvent {
            code: KeyCode::Up, ..
        } => state.cursor_up(history, content_width),
        KeyEvent {
            code: KeyCode::Down,
            ..
        } => state.cursor_down(history, content_width),
        _ => false,
    };

    if changed {
        InputOutcome::Redraw
    } else {
        InputOutcome::None
    }
}

fn byte_pos(s: &str, char_pos: usize) -> usize {
    s.char_indices()
        .nth(char_pos)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

fn accept_completion(state: &mut InputState, commands: &[completion::CommandInfo]) -> bool {
    if state.cursor() != state.current().chars().count() {
        return false;
    }
    let suffix = match completion::complete(state.current(), commands) {
        Some(s) => s,
        None => return false,
    };
    for ch in suffix.chars() {
        state.insert_char(ch);
    }
    state.insert_char(' ');
    true
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// Default content width used by tests that don't need wrapping.
    const W: usize = 80;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn alt_enter() -> KeyEvent {
        KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)
    }

    fn shift_enter() -> KeyEvent {
        KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)
    }

    fn ctrl_j() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL)
    }

    fn cmds() -> Vec<completion::CommandInfo> {
        completion::built_in_commands()
    }

    fn drive(s: &mut InputState, history: &[String], events: &[KeyEvent]) {
        drive_w(s, history, events, W);
    }

    fn drive_w(s: &mut InputState, history: &[String], events: &[KeyEvent], width: usize) {
        let cmds = completion::built_in_commands();
        let mut chord = ChordTracker::new();
        for &e in events {
            on_key(s, history, &mut chord, false, e, width, &cmds);
        }
    }

    // -----------------------------------------------------------------------
    // Existing tests (updated to pass content_width)
    // -----------------------------------------------------------------------

    #[test]
    fn typing_advances_cursor_and_appends() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('a')),
                key(KeyCode::Char('b')),
                key(KeyCode::Char('c')),
            ],
        );
        assert_eq!(s.current(), "abc");
        assert_eq!(s.cursor(), 3);
    }

    #[test]
    fn backspace_at_zero_is_noop() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Backspace),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::None);
        assert_eq!(s.current(), "");
    }

    #[test]
    fn backspace_in_middle_removes_left_char() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('a')),
                key(KeyCode::Char('b')),
                key(KeyCode::Char('c')),
                key(KeyCode::Left),
                key(KeyCode::Backspace),
            ],
        );
        assert_eq!(s.current(), "ac");
        assert_eq!(s.cursor(), 1);
    }

    #[test]
    fn delete_in_middle_removes_right_char() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('a')),
                key(KeyCode::Char('b')),
                key(KeyCode::Char('c')),
                key(KeyCode::Home),
                key(KeyCode::Delete),
            ],
        );
        assert_eq!(s.current(), "bc");
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn home_and_end_jump_cursor() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[key(KeyCode::Char('a')), key(KeyCode::Char('b'))],
        );
        drive(&mut s, &[], &[key(KeyCode::Home)]);
        assert_eq!(s.cursor(), 0);
        drive(&mut s, &[], &[key(KeyCode::End)]);
        assert_eq!(s.cursor(), 2);
    }

    #[test]
    fn enter_returns_submit_and_clears_state() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[key(KeyCode::Char('h')), key(KeyCode::Char('i'))],
        );
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Enter),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Submit("hi".to_string()));
        assert_eq!(s.current(), "");
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn enter_on_empty_line_returns_redraw() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Enter),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.current(), "");
    }

    #[test]
    fn enter_on_whitespace_only_returns_redraw() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[key(KeyCode::Char(' ')), key(KeyCode::Char(' '))],
        );
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Enter),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
    }

    #[test]
    fn history_up_saves_draft_and_recalls_last() {
        let mut s = InputState::default();
        let history = vec!["a".to_string(), "b".to_string()];
        drive(&mut s, &history, &[key(KeyCode::Char('d'))]);
        drive(&mut s, &history, &[key(KeyCode::Up)]);
        assert_eq!(s.current(), "b");
        drive(&mut s, &history, &[key(KeyCode::Up)]);
        assert_eq!(s.current(), "a");
    }

    #[test]
    fn history_down_past_end_restores_saved_draft() {
        let mut s = InputState::default();
        let history = vec!["a".to_string(), "b".to_string()];
        drive(&mut s, &history, &[key(KeyCode::Char('d'))]);
        drive(&mut s, &history, &[key(KeyCode::Up), key(KeyCode::Down)]);
        assert_eq!(s.current(), "d");
    }

    #[test]
    fn ctrl_c_idle_clears_input() {
        let mut s = InputState::default();
        drive(&mut s, &[], &[key(KeyCode::Char('x'))]);
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('c'), W, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.current(), "");
    }

    #[test]
    fn ctrl_c_while_working_signals_cancel() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, true, ctrl('c'), W, &cmds());
        assert_eq!(outcome, InputOutcome::Cancel);
    }

    #[test]
    fn double_ctrl_c_signals_exit() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let _ = on_key(&mut s, &[], &mut chord, false, ctrl('c'), W, &cmds());
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('c'), W, &cmds());
        assert_eq!(outcome, InputOutcome::Exit);
    }

    #[test]
    // Ctrl+D only triggers exit/cancel on an empty line; on non-empty
    // input the early-return above doesn't fire, and the catch-all Char
    // arm requires !CONTROL, so the keypress is a no-op.
    fn ctrl_d_on_nonempty_input_is_noop() {
        let mut s = InputState::default();
        drive(&mut s, &[], &[key(KeyCode::Char('x'))]);
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('d'), W, &cmds());
        assert_eq!(outcome, InputOutcome::None);
        assert_eq!(s.current(), "x");
    }

    #[test]
    fn double_ctrl_d_on_empty_signals_exit() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let _ = on_key(&mut s, &[], &mut chord, false, ctrl('d'), W, &cmds());
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('d'), W, &cmds());
        assert_eq!(outcome, InputOutcome::Exit);
    }

    #[test]
    fn tab_accepts_slash_command_completion() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('/')),
                key(KeyCode::Char('p')),
                key(KeyCode::Char('r')),
                key(KeyCode::Char('o')),
                key(KeyCode::Char('v')),
                key(KeyCode::Char('i')),
                key(KeyCode::Tab),
            ],
        );
        assert_eq!(s.current(), "/provider ");
        assert_eq!(s.cursor(), 10);
    }

    #[test]
    fn right_arrow_at_end_accepts_completion() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('/')),
                key(KeyCode::Char('m')),
                key(KeyCode::Char('o')),
                key(KeyCode::Char('d')),
                key(KeyCode::Right),
            ],
        );
        // "/mod" uniquely completes to "/model " now that /mode is removed.
        assert_eq!(s.current(), "/model ");
    }

    #[test]
    fn right_arrow_in_middle_moves_cursor() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('a')),
                key(KeyCode::Char('b')),
                key(KeyCode::Left),
                key(KeyCode::Right),
            ],
        );
        assert_eq!(s.cursor(), 2);
    }

    #[test]
    fn tab_with_no_completion_is_noop() {
        let mut s = InputState::default();
        drive(&mut s, &[], &[key(KeyCode::Char('h')), key(KeyCode::Tab)]);
        assert_eq!(s.current(), "h");
    }

    #[test]
    fn alt_enter_inserts_newline() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('a')),
                alt_enter(),
                key(KeyCode::Char('b')),
            ],
        );
        assert_eq!(s.current(), "a\nb");
        // Enter without Alt still submits the multi-line content.
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Enter),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Submit("a\nb".to_string()));
    }

    #[test]
    fn shift_enter_inserts_newline() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[
                key(KeyCode::Char('a')),
                shift_enter(),
                key(KeyCode::Char('b')),
            ],
        );
        assert_eq!(s.current(), "a\nb");
    }

    #[test]
    fn ctrl_j_inserts_newline() {
        let mut s = InputState::default();
        drive(
            &mut s,
            &[],
            &[key(KeyCode::Char('a')), ctrl_j(), key(KeyCode::Char('b'))],
        );
        assert_eq!(s.current(), "a\nb");
        // Plain Enter still submits.
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Enter),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Submit("a\nb".to_string()));
    }

    #[test]
    fn ctrl_v_returns_paste_image() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('v'), W, &cmds());
        assert_eq!(outcome, InputOutcome::PasteImage);
    }

    #[test]
    fn alt_v_returns_paste_image() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            KeyEvent::new(KeyCode::Char('v'), KeyModifiers::ALT),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::PasteImage);
    }

    #[test]
    fn ctrl_shift_v_does_not_trigger_paste_image() {
        // Regression: `contains(CONTROL)` is a bit-set check, so without
        // the SHIFT exclusion this combo would also match. Some Linux
        // terminal configurations leak Ctrl+Shift+V through to the TUI
        // instead of converting it to bracketed paste - letting it hit
        // PasteImage would surface a confusing "no image in clipboard"
        // warning on every text-paste attempt in those setups.
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            KeyEvent::new(
                KeyCode::Char('v'),
                KeyModifiers::CONTROL | KeyModifiers::SHIFT,
            ),
            W,
            &cmds(),
        );
        assert_ne!(outcome, InputOutcome::PasteImage);
    }

    #[test]
    fn plain_v_is_typed_normally() {
        // Regression: Ctrl/Alt + V triggers the paste path, but a bare 'v'
        // keypress must still insert the character into the input buffer.
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Char('v')),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.current(), "v");
    }

    #[test]
    fn shift_tab_returns_cycle_mode() {
        let mut s = InputState::default();
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::CycleMode);
    }

    // -----------------------------------------------------------------------
    // Ctrl+A / Ctrl+E tests
    // -----------------------------------------------------------------------

    #[test]
    fn ctrl_a_moves_to_start_of_unwrapped_line() {
        let mut s = InputState::default();
        for c in "hello".chars() {
            s.insert_char(c);
        }
        // Move cursor into the middle.
        drive(&mut s, &[], &[key(KeyCode::Left), key(KeyCode::Left)]);
        assert_eq!(s.cursor(), 3);
        // Ctrl+A jumps to position 0.
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), W, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn ctrl_a_already_at_start_is_noop() {
        let mut s = InputState::default();
        drive(&mut s, &[], &[key(KeyCode::Char('a'))]);
        drive(&mut s, &[], &[key(KeyCode::Home)]);
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), W, &cmds());
        assert_eq!(outcome, InputOutcome::None);
    }

    #[test]
    fn ctrl_a_on_wrapped_line_moves_to_visual_start() {
        // "abcdefghij" with width=5 wraps to: "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 7; // cursor on 'h' in the second visual line (vis_col=2)
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        // Should jump to start of second visual line: char index 5 ('f')
        assert_eq!(s.cursor(), 5);
    }

    #[test]
    fn ctrl_a_on_multiline_moves_to_visual_start() {
        // "hello\nworld" - cursor on 'r' (char index 8, visual line 1, col 2)
        let mut s = InputState::default();
        s.input = "hello\nworld".to_string();
        s.cursor = 8;
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        // Start of second visual line: char index 6 ('w')
        assert_eq!(s.cursor(), 6);
    }

    #[test]
    fn ctrl_a_chains_backwards_through_multiline() {
        // "hello\nworld" - start on second logical line, already at its start
        let mut s = InputState::default();
        s.input = "hello\nworld".to_string();
        s.cursor = 6; // start of second visual line ('w')
        let mut chord = ChordTracker::new();
        // First Ctrl+A: already at start of visual line 1, step back to line 0 start
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 0);
        // Second Ctrl+A: at start of visual line 0, nowhere to go
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::None);
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn ctrl_a_chains_backwards_through_soft_wrapped_lines() {
        // "abcdefghij" with width=5: visual lines "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 5; // start of second visual line ('f')
        let mut chord = ChordTracker::new();
        // Ctrl+A: already at start of visual line 1, step back to line 0 start
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 0);
        // Ctrl+A again: at start of visual line 0, nowhere to go
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('a'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::None);
    }

    #[test]
    fn ctrl_e_moves_to_end_of_unwrapped_line() {
        let mut s = InputState::default();
        s.input = "hello".to_string();
        s.cursor = 2;
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), W, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 5);
    }

    #[test]
    fn ctrl_e_already_at_end_is_noop() {
        let mut s = InputState::default();
        drive(&mut s, &[], &[key(KeyCode::Char('a'))]);
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), W, &cmds());
        assert_eq!(outcome, InputOutcome::None);
    }

    #[test]
    fn ctrl_e_on_wrapped_line_moves_to_visual_end() {
        // "abcdefghij" with width=5 wraps to: "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 6; // cursor on 'g' in the second visual line (vis_col=1)
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        // End of second visual line: char index 10 (end of input)
        assert_eq!(s.cursor(), 10);
    }

    #[test]
    fn ctrl_e_on_first_wrapped_line_moves_to_visual_end() {
        // "abcdefghij" with width=5 wraps to: "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 2; // cursor on first visual line
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        // End of first visual line: char index 5 ('f' start of next line)
        assert_eq!(s.cursor(), 5);
    }

    #[test]
    fn ctrl_e_on_multiline_moves_to_visual_end() {
        // "hello\nworld" - cursor at start of second line (char index 6)
        let mut s = InputState::default();
        s.input = "hello\nworld".to_string();
        s.cursor = 6;
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        // End of second visual line: char index 11 (end of input)
        assert_eq!(s.cursor(), 11);
    }

    #[test]
    fn ctrl_e_on_first_line_of_multiline_stops_before_newline() {
        // "hello\nworld" - cursor inside "hello". Ctrl+E must land at the end
        // of "hello" (char index 5, before the \n), not on the next line.
        let mut s = InputState::default();
        s.input = "hello\nworld".to_string();
        s.cursor = 2;
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 5);
    }

    #[test]
    fn ctrl_e_chains_forwards_through_multiline() {
        // "hello\nworld" - Ctrl+E goes to the end of "hello", then a second
        // press chains to the end of "world", then has nowhere left to go.
        let mut s = InputState::default();
        s.input = "hello\nworld".to_string();
        s.cursor = 2;
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 5); // end of "hello", before the \n
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 11); // chained to the end of "world"
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 80, &cmds());
        assert_eq!(outcome, InputOutcome::None);
        assert_eq!(s.cursor(), 11);
    }

    #[test]
    fn ctrl_e_chains_forwards_through_soft_wrapped_lines() {
        // "abcdefghij" with width=5: visual lines "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 2;
        let mut chord = ChordTracker::new();
        // Ctrl+E: to the end of visual line 0.
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 5);
        // Ctrl+E again: chains to the end of visual line 1.
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.cursor(), 10);
        // Ctrl+E again: at the end of the last line, nowhere to go.
        let outcome = on_key(&mut s, &[], &mut chord, false, ctrl('e'), 5, &cmds());
        assert_eq!(outcome, InputOutcome::None);
    }

    // -----------------------------------------------------------------------
    // Up/Down visual-line navigation tests
    // -----------------------------------------------------------------------

    #[test]
    fn up_on_single_line_navigates_history() {
        let mut s = InputState::default();
        let history = vec!["old".to_string()];
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &history,
            &mut chord,
            false,
            key(KeyCode::Up),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.current(), "old");
    }

    #[test]
    fn down_on_single_line_navigates_history() {
        let mut s = InputState::default();
        let history = vec!["a".to_string(), "b".to_string()];
        drive(&mut s, &history, &[key(KeyCode::Up)]);
        assert_eq!(s.current(), "b");
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &history,
            &mut chord,
            false,
            key(KeyCode::Down),
            W,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
        // Down past end restores draft (empty)
        assert_eq!(s.current(), "");
    }

    #[test]
    fn up_on_wrapped_line_moves_to_previous_visual_line() {
        // "abcdefghij" with width=5: visual lines "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 7; // 'h' on second visual line, col 2
        let mut chord = ChordTracker::new();
        let outcome = on_key(&mut s, &[], &mut chord, false, key(KeyCode::Up), 5, &cmds());
        assert_eq!(outcome, InputOutcome::Redraw);
        // Same column (2) on first visual line -> char index 2 ('c')
        assert_eq!(s.cursor(), 2);
    }

    #[test]
    fn down_on_wrapped_line_moves_to_next_visual_line() {
        // "abcdefghij" with width=5: visual lines "abcde" | "fghij"
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 2; // 'c' on first visual line, col 2
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            5,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
        // Same column (2) on second visual line -> char index 7 ('h')
        assert_eq!(s.cursor(), 7);
    }

    #[test]
    fn up_on_first_visual_line_falls_through_to_history() {
        // "abcdefghij" with width=5: cursor at start of first visual line
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 0;
        let history = vec!["old".to_string()];
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &history,
            &mut chord,
            false,
            key(KeyCode::Up),
            5,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::Redraw);
        assert_eq!(s.current(), "old");
    }

    #[test]
    fn down_on_last_visual_line_falls_through_to_history() {
        // "abcdefghij" with width=5: cursor at end (last visual line)
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 10;
        // history_next on no history_idx is a no-op -> returns false -> None
        let mut chord = ChordTracker::new();
        let outcome = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            5,
            &cmds(),
        );
        assert_eq!(outcome, InputOutcome::None);
    }

    #[test]
    fn up_down_preserves_column() {
        // "abcdefghij\nABCDEFGHIJ" - two logical lines of equal length.
        // Cursor at col 3 on second line (char index 14, 'D')
        let mut s = InputState::default();
        s.input = "abcdefghij\nABCDEFGHIJ".to_string();
        s.cursor = 14; // 'D'
        let mut chord = ChordTracker::new();
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Up),
            80,
            &cmds(),
        );
        // Col 3 on first line -> char index 3 ('d')
        assert_eq!(s.cursor(), 3);
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            80,
            &cmds(),
        );
        // Col 3 restored on second line -> char index 14 ('D')
        assert_eq!(s.cursor(), 14);
    }

    #[test]
    fn up_down_clamps_to_shorter_line() {
        // "abcdefghij\nxy" - first line 10 chars, second line 2 chars.
        let mut s = InputState::default();
        s.input = "abcdefghij\nxy".to_string();
        // Cursor at col 8 on first line (char index 8, 'i')
        s.cursor = 8;
        let mut chord = ChordTracker::new();
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            80,
            &cmds(),
        );
        // Second line is only 2 chars -> clamp to col 2 -> char index 13 (end of "xy")
        assert_eq!(s.cursor(), 13);
        // Now Up should restore desired_col=8 on the first line
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Up),
            80,
            &cmds(),
        );
        assert_eq!(s.cursor(), 8);
    }

    #[test]
    fn up_into_short_hard_wrapped_line_clamps_before_newline() {
        // "xy\nabcdefghij" - line 0 is a short, \n-terminated logical line.
        // Up from a column past its length must clamp to the end of "xy"
        // (char index 2, before the \n), not step onto the next line.
        let mut s = InputState::default();
        s.input = "xy\nabcdefghij".to_string();
        s.cursor = 9; // col 6 on line 1 ('g')
        let mut chord = ChordTracker::new();
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Up),
            80,
            &cmds(),
        );
        assert_eq!(s.cursor(), 2);
        // Down restores the remembered column on the longer line.
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            80,
            &cmds(),
        );
        assert_eq!(s.cursor(), 9);
    }

    #[test]
    fn desired_col_cleared_on_type() {
        let mut s = InputState::default();
        s.input = "abcdefghij\nxy".to_string();
        s.cursor = 8;
        let mut chord = ChordTracker::new();
        // Down sets desired_col
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            80,
            &cmds(),
        );
        assert_eq!(s.cursor(), 13);
        assert!(s.desired_col.is_some());
        // Typing clears desired_col
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Char('z')),
            80,
            &cmds(),
        );
        assert!(s.desired_col.is_none());
    }

    #[test]
    fn up_down_on_soft_wrapped_lines() {
        // "abcdefghijklmnopqrstuvwxyz" with width=10:
        // visual line 0: "abcdefghij" (chars 0-9)
        // visual line 1: "klmnopqrst" (chars 10-19)
        // visual line 2: "uvwxyz"     (chars 20-25)
        let mut s = InputState::default();
        s.input = "abcdefghijklmnopqrstuvwxyz".to_string();

        // Start at col 3 on visual line 1 (char index 13, 'n')
        s.cursor = 13;
        let mut chord = ChordTracker::new();

        // Up -> col 3 on visual line 0 (char index 3, 'd')
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Up),
            10,
            &cmds(),
        );
        assert_eq!(s.cursor(), 3);

        // Down -> col 3 on visual line 1 (char index 13, 'n')
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            10,
            &cmds(),
        );
        assert_eq!(s.cursor(), 13);

        // Down -> col 3 on visual line 2 (char index 23, 'x')
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            10,
            &cmds(),
        );
        assert_eq!(s.cursor(), 23);

        // Up -> col 3 restored on visual line 1 (char index 13)
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Up),
            10,
            &cmds(),
        );
        assert_eq!(s.cursor(), 13);
    }

    #[test]
    fn up_on_multiline_with_soft_wrap_navigates_visual_lines() {
        // "abcdefghij\nABCDEFGHIJ" with width=6:
        // visual line 0: "abcdef" (chars 0-5)
        // visual line 1: "ghij"   (chars 6-9)
        // visual line 2: "\n" -> "ABCDEF" (chars 11-16)
        // visual line 3: "GHIJ"   (chars 17-20)
        let mut s = InputState::default();
        s.input = "abcdefghij\nABCDEFGHIJ".to_string();
        // Cursor at char 18 ('H') - visual line 3, col 1
        s.cursor = 18;
        let mut chord = ChordTracker::new();
        let _ = on_key(&mut s, &[], &mut chord, false, key(KeyCode::Up), 6, &cmds());
        // Up to visual line 2, col 1 -> char 12 ('B')
        assert_eq!(s.cursor(), 12);
    }

    // -----------------------------------------------------------------------
    // Visual-line helper unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn ctrl_e_sets_row_end_affinity_other_keys_clear_it() {
        // "abcdefghij" at width 5 wraps to "abcde" | "fghij". Ctrl+E from the
        // first row lands on the boundary index 5 and asks to render at the
        // end of row 0; Ctrl+A clears the affinity so it renders at row 1's
        // start (column 0).
        let mut s = InputState::default();
        s.input = "abcdefghij".to_string();
        s.cursor = 2;
        let mut chord = ChordTracker::new();
        on_key(&mut s, &[], &mut chord, false, ctrl('e'), 5, &cmds());
        assert_eq!(s.cursor(), 5);
        assert!(s.prefer_row_end());
        // Ctrl+A back to the start of the current row clears the affinity.
        on_key(&mut s, &[], &mut chord, false, ctrl('a'), 5, &cmds());
        assert!(!s.prefer_row_end());
    }

    #[test]
    fn visual_line_info_measures_wide_glyph_columns() {
        // "俳句asd": cursor after the two kanji (char index 2) is at display
        // column 4, not char-count 2.
        let info = visual_line_info("俳句asd", 2, 80);
        assert_eq!(info.vis_line, 0);
        assert_eq!(info.vis_col, 4);
    }

    #[test]
    fn up_down_preserves_wide_glyph_column() {
        // Two logical lines whose first line leads with a wide kanji. Moving
        // down from after "句" (display col 4) lands at the matching display
        // column on the ASCII line, i.e. char index 4 of "wxyz...".
        let mut s = InputState::default();
        s.input = "句a\nwxyz".to_string();
        // Cursor after "句a" on line 0 -> display col 3 (2 + 1).
        s.cursor = 2;
        let mut chord = ChordTracker::new();
        let _ = on_key(
            &mut s,
            &[],
            &mut chord,
            false,
            key(KeyCode::Down),
            80,
            &cmds(),
        );
        // Line 1 "wxyz": display col 3 -> char index 3 + line start 3 = 6 ('z').
        assert_eq!(s.cursor(), 6);
    }

    #[test]
    fn visual_line_info_single_line_no_wrap() {
        let info = visual_line_info("hello", 3, 80);
        assert_eq!(info.vis_line, 0);
        assert_eq!(info.vis_col, 3);
        assert_eq!(info.line_starts, vec![0]);
    }

    #[test]
    fn visual_line_info_multiline_no_wrap() {
        // "abc\ndef" - a=0 b=1 c=2 \n=3 d=4 e=5 f=6
        let info = visual_line_info("abc\ndef", 6, 80);
        // cursor at 'f' (char 6), line 1, col 2
        assert_eq!(info.vis_line, 1);
        assert_eq!(info.vis_col, 2);
        assert_eq!(info.line_starts, vec![0, 4]);
    }

    #[test]
    fn visual_line_info_soft_wrap() {
        // "abcdefghij" with width=5: "abcde" | "fghij"
        let info = visual_line_info("abcdefghij", 7, 5);
        assert_eq!(info.vis_line, 1);
        assert_eq!(info.vis_col, 2); // 'h' is at col 2 on the second visual line
        assert_eq!(info.line_starts, vec![0, 5]);
    }

    #[test]
    fn visual_line_info_at_end() {
        let info = visual_line_info("abc", 3, 80);
        assert_eq!(info.vis_line, 0);
        assert_eq!(info.vis_col, 3);
        assert_eq!(info.line_starts, vec![0]);
    }

    #[test]
    fn visual_line_info_empty() {
        let info = visual_line_info("", 0, 80);
        assert_eq!(info.vis_line, 0);
        assert_eq!(info.vis_col, 0);
        assert_eq!(info.line_starts, vec![0]);
    }

    #[test]
    fn visual_line_info_full_line_before_newline_has_no_phantom_line() {
        // "abcde\nf" width 5: the first segment exactly fills the line and is
        // followed by a hard newline. The newline produces the break, so the
        // layout is "abcde" | "f" - no phantom empty visual line in between
        // (matching what `wrap_line` renders).
        let info = visual_line_info("abcde\nf", 6, 5);
        assert_eq!(info.line_starts, vec![0, 6]);
        assert_eq!(info.vis_line, 1);
        assert_eq!(info.vis_col, 0);
    }
}
