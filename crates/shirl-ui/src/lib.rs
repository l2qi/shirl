// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! ratatui-inline UI for shirl.
//!
//! The bottom ten rows of the terminal (`VIEWPORT_HEIGHT`) are an inline
//! viewport that always shows the input prompt and the status footer;
//! everything else (submitted inputs,
//! streamed assistant text, tool indicators) is appended to terminal
//! scrollback above the viewport via [`Terminal::insert_before`]. This pins
//! the status at the bottom of the visible terminal, the way Claude Code
//! does, while history scrolls naturally above.
//!
//! Streaming flushes line-by-line: content deltas are accumulated until a
//! `\n` arrives, then the completed line is inserted into scrollback. Any
//! trailing partial line is flushed at `on_turn_end`.

mod activity;
mod approval;
pub mod clipboard_image;
mod completion;
mod file_picker;
mod history;
mod input;
mod layout;
mod mention;
mod paste;
mod recap;
mod title;
pub mod transcript;

pub use completion::CommandInfo;
pub use file_picker::{FileEntry, FilePickerState};
pub use title::compute_title;

use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, Mutex as AsyncMutex};

use crossterm::event::{self, EnableBracketedPaste, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph};
use ratatui::{Terminal, TerminalOptions, Viewport};
use sweet_agent::AgentIo;
use sweet_core::{async_trait, MemoryItem, Message, PermissionMode, Role, Session, ToolCall};

use self::activity::{
    active_tools_render_count, breath_color, format_elapsed, spinner_word, spinner_word_past,
    ActiveTool, LastOutput, ACCENT,
};
use self::history::{default_history_path, History};
use self::input::{ChordTracker, InputOutcome, InputState};
pub(crate) use self::layout::{char_width, span_width};
use self::layout::{
    cursor_position, summarize_args, truncate_chars, unicode_display_width, visual_line_starts,
    wrap_line,
};
use self::mention::{mention_filter, quote_path_for_mention, splice_file_mention};
use self::paste::{handle_paste_image, sanitize_pasted_text, MUTED};
use self::recap::recap_entries;
use self::title::{detect_git_branch, short_cwd};

const HISTORY_CAPACITY: usize = 1000;
const VIEWPORT_HEIGHT: u16 = 10;
// The viewport is `VIEWPORT_HEIGHT` rows tall and bottom-aligned. Six rows are
// fixed — from the bottom up: blank, status, bottom rule, input (≥1 row,
// growing as the line wraps), top rule, working indicator. Any rows above
// hold a top spacer (which keeps the footer pinned to the bottom) and a live
// region for in-progress tool calls.

const HORIZONTAL_RULE_CHAR: char = '─';
const PROMPT_INDICATOR: &str = "› ";
const PROMPT_INDICATOR_WIDTH: u16 = 2;
const TOOL_RESULT_PREVIEW_LINES: usize = 3;

/// Maximum character count for the session title in the rule line.
const MAX_TITLE_CHARS: usize = 40;

/// Width of the picker popup as a fraction of the screen width.
const PICKER_POPUP_WIDTH_RATIO: f64 = 0.7;
/// Minimum popup width regardless of how narrow the screen is.
const PICKER_POPUP_MIN_WIDTH: u16 = 40;
/// Total width consumed by the popup border (1 char on the left + 1 on the right).
const PICKER_POPUP_BORDER_WIDTH: u16 = 2;
/// Width consumed by the per-row prefix (e.g. `"▸ "`, `"● "`, `"  "`).
const PICKER_ROW_PREFIX_WIDTH: u16 = 2;

/// Width of the picker popup for a given screen width.
pub fn picker_popup_width(screen_width: u16) -> u16 {
    ((screen_width as f64 * PICKER_POPUP_WIDTH_RATIO) as u16)
        .max(PICKER_POPUP_MIN_WIDTH)
        .min(screen_width)
}

/// Usable width for a picker row's text content (popup width minus borders
/// and the row prefix). This is the value picker layout code should target
/// when truncating columns.
pub fn picker_row_width(screen_width: u16) -> u16 {
    picker_popup_width(screen_width)
        .saturating_sub(PICKER_POPUP_BORDER_WIDTH)
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
}

/// Commands sent from the background input thread to the main loop.
#[derive(Clone, Debug)]
pub enum Command {
    Submit(String),
    Partial(String),
    SelectMove(i32),
    /// Terminal was resized while a picker was active. Picker loops should
    /// rebuild any width-dependent layout. Non-picker loops can ignore it.
    Resize,
    Cancel,
    Exit,
    /// Shift+Tab: cycle the permission mode.
    CycleMode,
    /// Direct keypress from the approval popup (e.g. 'y', 'a', 'n').
    ApprovalKey(char),
    /// Ctrl+O: toggle the full-conversation transcript view.
    ToggleTranscript,
    /// The user typed `@` (or appended characters after `@`), opening
    /// or updating the inline file-path picker. The payload is the text
    /// after the nearest `@` before the cursor — the fuzzy-search filter.
    FilePickerFilter(String),
    /// The user pressed Enter/Tab while the file picker was open, accepting
    /// the selected entry. The payload is the selected file path.
    FilePickerAccept,
    /// The user pressed Esc or Backspace-deleted the `@`, closing the file
    /// picker without accepting.
    FilePickerClose,
}

/// Status info rendered in the inline footer.
#[derive(Clone)]
pub struct StatusInfo {
    pub version: String,
    pub model: String,
    pub cwd: String,
    pub git_branch: Option<String>,
    pub context_window: Option<usize>,
    pub used: usize,
    pub mode: Option<String>,
    pub title: Option<String>,
    pub permission_mode: Option<PermissionMode>,
}

impl StatusInfo {
    pub fn new(model: String, context_window: Option<usize>) -> Self {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let git_branch = detect_git_branch(&cwd);
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model,
            cwd,
            git_branch,
            context_window,
            used: 0,
            mode: None,
            title: None,
            permission_mode: None,
        }
    }

    fn format(&self) -> Line<'static> {
        let context = match self.context_window {
            Some(max) if max > 0 => {
                let pct = self.used as f64 / max as f64 * 100.0;
                format!(
                    "{:.0}% ({:.1}k/{:.0}k)",
                    pct,
                    self.used as f64 / 1000.0,
                    max as f64 / 1000.0
                )
            }
            _ => format!("{:.1}k tokens", self.used as f64 / 1000.0),
        };
        let mode_str = match &self.mode {
            Some(m) => format!(" [{}]", m),
            None => String::new(),
        };
        let cwd_display = match &self.git_branch {
            Some(branch) => format!("{} ({})", short_cwd(&self.cwd), branch),
            None => short_cwd(&self.cwd),
        };
        let base = format!(
            "shirl {}{} · {} · {} · {}",
            self.version, mode_str, self.model, cwd_display, context
        );
        let mut spans = vec![Span::raw(base)];
        // Permission badge — appended at the end, colored, hidden for Ask.
        if let Some(mode) = &self.permission_mode {
            let (label, color) = match mode {
                PermissionMode::AutoEdit => ("accept edits on", Color::Rgb(180, 120, 220)),
                PermissionMode::FullAuto => ("auto mode on", Color::Rgb(80, 200, 120)),
                PermissionMode::Normal => ("", Color::White),
            };
            if !label.is_empty() {
                spans.push(Span::raw(" · "));
                spans.push(Span::styled(label.to_string(), Style::default().fg(color)));
            }
        }
        Line::from(spans)
    }
}

/// Render the normal chat viewport (prompt + status).
#[allow(clippy::too_many_arguments)] // pure rendering fn; splitting hurts readability
fn render_chat(
    f: &mut ratatui::Frame,
    working_since: Option<Instant>,
    spinner_seed: u64,
    status_text: &Line<'_>,
    input: &str,
    cursor: usize,
    prefer_row_end: bool,
    completion_suffix: Option<&str>,
    discovery_hint: Option<&str>,
    active_tools: &[ActiveTool],
    title: Option<&str>,
    approval: Option<&approval::ApprovalRenderState>,
    file_picker: Option<&file_picker::FilePickerState>,
) {
    let area = f.area();
    let width = area.width as usize;

    let input_width = width.saturating_sub(PROMPT_INDICATOR_WIDTH as usize);

    let (
        input_lines,
        wrapped_input,
        wrapped_hint,
        cursor_line,
        cursor_col,
        visible_start,
        max_input_lines,
    ) = if let Some(_appr) = approval {
        // Approval prompt: fixed 2 rows, no text input.
        (2u16, Vec::new(), Vec::new(), 0usize, 0usize, 0usize, 0usize)
    } else {
        let wrapped_input = wrap_line(input, input_width);
        let wrapped_hint = match discovery_hint {
            Some(hint) => wrap_line(hint, input_width),
            None => Vec::new(),
        };
        let max_input_lines = area.height.saturating_sub(5) as usize;
        let input_lines = (wrapped_input.len() + wrapped_hint.len()).min(max_input_lines) as u16;
        let (cursor_line, cursor_col) = if input_width == 0 {
            (0usize, 0usize)
        } else {
            cursor_position(input, cursor, input_width, prefer_row_end)
        };
        let visible_start = if cursor_line + 1 > max_input_lines {
            (cursor_line + 1).saturating_sub(max_input_lines)
        } else {
            0
        };
        (
            input_lines,
            wrapped_input,
            wrapped_hint,
            cursor_line,
            cursor_col,
            visible_start,
            max_input_lines,
        )
    };

    // Live region for in-progress tool calls — bounded by the free rows that
    // would otherwise be spacer.
    let fp_base = file_picker.map(|fp| fp.height()).unwrap_or(0);
    // When the file picker is open, reclaim the working row — the user is
    // idle and the breathing `⏺` serves no purpose. This gives 5 visible
    // picker rows instead of 4 without growing the viewport.
    let (fp_height, working_rows) = if fp_base > 0 {
        (fp_base + 1, 0)
    } else {
        (0, 1)
    };
    let available_for_live = area.height.saturating_sub(5 + input_lines + fp_height);
    let live_lines = active_tools_render_count(active_tools.len(), available_for_live);

    // Bottom-aligned layout: spacer pushes content down.
    let content_height = 5 + input_lines + live_lines + fp_height;
    let spacer_height = area.height.saturating_sub(content_height);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(spacer_height), // spacer
            Constraint::Length(fp_height),     // file picker
            Constraint::Length(live_lines),    // in-progress tool calls
            Constraint::Length(working_rows),  // working (0 when picker open)
            Constraint::Length(1),             // top rule
            Constraint::Length(input_lines),   // input
            Constraint::Length(1),             // bottom rule
            Constraint::Length(1),             // status
            Constraint::Length(1),             // blank
        ])
        .split(area);

    let fp_idx = 1;
    let live_idx = 2;
    let working_idx = 3;
    let top_rule_idx = 4;
    let input_idx = 5;
    let bottom_rule_idx = 6;
    let status_idx = 7;

    // Activity glyph color — shared across the working row and each
    // in-progress tool line so they breathe as one heartbeat. Body text uses
    // `ACCENT` so the dot at peak matches it exactly.
    let elapsed = working_since.map(|s| s.elapsed()).unwrap_or_default();
    let glyph_style = Style::default().fg(breath_color(elapsed));
    let body_style = Style::default().fg(ACCENT);

    // In-progress tool live region — breathing `⏺` per running tool.
    if live_lines > 0 {
        let mut lines: Vec<Line> = Vec::with_capacity(live_lines as usize);
        let visible = (live_lines as usize).min(active_tools.len());
        // If we have to truncate, reserve the last line for "(+N more)".
        let (show_full, more) = if active_tools.len() > visible {
            (
                visible.saturating_sub(1),
                active_tools.len() - (visible - 1),
            )
        } else {
            (visible, 0)
        };
        for tool in active_tools.iter().take(show_full) {
            lines.push(Line::from(vec![
                Span::styled("⏺", glyph_style),
                Span::styled(format!(" {}({})", tool.name, tool.args), body_style),
            ]));
        }
        if more > 0 {
            lines.push(Line::from(Span::styled(
                format!("  (+{} more)", more),
                Style::default().fg(MUTED),
            )));
        }
        f.render_widget(Paragraph::new(Text::from(lines)), chunks[live_idx]);
    }

    // File picker — inline list of matching paths above the input.
    if let Some(fp) = file_picker {
        file_picker::render_file_picker(f, fp, chunks[fp_idx]);
    }

    // Working indicator — `⏺` breathes; a whimsical word names the activity
    // and the elapsed-time text counts up.
    if let Some(since) = working_since {
        let elapsed = since.elapsed();
        let line = Line::from(vec![
            Span::styled("⏺", glyph_style),
            Span::styled(
                format!(
                    " {}… ({})",
                    spinner_word(spinner_seed),
                    format_elapsed(elapsed)
                ),
                body_style,
            ),
        ]);
        f.render_widget(Paragraph::new(line), chunks[working_idx]);
    }

    // Input area: either the approval prompt or the normal text input.
    if let Some(appr) = approval {
        approval::render_approval_inline(f, appr, chunks[input_idx]);
        // Hide the cursor while the approval prompt is active.
    } else {
        let prompt_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let ghost_style = Style::default().fg(MUTED);
        let visible_lines = &wrapped_input[visible_start..];
        let mut prompt_lines = Vec::new();
        for (i, line) in visible_lines.iter().enumerate() {
            let mut spans = if visible_start == 0 && i == 0 {
                vec![
                    Span::styled(PROMPT_INDICATOR, prompt_style),
                    Span::raw(line.as_str()),
                ]
            } else {
                vec![Span::raw("  "), Span::raw(line.as_str())]
            };
            let is_last_input_line = i == visible_lines.len() - 1;
            if is_last_input_line {
                if let Some(suffix) = completion_suffix {
                    spans.push(Span::styled(suffix.to_string(), ghost_style));
                }
            }
            prompt_lines.push(Line::from(spans));
        }
        if prompt_lines.is_empty() {
            prompt_lines.push(Line::from(vec![Span::styled(
                PROMPT_INDICATOR,
                prompt_style,
            )]));
        }
        let remaining_slots = max_input_lines;
        let used_slots = prompt_lines.len();
        for hint_line in wrapped_hint
            .iter()
            .take(remaining_slots.saturating_sub(used_slots))
        {
            prompt_lines.push(Line::from(Span::styled(
                format!("  {hint_line}"),
                ghost_style,
            )));
        }
        f.render_widget(Paragraph::new(Text::from(prompt_lines)), chunks[input_idx]);

        // Cursor position within the visible wrapped text.
        let visible_cursor_line = cursor_line.saturating_sub(visible_start);
        let cursor_x = chunks[input_idx].x + PROMPT_INDICATOR_WIDTH + cursor_col as u16;
        let cursor_y = chunks[input_idx].y + visible_cursor_line as u16;
        f.set_cursor_position(Position::new(cursor_x, cursor_y));
    }

    // Horizontal rules.
    // Bottom rule: always plain.
    let bottom_rule = HORIZONTAL_RULE_CHAR
        .to_string()
        .repeat(chunks[bottom_rule_idx].width as usize);
    f.render_widget(Paragraph::new(bottom_rule), chunks[bottom_rule_idx]);

    // Top rule: embed the session title if present.
    let top_rule = match title {
        Some(t) if !t.is_empty() => {
            let display = truncate_chars(t, MAX_TITLE_CHARS);
            let display_width = unicode_display_width(&display);
            let total = chunks[top_rule_idx].width as usize;
            // Layout: <left rule> + " " + title + " " + <right rule>.
            // Title ends ~15% from the right border.
            let right = (total * 15 / 100).max(2);
            let left = total.saturating_sub(display_width + 2 + right);
            let mut s = HORIZONTAL_RULE_CHAR.to_string().repeat(left);
            s.push(' ');
            s.push_str(&display);
            s.push(' ');
            s.push_str(&HORIZONTAL_RULE_CHAR.to_string().repeat(right));
            s
        }
        _ => HORIZONTAL_RULE_CHAR
            .to_string()
            .repeat(chunks[top_rule_idx].width as usize),
    };
    f.render_widget(Paragraph::new(top_rule), chunks[top_rule_idx]);

    // Status line.
    f.render_widget(Paragraph::new(status_text.clone()), chunks[status_idx]);
}

fn render_popup(
    f: &mut ratatui::Frame,
    picker: &PickerRenderState,
    input: &str,
    cursor: usize,
    scrollback: &[String],
    status_text: &Line<'_>,
) {
    let screen = f.area();

    let dim_style = Style::default().fg(MUTED);
    let max_back = screen.height as usize;
    let start = scrollback.len().saturating_sub(max_back);
    let backdrop_lines: Vec<Line> = scrollback[start..]
        .iter()
        .map(|l| Line::from(Span::styled(l.as_str(), dim_style)))
        .collect();
    let backdrop = Text::from(backdrop_lines);
    f.render_widget(Paragraph::new(backdrop), screen);

    let status_line_y = screen.height.saturating_sub(1);
    let status_area = Rect::new(0, status_line_y, screen.width, 1);
    f.render_widget(
        Paragraph::new(status_text.clone()).style(Style::default().fg(MUTED)),
        status_area,
    );

    let popup_width = picker_popup_width(screen.width);
    let popup_height = (screen.height as f64 * 0.5)
        .max(10.0)
        .min(screen.height as f64) as u16;
    let popup_x = (screen.width.saturating_sub(popup_width)) / 2;
    let popup_y = (screen.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    f.render_widget(ratatui::widgets::Clear, popup_area);

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(format!(" {} ", picker.title))
        .title_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // search
            Constraint::Min(0),    // entries
            Constraint::Length(1), // scroll indicator
            Constraint::Length(1), // hint
        ])
        .split(inner);

    let search_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    f.render_widget(
        Paragraph::new(format!("{}{}", picker.input_prefix, input)).style(search_style),
        chunks[0],
    );
    let cursor_x = chunks[0].x + picker.input_prefix.len() as u16 + cursor as u16;
    let cursor_y = chunks[0].y;
    f.set_cursor_position(Position::new(cursor_x, cursor_y));

    let entries_height = chunks[1].height as usize;
    let highlight_style = Style::default()
        .fg(Color::White)
        .bg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let header_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let current_style = Style::default().fg(Color::Green);

    let mut all_rows: Vec<(Option<String>, String, bool, bool)> = Vec::new();
    let mut abs = 0usize;
    for section in &picker.sections {
        if let Some(ref header) = section.header {
            if !all_rows.is_empty() {
                all_rows.push((None, String::new(), false, false));
            }
            all_rows.push((Some(header.clone()), String::new(), false, false));
        }
        for entry in &section.entries {
            all_rows.push((
                None,
                entry.display.clone(),
                abs == picker.selected_index,
                entry.is_current,
            ));
            abs += 1;
        }
    }

    let max_scroll = all_rows.len().saturating_sub(entries_height);
    let selected_local = all_rows
        .iter()
        .position(|(_, _, selected, _)| *selected)
        .unwrap_or(0);
    // Scroll just enough to keep the selected row visible.
    let scroll_offset = if selected_local >= entries_height {
        (selected_local + 1 - entries_height).min(max_scroll)
    } else {
        0
    };

    let mut lines = Vec::new();
    for (_i, (header, text, selected, is_current)) in all_rows
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(entries_height)
    {
        if let Some(ref h) = header {
            lines.push(Line::from(Span::styled(format!("  {h}"), header_style)));
        } else if *selected {
            lines.push(Line::from(Span::styled(
                format!("▸ {}", text),
                highlight_style,
            )));
        } else if *is_current {
            lines.push(Line::from(Span::styled(
                format!("● {}", text),
                current_style,
            )));
        } else {
            lines.push(Line::from(format!("  {}", text)));
        }
    }
    f.render_widget(Paragraph::new(Text::from(lines)), chunks[1]);

    let total_entries = picker.total_entries();
    let up = if scroll_offset > 0 { "↑" } else { " " };
    let down = if scroll_offset < max_scroll {
        "↓"
    } else {
        " "
    };
    let scroll_style = Style::default().fg(MUTED);
    let scroll_text = if total_entries > 0 {
        format!(" {up} {total_entries} {} {down}", picker.item_label)
    } else {
        String::new()
    };
    f.render_widget(Paragraph::new(scroll_text).style(scroll_style), chunks[2]);

    let hint_style = Style::default().fg(MUTED);
    f.render_widget(
        Paragraph::new(picker.hint.as_str()).style(hint_style),
        chunks[3],
    );
}

/// A single selectable entry in the picker list.
#[derive(Clone, Debug)]
pub struct PickerEntry {
    /// Fully qualified id used for selection (e.g. "anthropic/claude-sonnet-4-20250514").
    pub id: String,
    /// Display text shown in the list row.
    pub display: String,
    /// Whether this is the currently active model.
    pub is_current: bool,
}

/// A group of entries under an optional section header.
#[derive(Clone, Debug)]
pub struct PickerSection {
    /// Bold section header (provider name). `None` when filtering flattens the list.
    pub header: Option<String>,
    pub entries: Vec<PickerEntry>,
}

/// Picker rendering state for the model-selection popup.
#[derive(Clone, Debug)]
pub struct PickerRenderState {
    pub title: String,
    pub sections: Vec<PickerSection>,
    pub selected_index: usize,
    pub filter: String,
    pub hint: String,
    pub item_label: String,
    pub input_prefix: String,
}

impl PickerRenderState {
    /// Count all entries across all sections.
    pub fn total_entries(&self) -> usize {
        self.sections.iter().map(|s| s.entries.len()).sum()
    }
}

/// `AgentIo` impl built on a ratatui inline viewport.
pub struct ReplIo {
    terminal: Arc<Mutex<Terminal<CrosstermBackend<io::Stdout>>>>,
    history: History,
    status: Arc<Mutex<StatusInfo>>,
    pending_line: String,
    raw_mode: bool,
    input: InputState,
    /// `Some(start_instant)` for the duration of a turn (`on_turn_start` →
    /// `on_turn_end` / abort). Drives the elapsed-seconds counter and the
    /// in-progress tool blink. Touched only under the `ReplIo` async lock.
    working_since: Option<Instant>,
    /// Per-turn seed for the whimsical working-indicator word. Reset on each
    /// `on_turn_start` so the word varies between turns.
    spinner_seed: u64,
    /// Tool calls announced by the model but not yet completed. Rendered in
    /// the live region with a pulsing `⏺`; flushed to scrollback on
    /// `on_tool_result`.
    active_tools: Vec<ActiveTool>,
    cmd_tx: mpsc::Sender<Command>,
    pub pending_command: Option<Command>,
    pub picker: Option<PickerRenderState>,
    popup_terminal: Option<Terminal<CrosstermBackend<io::Stdout>>>,
    scrollback: Vec<String>,
    last_output: LastOutput,
    pub approval: Option<approval::ApprovalRenderState>,
    /// Inline file-path picker state. `Some` while the user is typing after `@`
    /// and the picker is showing matching files.
    pub file_picker: Option<file_picker::FilePickerState>,
    /// The full slash-command list driving autocomplete and discovery hints:
    /// built-in commands followed by any user/project-discovered custom ones.
    /// Built once at startup; `set_custom_commands` rebuilds it.
    commands: Vec<completion::CommandInfo>,
}

impl ReplIo {
    /// Splice a selected file-path into the input buffer, replacing the
    /// `@filter` token under the cursor with `@path` + trailing space.
    ///
    /// Paths containing whitespace or `"` characters are automatically
    /// quoted and escaped (`"` → `\"`, `\` → `\\`) so the `resolve_images`
    /// parser can extract them correctly.
    pub fn insert_file_mention(&mut self, path: &str) {
        let quoted = quote_path_for_mention(path);
        let current = self.input.current().to_string();
        let cursor = self.input.cursor();
        if let Some((new_input, new_cursor)) = splice_file_mention(&current, cursor, &quoted) {
            self.input.set(&new_input);
            self.input.set_cursor(new_cursor);
        }
    }

    /// Set the custom (dynamically discovered) slash commands, rebuilding the
    /// merged command list (built-in followed by custom). Call once at startup
    /// after custom commands are loaded from disk.
    pub fn set_custom_commands(&mut self, commands: Vec<completion::CommandInfo>) {
        let mut all = completion::built_in_commands();
        all.extend(commands);
        self.commands = all;
    }
}

/// Cloneable wrapper that lets the model task and main loop share a [`ReplIo`].
#[derive(Clone)]
pub struct SharedIo {
    inner: Arc<AsyncMutex<ReplIo>>,
    /// Channel for approval requests from the agent task to the main loop.
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

/// An approval request sent from the agent task (via `AgentIo::on_tool_approval`)
/// to the main loop, which shows the popup and sends the decision back.
pub struct ApprovalRequest {
    pub call: ToolCall,
    pub risk: sweet_core::ToolRisk,
    pub response_tx: tokio::sync::oneshot::Sender<sweet_core::ApprovalDecision>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl SharedIo {
    pub fn new(inner: Arc<AsyncMutex<ReplIo>>) -> (Self, mpsc::Receiver<ApprovalRequest>) {
        let (approval_tx, approval_rx) = mpsc::channel(16);
        (Self { inner, approval_tx }, approval_rx)
    }

    pub async fn lock(&self) -> tokio::sync::MutexGuard<'_, ReplIo> {
        self.inner.lock().await
    }
}

#[async_trait]
impl AgentIo for SharedIo {
    async fn read_input(&mut self) -> sweet_core::Result<Option<String>> {
        std::future::pending().await
    }

    async fn write_reply(
        &mut self,
        message: &Message,
        session: &dyn Session,
    ) -> sweet_core::Result<()> {
        let mut io = self.inner.lock().await;
        io.write_reply(message, session).await
    }

    async fn on_turn_start(&mut self) -> sweet_core::Result<()> {
        let mut io = self.inner.lock().await;
        io.on_turn_start().await
    }

    async fn on_content_delta(&mut self, delta: &str) -> sweet_core::Result<()> {
        let mut io = self.inner.lock().await;
        io.on_content_delta(delta).await
    }

    async fn on_tool_call(&mut self, call: &ToolCall) -> sweet_core::Result<()> {
        let mut io = self.inner.lock().await;
        io.on_tool_call(call).await
    }

    async fn on_tool_result(&mut self, call: &ToolCall, result: &str) -> sweet_core::Result<()> {
        let mut io = self.inner.lock().await;
        io.on_tool_result(call, result).await
    }

    async fn on_turn_end(&mut self, session: &dyn Session) -> sweet_core::Result<()> {
        let mut io = self.inner.lock().await;
        io.on_turn_end(session).await
    }

    async fn on_tool_approval(
        &mut self,
        call: &ToolCall,
        risk: sweet_core::ToolRisk,
    ) -> sweet_core::Result<sweet_core::ApprovalDecision> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                call: call.clone(),
                risk,
                response_tx,
            })
            .await
            .map_err(|e| sweet_core::Error::Io(std::io::Error::other(e.to_string())))?;
        response_rx.await.map_err(|e| {
            sweet_core::Error::Io(std::io::Error::other(format!(
                "approval response dropped: {e}"
            )))
        })
    }
}

impl ReplIo {
    pub fn new(
        model: String,
        context_window: Option<usize>,
        cmd_tx: mpsc::Sender<Command>,
    ) -> sweet_core::Result<Self> {
        enable_raw_mode().map_err(io_err)?;
        // Enable bracketed paste so multi-line paste arrives as a single
        // Event::Paste instead of individual keystrokes (including Enter,
        // which would prematurely submit).
        let _ = crossterm::execute!(io::stdout(), EnableBracketedPaste);
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Arc::new(Mutex::new(
            Terminal::with_options(
                backend,
                TerminalOptions {
                    viewport: Viewport::Inline(VIEWPORT_HEIGHT),
                },
            )
            .map_err(io_err)?,
        ));
        let history = History::load(default_history_path(), HISTORY_CAPACITY);
        let status = Arc::new(Mutex::new(StatusInfo::new(model, context_window)));
        Ok(Self {
            terminal,
            history,
            status,
            pending_line: String::new(),
            raw_mode: true,
            input: InputState::default(),
            working_since: None,
            spinner_seed: 0,
            active_tools: Vec::new(),
            cmd_tx,
            pending_command: None,
            picker: None,
            popup_terminal: None,
            scrollback: Vec::new(),
            last_output: LastOutput::Start,
            approval: None,
            file_picker: None,
            commands: completion::built_in_commands(),
        })
    }

    pub fn print_banner(&mut self, session_id: &str) -> sweet_core::Result<()> {
        self.insert_lines(&[
            String::new(),
            format!("shirl · session {session_id}"),
            String::new(),
        ])?;
        self.draw()
    }

    /// Show a compact recap of the last few messages from a resumed session.
    ///
    /// Renders the last `RESUME_MAX_MESSAGES` user/assistant messages into
    /// scrollback, truncated to `RESUME_LINES_PER_MESSAGE` lines each. Tool
    /// and system messages are skipped. The goal is to give the user enough
    /// context to recognise what they were working on without overwhelming
    /// the terminal.
    pub fn print_resumed_messages(&mut self, items: &[MemoryItem]) -> sweet_core::Result<()> {
        let (omitted, entries) = recap_entries(items);
        if entries.is_empty() {
            return Ok(());
        }

        let meta = Style::default().fg(MUTED);
        self.insert_bordered_line("── session recap", meta)?;
        if omitted > 0 {
            self.insert_bordered_line(&format!("({omitted} earlier messages omitted)"), meta)?;
        }

        let indent = " ".repeat(PROMPT_INDICATOR_WIDTH as usize);
        for entry in &entries {
            match entry.role {
                Role::User => {
                    let mut formatted = String::new();
                    for (i, line) in entry.lines.iter().enumerate() {
                        if i == 0 {
                            formatted.push_str(PROMPT_INDICATOR);
                        } else {
                            formatted.push('\n');
                            formatted.push_str(&indent);
                        }
                        formatted.push_str(line);
                    }
                    self.insert_bordered_line(&formatted, Style::default())?;
                }
                Role::Assistant => {
                    self.insert_bordered_line(&entry.lines.join("\n"), Style::default())?;
                }
                _ => continue,
            }

            if entry.omitted_lines > 0 {
                let extra = entry.omitted_lines;
                self.insert_bordered_line(&format!("({extra} more lines)"), meta)?;
            }
        }

        self.insert_bordered_line("──", meta)?;
        self.insert_styled_line("", Style::default())?;
        self.draw()
    }

    /// Clear the "working" timer. Does not touch `active_tools` — callers that
    /// want to acknowledge a cancellation should use [`Self::show_cancelled`], which
    /// flushes in-flight tools to scrollback first so they appear in natural
    /// reading order (tools, then summary).
    pub fn clear_working(&mut self) {
        self.working_since = None;
    }

    pub fn abort_cleanup(&mut self) -> sweet_core::Result<()> {
        self.working_since = None;
        self.flush_pending_all()?;
        // Defensive: if `show_cancelled` ran first the vec is already empty,
        // and this is a no-op; if not, we still leave a trace in scrollback.
        self.flush_active_tools_cancelled()?;
        self.last_output = LastOutput::Start;
        self.insert_styled_line("", Style::default())?;
        self.draw()
    }

    /// Drain `active_tools` into scrollback marked as cancelled. Used when a
    /// turn is aborted (Ctrl+C) so the in-flight tool calls leave a trace.
    fn flush_active_tools_cancelled(&mut self) -> sweet_core::Result<()> {
        if self.active_tools.is_empty() {
            return Ok(());
        }
        let tools = std::mem::take(&mut self.active_tools);
        for tool in tools {
            self.insert_styled_line("", Style::default())?;
            let line = format!("⏺ {}({})", tool.name, tool.args);
            self.insert_styled_line(&line, Style::default().fg(ACCENT))?;
            self.insert_styled_line("  ↳ (cancelled)", Style::default().fg(MUTED))?;
        }
        Ok(())
    }

    /// Acknowledge a cancelled turn. Flushes any in-flight tools to scrollback
    /// first (so the cancelled tools appear above the summary line in natural
    /// reading order), then prints the summary.
    pub fn show_cancelled(&mut self, repaired: bool) -> sweet_core::Result<()> {
        self.flush_active_tools_cancelled()?;
        let msg = if repaired {
            "⏺ Cancelled (repaired incomplete tool results)"
        } else {
            "⏺ Cancelled"
        };
        self.insert_styled_line(msg, Style::default().fg(ACCENT))?;
        self.draw()
    }

    pub fn show_session_repaired(&mut self) -> sweet_core::Result<()> {
        self.insert_styled_line(
            "⏺ Repaired incomplete tool results from a previous session",
            Style::default().fg(ACCENT),
        )?;
        self.draw()
    }

    pub fn set_git_branch(&mut self, branch: Option<String>) -> sweet_core::Result<()> {
        lock(&*self.status).git_branch = branch;
        self.draw()
    }

    pub fn set_mode(&mut self, mode: Option<String>) -> sweet_core::Result<()> {
        lock(&*self.status).mode = mode;
        self.draw()
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) -> sweet_core::Result<()> {
        lock(&*self.status).permission_mode = Some(mode);
        self.draw()
    }

    pub fn set_title(&mut self, title: String) -> sweet_core::Result<()> {
        lock(&*self.status).title = Some(title);
        self.draw()
    }

    pub fn clear_title(&mut self) -> sweet_core::Result<()> {
        lock(&*self.status).title = None;
        self.draw()
    }

    pub fn set_model(&mut self, model: String) -> sweet_core::Result<()> {
        lock(&*self.status).model = model;
        self.draw()
    }

    pub fn set_context_window(&mut self, ctx: Option<usize>) -> sweet_core::Result<()> {
        lock(&*self.status).context_window = ctx;
        self.draw()
    }

    pub fn print_resume_hint(&mut self, session_id: &str) -> sweet_core::Result<()> {
        self.insert_lines(&[
            String::new(),
            format!("Resume this session with: shirl --resume {session_id}"),
        ])
    }

    pub fn insert_lines(&mut self, lines: &[String]) -> sweet_core::Result<()> {
        let width = lock(&*self.terminal)
            .size()
            .map(|r| r.width as usize)
            .unwrap_or(80);
        let mut wrapped = Vec::new();
        for line in lines {
            if line.is_empty() {
                wrapped.push(String::new());
            } else {
                wrapped.extend(wrap_line(line, width));
            }
        }
        let count = wrapped.len() as u16;
        if count == 0 {
            return Ok(());
        }
        self.append_scrollback(&wrapped);
        lock(&*self.terminal)
            .insert_before(count, |buf| {
                for (i, line) in wrapped.iter().enumerate() {
                    buf.set_string(0, i as u16, line, Style::default());
                }
            })
            .map_err(io_err)?;
        self.draw()
    }

    /// Push one styled line into terminal scrollback above the viewport.
    ///
    /// Does NOT redraw the viewport — callers that mutate further or batch
    /// multiple inserts must end with their own `self.draw()`. The inserted
    /// line itself is already visible (it's written directly into terminal
    /// scrollback via `insert_before`); the deferred draw only updates the
    /// inline viewport's footer position.
    fn insert_styled_line(&mut self, line: &str, style: Style) -> sweet_core::Result<()> {
        let width = lock(&*self.terminal)
            .size()
            .map(|r| r.width as usize)
            .unwrap_or(80);
        let wrapped = wrap_line(line, width);
        let count = wrapped.len() as u16;
        if count == 0 {
            return Ok(());
        }
        self.append_scrollback(&wrapped);
        lock(&*self.terminal)
            .insert_before(count, |buf| {
                for (i, line) in wrapped.iter().enumerate() {
                    buf.set_string(0, i as u16, line, style);
                }
            })
            .map_err(io_err)?;
        Ok(())
    }

    /// Insert a line into scrollback with a `│ ` left border in [`MUTED`] and
    /// `content` in `style`. Wraps to the terminal width minus the 2-column
    /// border prefix.
    fn insert_bordered_line(&mut self, content: &str, style: Style) -> sweet_core::Result<()> {
        let border = "│ ";
        let border_style = Style::default().fg(MUTED);
        let width = lock(&*self.terminal)
            .size()
            .map(|r| r.width as usize)
            .unwrap_or(80);
        let content_width = width.saturating_sub(2);
        let wrapped = wrap_line(content, content_width);
        let count = wrapped.len() as u16;
        let full_lines: Vec<String> = wrapped.iter().map(|l| format!("{border}{l}")).collect();
        self.append_scrollback(&full_lines);
        lock(&*self.terminal)
            .insert_before(count, |buf| {
                for (i, line) in wrapped.iter().enumerate() {
                    buf.set_string(0, i as u16, border, border_style);
                    buf.set_string(2, i as u16, line, style);
                }
            })
            .map_err(io_err)?;
        Ok(())
    }

    fn append_scrollback(&mut self, lines: &[String]) {
        let max = 500;
        self.scrollback.extend(lines.iter().cloned());
        if self.scrollback.len() > max {
            let excess = self.scrollback.len() - max;
            self.scrollback.drain(..excess);
        }
    }

    /// Drain completed lines (those terminated by `\n`) from the streaming
    /// buffer into scrollback. Leaves any trailing partial line in place.
    fn flush_pending_completed(&mut self) -> sweet_core::Result<()> {
        while let Some(idx) = self.pending_line.find('\n') {
            let mut line: String = self.pending_line.drain(..=idx).collect();
            // Drop the trailing `\n`.
            line.pop();
            self.insert_styled_line(&line, Style::default())?;
        }
        Ok(())
    }

    /// Drain any remaining buffered streaming text (including a partial last
    /// line) into scrollback.
    fn flush_pending_all(&mut self) -> sweet_core::Result<()> {
        self.flush_pending_completed()?;
        if !self.pending_line.is_empty() {
            let line = std::mem::take(&mut self.pending_line);
            self.insert_styled_line(&line, Style::default())?;
        }
        Ok(())
    }

    fn refresh_used(&self, session: &dyn Session) {
        lock(&*self.status).used = session.context_size();
    }

    pub fn draw(&mut self) -> sweet_core::Result<()> {
        if self.popup_terminal.is_some() {
            self.draw_popup()?;
            return Ok(());
        }
        let status = lock(&*self.status);
        let status_text = status.format();
        let title = status.title.clone();
        drop(status);
        let input = self.input.current().to_string();
        let cursor = self.input.cursor();
        let prefer_row_end = self.input.prefer_row_end();
        let working_since = self.working_since;
        let spinner_seed = self.spinner_seed;
        let completion = completion::complete(&input, &self.commands);
        let hint = completion::hint(&input, &self.commands);
        let active_tools = self.active_tools.clone();
        let approval = self.approval.clone();
        let file_picker = self.file_picker.clone();
        lock(&*self.terminal)
            .draw(|f| {
                render_chat(
                    f,
                    working_since,
                    spinner_seed,
                    &status_text,
                    &input,
                    cursor,
                    prefer_row_end,
                    completion.as_deref(),
                    hint.as_deref(),
                    &active_tools,
                    title.as_deref(),
                    approval.as_ref(),
                    file_picker.as_ref(),
                );
            })
            .map_err(io_err)?;
        Ok(())
    }

    pub fn open_popup(&mut self) -> sweet_core::Result<()> {
        use crossterm::execute;
        use crossterm::terminal::EnterAlternateScreen;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen).map_err(io_err)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).map_err(io_err)?;
        terminal.clear().map_err(io_err)?;
        self.popup_terminal = Some(terminal);
        Ok(())
    }

    pub fn close_popup(&mut self) -> sweet_core::Result<()> {
        if let Some(mut term) = self.popup_terminal.take() {
            use crossterm::execute;
            use crossterm::terminal::LeaveAlternateScreen;
            let _ = term.clear();
            let mut stdout = io::stdout();
            execute!(stdout, LeaveAlternateScreen).map_err(io_err)?;
            drop(term);
        }
        self.draw()
    }

    /// Render the transcript view into the popup terminal.
    pub fn render_transcript(
        &mut self,
        view: &transcript::TranscriptView,
    ) -> sweet_core::Result<()> {
        if let Some(ref mut term) = self.popup_terminal {
            transcript::render_transcript(term, view.lines(), view.scroll_offset())?;
        }
        Ok(())
    }

    /// Number of rows the popup terminal is currently using, or `None` if
    /// the popup is closed.
    pub fn popup_height(&self) -> Option<usize> {
        self.popup_terminal
            .as_ref()
            .and_then(|t| t.size().ok())
            .map(|r| r.height as usize)
    }
}

impl ReplIo {
    /// Write a rich diff or content preview to scrollback, shown before the
    /// inline approval prompt appears in the viewport.
    ///
    /// For file-edit tools this gives the user enough context to make an
    /// informed y/n decision. Lines are rendered with color: red for `-`
    /// (removed), green for `+` (added), muted for context/headers.
    ///
    /// Errors are intentionally swallowed — the preview is cosmetic. If
    /// rendering fails the approval prompt should still appear so the user
    /// can approve or deny.
    pub fn flush_approval_preview(&mut self, preview: &sweet_core::ApprovalPreview) {
        use layout::PREVIEW_LINE_CAP;

        let header_style = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        let removed_style = Style::default().fg(Color::Red);
        let added_style = Style::default().fg(Color::Green);
        let ctx_style = Style::default().fg(MUTED);

        match preview {
            sweet_core::ApprovalPreview::None => return,
            sweet_core::ApprovalPreview::Diff {
                tool_name,
                path,
                diff,
            } => {
                let _ = self.insert_styled_line(
                    &format!("\u{2500}\u{2500}\u{2500} {tool_name}: {path}"),
                    header_style,
                );
                // The `--- ` / `+++ ` file headers are always the first two
                // lines; classify the body only after them, by diff prefix.
                // Sniffing the prefix on every line would miscolor a removed
                // line whose text starts with `--` (rendered as `---…`).
                for (i, line) in diff.lines().enumerate() {
                    let style = if i < 2 || line.starts_with("@@") {
                        ctx_style
                    } else if line.starts_with('-') {
                        removed_style
                    } else if line.starts_with('+') {
                        added_style
                    } else {
                        Style::default()
                    };
                    let _ = self.insert_styled_line(line, style);
                }
            }
            sweet_core::ApprovalPreview::NewFile { path, content } => {
                let _ = self.insert_styled_line(
                    &format!("\u{2500}\u{2500}\u{2500} write_file: {path} (new file)"),
                    header_style,
                );
                let mut lines: Vec<&str> = content.lines().collect();
                let truncated = lines.len() > PREVIEW_LINE_CAP;
                if truncated {
                    lines.truncate(PREVIEW_LINE_CAP);
                }
                // A new file is entirely new content — render every line as
                // an addition rather than sniffing diff prefixes it lacks.
                for line in &lines {
                    let _ = self.insert_styled_line(line, added_style);
                }
                if truncated {
                    let _ = self.insert_styled_line("(content truncated)", ctx_style);
                }
            }
        }

        // Blank separator after the preview.
        let _ = self.insert_styled_line("", Style::default());
    }

    /// Set the approval rendering state and redraw. The approval prompt
    /// replaces the text input area in the viewport until
    /// [`Self::clear_approval`] is called.
    pub fn set_approval(
        &mut self,
        tool_name: &str,
        risk: sweet_core::ToolRisk,
        args: &serde_json::Value,
    ) -> sweet_core::Result<()> {
        self.approval = Some(approval::ApprovalRenderState::new(tool_name, risk, args));
        self.draw()
    }

    /// Clear the approval rendering state and redraw, restoring the
    /// normal text input.
    pub fn clear_approval(&mut self) -> sweet_core::Result<()> {
        self.approval = None;
        self.draw()
    }

    fn draw_popup(&mut self) -> sweet_core::Result<()> {
        if let Some(ref picker) = self.picker {
            let picker = picker.clone();
            let input = self.input.current().to_string();
            let cursor = self.input.cursor();
            let scrollback = self.scrollback.clone();
            let status_text = lock(&*self.status).format();
            if let Some(ref mut term) = self.popup_terminal {
                term.draw(|f| {
                    render_popup(f, &picker, &input, cursor, &scrollback, &status_text);
                })
                .map_err(io_err)?;
            }
        }
        Ok(())
    }

    /// Start a background thread that reads crossterm events and sends
    /// [`Command`]s to the main loop.
    pub fn spawn_input_thread(self_arc: Arc<AsyncMutex<ReplIo>>, handle: tokio::runtime::Handle) {
        std::thread::spawn(move || {
            let mut chord = ChordTracker::new();

            loop {
                let ev = match event::read() {
                    Ok(ev) => ev,
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                };

                let mut io = handle.block_on(self_arc.lock());

                match ev {
                    Event::Key(key) => {
                        let picker_active = io.picker.is_some();
                        let approval_active = io.approval.is_some();

                        if approval_active && !picker_active {
                            let ctrl = key.modifiers.contains(event::KeyModifiers::CONTROL);
                            match key.code {
                                KeyCode::Char('y') | KeyCode::Char('Y') if !ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::ApprovalKey('y'));
                                }
                                KeyCode::Char('a') | KeyCode::Char('A') if !ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::ApprovalKey('a'));
                                }
                                KeyCode::Char('n') | KeyCode::Char('N') if !ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::ApprovalKey('n'));
                                }
                                // Esc or Ctrl+C cancels the whole turn.
                                KeyCode::Esc => {
                                    let _ = io.cmd_tx.try_send(Command::Cancel);
                                }
                                KeyCode::Char('c') if ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::Cancel);
                                }
                                _ => {}
                            }
                            continue;
                        }

                        // File-picker mode: intercept navigation and accept only
                        // when the picker has actionable entries. Reading the
                        // shared state directly (vs. re-deriving from input text)
                        // means Enter falls through to Submit when the user's
                        // filter matches nothing — they're not held hostage by
                        // an empty picker.
                        let picker_has_entries = io
                            .file_picker
                            .as_ref()
                            .is_some_and(|fp| !fp.entries.is_empty());
                        let picker_visible = io.file_picker.is_some();
                        if picker_has_entries {
                            match key.code {
                                KeyCode::Up => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(-1));
                                    continue;
                                }
                                KeyCode::Down => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(1));
                                    continue;
                                }
                                KeyCode::Enter | KeyCode::Tab => {
                                    let _ = io.cmd_tx.try_send(Command::FilePickerAccept);
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        if picker_visible {
                            // Esc / Ctrl+C close the picker even when it has no
                            // entries, but never swallow regular typing.
                            let ctrl = key.modifiers.contains(event::KeyModifiers::CONTROL);
                            match key.code {
                                KeyCode::Esc => {
                                    let _ = io.cmd_tx.try_send(Command::FilePickerClose);
                                    let _ = io.draw();
                                    continue;
                                }
                                KeyCode::Char('c') if ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::FilePickerClose);
                                    let _ = io.draw();
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        if picker_active {
                            match key.code {
                                KeyCode::Esc => {
                                    if io.input.is_empty() {
                                        let _ = io.cmd_tx.try_send(Command::Cancel);
                                    } else {
                                        io.input.clear();
                                        let _ = io.draw();
                                        let _ = io.cmd_tx.try_send(Command::Partial(String::new()));
                                    }
                                    continue;
                                }
                                KeyCode::Up => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(-1));
                                    continue;
                                }
                                KeyCode::Down => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(1));
                                    continue;
                                }
                                KeyCode::PageUp => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(-10));
                                    continue;
                                }
                                KeyCode::PageDown => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(10));
                                    continue;
                                }
                                _ => {
                                    let ctrl = key.modifiers.contains(event::KeyModifiers::CONTROL);
                                    if key.code == KeyCode::Char('c') && ctrl {
                                        let _ = io.cmd_tx.try_send(Command::Cancel);
                                        continue;
                                    }
                                    if key.code == KeyCode::Char('d') && ctrl && io.input.is_empty()
                                    {
                                        let _ = io.cmd_tx.try_send(Command::Cancel);
                                        continue;
                                    }
                                }
                            }
                        } else if io.popup_terminal.is_some() {
                            // Transcript view: route navigation keys.
                            let ctrl = key.modifiers.contains(event::KeyModifiers::CONTROL);
                            match key.code {
                                KeyCode::Char('k') if !ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(-1));
                                }
                                KeyCode::Char('j') if !ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(1));
                                }
                                KeyCode::Up => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(-1));
                                }
                                KeyCode::Down => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(1));
                                }
                                KeyCode::PageUp => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(-10));
                                }
                                KeyCode::PageDown => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(10));
                                }
                                KeyCode::Home => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(i32::MIN));
                                }
                                KeyCode::End => {
                                    let _ = io.cmd_tx.try_send(Command::SelectMove(i32::MAX));
                                }
                                KeyCode::Esc | KeyCode::Char('q') => {
                                    let _ = io.cmd_tx.try_send(Command::ToggleTranscript);
                                }
                                KeyCode::Char('c') if ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::ToggleTranscript);
                                }
                                KeyCode::Char('o') if ctrl => {
                                    let _ = io.cmd_tx.try_send(Command::ToggleTranscript);
                                }
                                _ => {}
                            }
                            continue;
                        } else if matches!(key.code, KeyCode::Esc) {
                            continue;
                        }

                        let working = io.working_since.is_some();
                        let history_snapshot = io.history.entries().to_vec();
                        let content_width = lock(&*io.terminal)
                            .size()
                            .map(|r| {
                                (r.width as usize).saturating_sub(PROMPT_INDICATOR_WIDTH as usize)
                            })
                            .unwrap_or(78);
                        // Reborrow as a plain `&mut ReplIo` so `input` and
                        // `commands` can be borrowed as disjoint fields (the
                        // MutexGuard's Deref would otherwise conflict).
                        let io_mut: &mut ReplIo = &mut io;
                        let outcome = input::on_key(
                            &mut io_mut.input,
                            &history_snapshot,
                            &mut chord,
                            working,
                            key,
                            content_width,
                            &io_mut.commands,
                        );
                        match outcome {
                            InputOutcome::Submit(line) => {
                                // Echo submitted text to scrollback.
                                // First line gets the › prompt indicator;
                                // continuation lines are indented to align.
                                let mut iter = line.split('\n');
                                if let Some(first) = iter.next() {
                                    let _ = io.insert_styled_line(
                                        &format!("{PROMPT_INDICATOR}{first}"),
                                        Style::default(),
                                    );
                                }
                                let indent = " ".repeat(PROMPT_INDICATOR_WIDTH as usize);
                                for cont in iter {
                                    let _ = io.insert_styled_line(
                                        &format!("{indent}{cont}"),
                                        Style::default(),
                                    );
                                }
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    io.history.push(trimmed.to_string());
                                }
                                // Submitting clears the buffer, so any open
                                // file-picker is now stale — drop it before
                                // redrawing.
                                io.file_picker = None;
                                let _ = io.draw();
                                let _ = io.cmd_tx.try_send(Command::Submit(line));
                            }
                            InputOutcome::Redraw => {
                                if picker_active {
                                    let current = io.input.current().to_string();
                                    let _ = io.cmd_tx.try_send(Command::Partial(current));
                                }
                                // Detect @-mention trigger for the file picker.
                                // The input thread detects picker mode from the
                                // text — no cross-thread state needed.
                                match mention_filter(io.input.current(), io.input.cursor()) {
                                    Some(filter) => {
                                        let _ =
                                            io.cmd_tx.try_send(Command::FilePickerFilter(filter));
                                    }
                                    None => {
                                        // No `@` trigger — close any open picker.
                                        // This handles backspacing over the `@`.
                                        let _ = io.cmd_tx.try_send(Command::FilePickerClose);
                                    }
                                }
                                let _ = io.draw();
                            }
                            InputOutcome::Cancel => {
                                let _ = io.cmd_tx.try_send(Command::Cancel);
                            }
                            InputOutcome::Exit => {
                                let _ = io.cmd_tx.try_send(Command::Exit);
                                return;
                            }
                            InputOutcome::CycleMode => {
                                let _ = io.cmd_tx.try_send(Command::CycleMode);
                            }
                            InputOutcome::ToggleTranscript => {
                                let _ = io.cmd_tx.try_send(Command::ToggleTranscript);
                            }
                            InputOutcome::PasteImage => {
                                handle_paste_image(&mut io);
                            }
                            InputOutcome::None => {}
                        }
                    }
                    Event::Paste(text) => {
                        // Bracketed paste: insert the entire text, but
                        // normalize line endings (Windows clipboards use
                        // \r\n; some sources use bare \r) and drop other
                        // ASCII control chars that would corrupt the
                        // buffer or terminal state.
                        let cleaned = sanitize_pasted_text(&text);
                        io.input.insert_str(&cleaned);
                        let _ = io.draw();
                    }
                    Event::Resize(_, _) => {
                        let picker_active = io.picker.is_some();
                        let _ = io.draw();
                        if picker_active {
                            let _ = io.cmd_tx.try_send(Command::Resize);
                        }
                    }
                    _ => {}
                }
            }
        });
    }
}

impl Drop for ReplIo {
    fn drop(&mut self) {
        if self.raw_mode {
            // Best-effort cleanup; ignore errors during shutdown.
            use crossterm::event::DisableBracketedPaste;
            let _ = crossterm::execute!(io::stdout(), DisableBracketedPaste);
            let _ = lock(&*self.terminal).clear();
            let _ = disable_raw_mode();
        }
    }
}

#[async_trait]
impl AgentIo for ReplIo {
    async fn read_input(&mut self) -> sweet_core::Result<Option<String>> {
        std::future::pending().await
    }

    async fn write_reply(
        &mut self,
        message: &Message,
        session: &dyn Session,
    ) -> sweet_core::Result<()> {
        // Used by the runloop for command/system messages (e.g. /new feedback).
        // Streamed assistant replies arrive via on_content_delta instead.
        let text = message.text_content();
        if !text.is_empty() {
            for line in text.lines() {
                self.insert_styled_line(line, Style::default().fg(Color::Cyan))?;
            }
        }
        self.refresh_used(session);
        self.draw()?;
        Ok(())
    }

    async fn on_turn_start(&mut self) -> sweet_core::Result<()> {
        self.last_output = LastOutput::Start;
        self.working_since = Some(Instant::now());
        // Vary the whimsical word between turns. Sub-second nanos are plenty
        // of entropy for picking a starting word — no need to pull in `rand`.
        self.spinner_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        self.draw()?;
        Ok(())
    }

    async fn on_content_delta(&mut self, delta: &str) -> sweet_core::Result<()> {
        // Note: we deliberately do NOT clear `working_since` here. A turn
        // routinely streams text → calls tools → streams more, so the
        // indicator must stay alive until `on_turn_end`.
        let mut dirty = false;
        if matches!(self.last_output, LastOutput::Start | LastOutput::ToolResult) {
            self.insert_styled_line("", Style::default())?;
            dirty = true;
        }
        self.last_output = LastOutput::Content;
        self.pending_line.push_str(delta);
        // Only completed lines (terminated by `\n`) are flushed to scrollback;
        // partial text stays in `pending_line` until the next `\n`, a tool
        // call, or turn-end flushes it. The 150 ms redraw tick repaints the
        // viewport but does not surface buffered partial text.
        if delta.contains('\n') {
            self.flush_pending_completed()?;
            dirty = true;
        }
        if dirty {
            self.draw()?;
        }
        Ok(())
    }

    async fn on_tool_call(&mut self, call: &ToolCall) -> sweet_core::Result<()> {
        self.flush_pending_all()?;
        self.last_output = LastOutput::ToolCall;
        let args = summarize_args(&call.arguments);
        self.active_tools.push(ActiveTool {
            id: call.id.clone(),
            name: call.name.clone(),
            args,
        });
        // Held in the viewport live region — drawn (with pulsing ⏺) by the
        // main loop's tick. Flushed to scrollback when its result arrives.
        self.draw()?;
        Ok(())
    }

    async fn on_tool_result(&mut self, call: &ToolCall, result: &str) -> sweet_core::Result<()> {
        // Remove the matching active tool. Dispatch is sequential and
        // in-order, so it is almost always at index 0; match by id for safety.
        let removed = self
            .active_tools
            .iter()
            .position(|t| t.id == call.id)
            .map(|i| self.active_tools.remove(i));

        // Blank separator before the completed tool's block.
        self.insert_styled_line("", Style::default())?;
        let (name, args) = match removed {
            Some(t) => (t.name, t.args),
            // No active entry (e.g. mid-session resume): fall back to the call.
            None => (call.name.clone(), summarize_args(&call.arguments)),
        };
        let line = format!("⏺ {}({})", name, args);
        self.insert_styled_line(&line, Style::default().fg(ACCENT))?;

        self.last_output = LastOutput::ToolResult;
        for line in result.lines().take(TOOL_RESULT_PREVIEW_LINES) {
            self.insert_styled_line(&format!("  ↳ {}", line.trim_end()), Style::default())?;
        }
        let extra = result
            .lines()
            .count()
            .saturating_sub(TOOL_RESULT_PREVIEW_LINES);
        if extra > 0 {
            let s = format!(
                "  ({} more line{})",
                extra,
                if extra == 1 { "" } else { "s" }
            );
            self.insert_styled_line(&s, Style::default())?;
        }
        // Redraw so the live region updates (one fewer active tool).
        self.draw()?;
        Ok(())
    }

    async fn on_turn_end(&mut self, session: &dyn Session) -> sweet_core::Result<()> {
        self.last_output = LastOutput::Start;
        // Summarize the turn with the same whimsical word in past tense, e.g.
        // "Sparkled for 1m 3s." Capture the elapsed time before clearing.
        let summary = self.working_since.map(|since| {
            format!(
                "{} for {}.",
                spinner_word_past(self.spinner_seed),
                format_elapsed(since.elapsed())
            )
        });
        self.working_since = None;
        // Defensively clear — should already be empty in the normal path.
        self.active_tools.clear();
        self.flush_pending_all()?;
        // Blank separator between assistant response and the summary.
        self.insert_styled_line("", Style::default())?;
        if let Some(summary) = summary {
            self.insert_styled_line(&summary, Style::default().fg(MUTED))?;
            // Blank line between the summary and the next `›` prompt.
            self.insert_styled_line("", Style::default())?;
        }
        self.refresh_used(session);
        self.draw()?;
        Ok(())
    }
}

pub(crate) fn io_err<E: std::fmt::Display>(e: E) -> sweet_core::Error {
    sweet_core::Error::Io(io::Error::other(e.to_string()))
}
