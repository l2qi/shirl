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

mod approval;
pub mod clipboard_image;
mod completion;
mod file_picker;
mod history;
mod input;
pub mod transcript;

pub use completion::CommandInfo;
pub use file_picker::{FileEntry, FilePickerState};

use std::fmt::Write;
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
use sweet_core::{
    async_trait, MemoryItem, Message, Model, PermissionMode, Role, Session, ToolCall,
};
use unicode_width::UnicodeWidthChar;

use self::history::{default_history_path, History};
use self::input::{ChordTracker, InputOutcome, InputState};

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
const TOOL_ARGS_PREVIEW_CHARS: usize = 80;
/// Maximum number of user/assistant messages shown in the resumed-session recap.
const RESUME_MAX_MESSAGES: usize = 10;
/// Maximum lines rendered per message in the resumed-session recap.
const RESUME_LINES_PER_MESSAGE: usize = 3;
/// Maximum number of preview lines before truncation.
const PREVIEW_LINE_CAP: usize = 200;
/// Full breath cycle of the activity `⏺` (dim → bright → dim), in milliseconds.
/// ~2.5 s reads as a calm "ThinkPad suspend LED" breath; not too eager.
const BREATH_PERIOD_MS: u128 = 2500;
/// Shared muted color for secondary UI text (overflow hints, ghost text,
/// scroll indicators, cancelled labels). Chosen for ~4.5:1 contrast on
/// dark backgrounds (~#1e1e1e) and ~3.5:1 on light (~#ffffff), readable
/// on both themes without being visually prominent.
const MUTED: Color = Color::Rgb(120, 120, 120);
/// Shared accent color for activity indicators: the working row body, the
/// in-progress tool body, the solid `⏺` on completed/cancelled tool lines,
/// and the bright endpoint of the breathing `⏺`. Using a single truecolor
/// value (instead of palette `Color::Yellow`) makes the dot at peak exactly
/// match the surrounding text regardless of terminal theme.
const ACCENT: Color = Color::Rgb(217, 119, 87);

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

/// Detect the current git branch name for the given working directory.
/// Returns `None` if `git` is unavailable, the directory is not a repo,
/// or HEAD is detached.
///
/// Shells out to `git` synchronously — call only at startup or in response
/// to explicit user action, never in a hot loop.
fn detect_git_branch(cwd: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    // "HEAD" means detached HEAD — nothing useful to display.
    if branch.is_empty() || branch == "HEAD" {
        return None;
    }
    Some(branch)
}

fn short_cwd(cwd: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy().into_owned();
        if let Some(rest) = cwd.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    cwd.to_string()
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

#[derive(Default, PartialEq)]
enum LastOutput {
    #[default]
    Start,
    ToolCall,
    ToolResult,
    Content,
}

/// One tool call currently in flight — held in the redrawable viewport so its
/// `⏺` can pulse. Removed and flushed to scrollback when its result arrives.
#[derive(Clone, Debug)]
pub(crate) struct ActiveTool {
    pub id: String,
    pub name: String,
    pub args: String,
}

/// Format a `Duration` as a compact elapsed-time string: `5s`, `1m 35s`,
/// `1h 2m 3s`, `1d 1h 2m 3s`. Only non-zero larger units are shown; seconds
/// are always present.
fn format_elapsed(d: Duration) -> String {
    let total_secs = d.as_secs();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    let mut s = String::with_capacity(16); // "1d 1h 2m 3s" worst case
    if days > 0 {
        write!(s, "{}d ", days).unwrap();
    }
    if hours > 0 {
        write!(s, "{}h ", hours).unwrap();
    }
    if mins > 0 {
        write!(s, "{}m ", mins).unwrap();
    }
    write!(s, "{}s", secs).unwrap();
    s
}

/// Breath phase in `[0.0, 1.0]` for the activity `⏺`: 0 = dimmest, 1 =
/// brightest. Cosine-shaped so the transitions are smooth at both endpoints
/// (no hard step at the cycle boundary).
fn breath_phase(elapsed: Duration) -> f32 {
    let t = (elapsed.as_millis() % BREATH_PERIOD_MS) as f32 / BREATH_PERIOD_MS as f32;
    // (1 - cos(2π·t)) / 2 — 0 at t=0, 1 at t=0.5, 0 at t=1.
    (1.0 - (t * std::f32::consts::TAU).cos()) * 0.5
}

/// Color of the activity `⏺` at this point in the breath. Lerps from a
/// perceptual mid-grey at the trough up to [`ACCENT`] at the peak. The
/// mid-grey is theme-agnostic — neither a stark dark dot on a light
/// background nor a harsh light dot on a dark one — at the cost of the dot
/// never quite vanishing (terminals have no foreground alpha, so a true fade
/// to the terminal background is impossible without querying it).
fn breath_color(elapsed: Duration) -> Color {
    let t = breath_phase(elapsed);
    let r = lerp_u8(90, 217, t);
    let g = lerp_u8(90, 119, t);
    let b = lerp_u8(90, 87, t);
    Color::Rgb(r, g, b)
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let f = a as f32 + (b as f32 - a as f32) * t;
    // Round (not truncate) so that f32 `cos(π)` precision doesn't land the
    // peak at 254 instead of 255.
    f.round().clamp(0.0, 255.0) as u8
}

/// Whimsical activity words for the working indicator — Shirl's answer to
/// Claude Code's spinner verbs. Cute and a little sparkly, with `Shirling` as
/// the self-referential wink (Claude has `Clauding`). Static and curated: the
/// indicator just picks one, never the model.
///
/// Each entry is `(present participle, simple past)`. The live indicator shows
/// the present (`Sparkling…`); the end-of-turn summary reuses the same word in
/// past tense (`Sparkled for 3s.`). Both forms are stored rather than derived
/// because English past tense is irregular (`Weaving` → `Wove`).
const SPINNER_WORDS: &[(&str, &str)] = &[
    ("Sparkling", "Sparkled"),
    ("Twirling", "Twirled"),
    ("Shimmering", "Shimmered"),
    ("Daydreaming", "Daydreamed"),
    ("Doodling", "Doodled"),
    ("Blooming", "Bloomed"),
    ("Swirling", "Swirled"),
    ("Sprinkling", "Sprinkled"),
    ("Glittering", "Glittered"),
    ("Whisking", "Whisked"),
    ("Conjuring", "Conjured"),
    ("Noodling", "Noodled"),
    ("Musing", "Mused"),
    ("Brewing", "Brewed"),
    ("Tinkering", "Tinkered"),
    ("Flourishing", "Flourished"),
    ("Enchanting", "Enchanted"),
    ("Weaving", "Wove"),
    ("Humming", "Hummed"),
    ("Pondering", "Pondered"),
    ("Dreaming", "Dreamed"),
    ("Wondering", "Wondered"),
    ("Frolicking", "Frolicked"),
    ("Marinating", "Marinated"),
    ("Stargazing", "Stargazed"),
    ("Untangling", "Untangled"),
    ("Imagining", "Imagined"),
    ("Fluttering", "Fluttered"),
    ("Bedazzling", "Bedazzled"),
    ("Shirling", "Shirled"),
];

/// The present-participle word for the live indicator. `seed` is chosen once
/// per turn, so the word stays fixed for the whole turn and a fresh one is
/// picked on the next.
fn spinner_word(seed: u64) -> &'static str {
    SPINNER_WORDS[(seed % SPINNER_WORDS.len() as u64) as usize].0
}

/// The simple-past form of the same word [`spinner_word`] picked for `seed` —
/// used for the end-of-turn summary so it matches the word shown while working.
fn spinner_word_past(seed: u64) -> &'static str {
    SPINNER_WORDS[(seed % SPINNER_WORDS.len() as u64) as usize].1
}

/// How many rows the live tool region should occupy given `count` active tools
/// and `available` free rows. Caps at `available`; renders all when it fits.
fn active_tools_render_count(count: usize, available: u16) -> u16 {
    let avail = available as usize;
    if count == 0 || avail == 0 {
        return 0;
    }
    if count <= avail {
        count as u16
    } else {
        avail as u16
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

/// Quote and escape a path for insertion after `@`.
///
/// If the path contains whitespace or `"` characters, wrap it in double
/// quotes and escape any internal `"` as `\"` and `\` as `\\`. Otherwise
/// return the path as-is (no quoting needed).
fn quote_path_for_mention(path: &str) -> String {
    let needs_quoting = path.contains(|c: char| c.is_whitespace() || c == '"');
    if !needs_quoting {
        return path.to_string();
    }
    let mut out = String::with_capacity(path.len() + 4);
    out.push('"');
    for ch in path.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// Pure splice of `@path ` into `input` at the cursor, replacing the
/// `@filter` token under the cursor. Returns `None` if there's no `@`
/// before the cursor to anchor onto. Cursor and the returned cursor are
/// **char** indices.
fn splice_file_mention(input: &str, cursor: usize, path: &str) -> Option<(String, usize)> {
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    if cursor == 0 || cursor > chars.len() {
        return None;
    }
    // Find the `@` char before the cursor (char index).
    let mut at_char = cursor;
    while at_char > 0 {
        at_char -= 1;
        if chars[at_char].1 == '@' {
            break;
        }
    }
    if chars.get(at_char).map(|(_, c)| *c) != Some('@') {
        return None;
    }

    let at_byte = chars[at_char].0;
    let cursor_byte = if cursor < chars.len() {
        chars[cursor].0
    } else {
        input.len()
    };

    let replacement = format!("@{} ", path);
    let new_cursor = at_char + replacement.chars().count();
    let new_input = format!(
        "{}{}{}",
        &input[..at_byte],
        replacement,
        &input[cursor_byte..]
    );
    Some((new_input, new_cursor))
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

fn summarize_args(args: &serde_json::Value) -> String {
    let s = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
    truncate_chars(&s, TOOL_ARGS_PREVIEW_CHARS)
}

pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Char index at which each visual line begins, given soft-wrap `width`
/// (terminal display columns) and hard `\n` breaks. The first entry is always
/// `0`; `width == 0` disables soft-wrapping so only hard newlines break.
///
/// Single source of truth for line layout: [`wrap_line`], [`cursor_position`],
/// and the input editor's visual navigation all derive their boundaries from
/// it, so the cursor never lands where the text isn't drawn. A character is
/// placed on the current line when it fits (`col + width(ch) <= width`); a
/// character that would overflow begins a new line. A leading character on an
/// empty line is always placed, so a glyph wider than the whole line still
/// renders. A `\n` ends its line and occupies no cell.
pub(crate) fn visual_line_starts(text: &str, width: usize) -> Vec<usize> {
    let mut starts = vec![0];
    if width == 0 {
        for (i, ch) in text.chars().enumerate() {
            if ch == '\n' {
                starts.push(i + 1);
            }
        }
        return starts;
    }
    let mut col = 0;
    for (i, ch) in text.chars().enumerate() {
        if ch == '\n' {
            col = 0;
            starts.push(i + 1);
        } else {
            let w = char_width(ch);
            if col > 0 && col + w > width {
                starts.push(i);
                col = w;
            } else {
                col += w;
            }
        }
    }
    starts
}

/// Sum of display widths of the chars in `text` from char index `from`
/// (inclusive) to `to` (exclusive). Used to convert a char span into a column
/// offset within a visual line.
pub(crate) fn span_width(text: &str, from: usize, to: usize) -> usize {
    text.chars()
        .skip(from)
        .take(to - from)
        .map(char_width)
        .sum()
}

/// Compute the (line, column) position of a cursor within text that is
/// soft-wrapped at `width` columns and hard-wrapped at `\n` characters.
///
/// Columns are measured in terminal display cells (a CJK glyph or emoji is
/// two cells wide), so the rendered cursor lands on the same cell ratatui
/// draws the character into.
///
/// A cursor sitting exactly on a *soft*-wrap boundary is ambiguous: the index
/// is both the end of one visual row and the start of the next. `prefer_row_end`
/// (set by `Ctrl+E`) renders it at the right edge of the row it closes; the
/// default renders it at column 0 of the next row, which is what typing,
/// `Ctrl+A`, and the arrows want. A *hard* newline is unambiguous — the cursor
/// always belongs on the new row.
fn cursor_position(
    text: &str,
    cursor_char_idx: usize,
    width: usize,
    prefer_row_end: bool,
) -> (usize, usize) {
    let starts = visual_line_starts(text, width);
    let mut line = starts
        .iter()
        .rposition(|&s| s <= cursor_char_idx)
        .unwrap_or(0);
    if prefer_row_end
        && line > 0
        && starts[line] == cursor_char_idx
        && text.chars().nth(cursor_char_idx - 1) != Some('\n')
    {
        line -= 1;
    }
    (line, span_width(text, starts[line], cursor_char_idx))
}

/// Scan backward from `cursor` in `input` looking for a `@` that begins a
/// mention token. Returns the text between `@` and cursor (the fuzzy-search
/// filter), or `None` if no trigger is active.
///
/// A `@` only counts as a trigger when it sits at the start of input or
/// directly after whitespace — this avoids false triggers mid-token (e.g.
/// while typing `name@host`).
///
/// `cursor` is a **char** index (from `InputState::cursor`).
fn mention_filter(input: &str, cursor: usize) -> Option<String> {
    if cursor == 0 {
        return None;
    }
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    if cursor > chars.len() {
        return None;
    }
    // Walk backward from cursor looking for an `@` preceded by start-of-input
    // or whitespace. Stop at any whitespace inside the token.
    let mut i = cursor;
    while i > 0 {
        i -= 1;
        let ch = chars[i].1;
        if ch == '@' {
            let preceded_by_boundary = i == 0 || matches!(chars[i - 1].1, ' ' | '\t' | '\n');
            if !preceded_by_boundary {
                return None;
            }
            let start_byte = chars[i].0 + '@'.len_utf8();
            let end_byte = if cursor < chars.len() {
                chars[cursor].0
            } else {
                input.len()
            };
            return Some(input[start_byte..end_byte].to_string());
        }
        if matches!(ch, ' ' | '\t' | '\n') {
            return None;
        }
    }
    None
}

/// Wrap text to fit within `width` columns. Splits on explicit `\n`
/// newlines first, then wraps each resulting segment at character boundaries.
/// One message prepared for the resumed-session recap.
struct RecapEntry {
    /// `User` or `Assistant` — the recap skips every other role.
    role: Role,
    /// The first [`RESUME_LINES_PER_MESSAGE`] lines of the message content.
    lines: Vec<String>,
    /// How many further content lines were dropped after `lines`.
    omitted_lines: usize,
}

/// Select and truncate the messages shown in a resumed-session recap.
///
/// Keeps the last [`RESUME_MAX_MESSAGES`] non-empty user/assistant messages,
/// each truncated to [`RESUME_LINES_PER_MESSAGE`] lines. Returns the count of
/// older messages dropped from the front alongside the per-message entries.
fn recap_entries(items: &[MemoryItem]) -> (usize, Vec<RecapEntry>) {
    let messages: Vec<&Message> = items
        .iter()
        .filter_map(|item| match item {
            MemoryItem::Message(msg)
                if matches!(msg.role, Role::User | Role::Assistant)
                    && (!msg.text_content().is_empty() || msg.has_images()) =>
            {
                Some(msg)
            }
            _ => None,
        })
        .collect();

    let omitted = messages.len().saturating_sub(RESUME_MAX_MESSAGES);
    let entries = messages[omitted..]
        .iter()
        .map(|msg| {
            // Use Display (includes image placeholders like
            // "[image: image/png, 1 KB]") so image-only messages render.
            let text = msg.to_string();
            let all: Vec<&str> = text.lines().collect();
            let shown = all.len().min(RESUME_LINES_PER_MESSAGE);
            RecapEntry {
                role: msg.role,
                lines: all[..shown].iter().map(|s| (*s).to_string()).collect(),
                omitted_lines: all.len() - shown,
            }
        })
        .collect();

    (omitted, entries)
}

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return line.split('\n').map(String::from).collect();
    }
    let chars: Vec<char> = line.chars().collect();
    let starts = visual_line_starts(line, width);
    let mut lines = Vec::with_capacity(starts.len());
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(chars.len());
        // A hard newline ends its visual line but is not drawn — drop it.
        let slice_end = if end > start && chars[end - 1] == '\n' {
            end - 1
        } else {
            end
        };
        lines.push(chars[start..slice_end].iter().collect());
    }
    lines
}

/// Handle [`InputOutcome::PasteImage`]: read an image from the system
/// clipboard, persist it under `~/.shirl/cache/clipboard/`, and splice the
/// resulting `@"path"` token into the input buffer at the cursor. The
/// existing `image_input::resolve_images` pass in shirl-cli picks the
/// `@"..."` token up at submit time and turns it into a `ContentBlock::Image`.
///
/// Failures (no image present, no clipboard backend, decode error) surface
/// as a single muted line in scrollback so the user knows the paste was
/// seen but produced nothing.
fn handle_paste_image(io: &mut ReplIo) {
    let muted = Style::default().fg(MUTED);
    let warn = |io: &mut ReplIo, msg: &str| {
        let _ = io.insert_styled_line(msg, muted);
        let _ = io.draw();
    };

    let bytes = match clipboard_image::read_clipboard_png() {
        Ok(bytes) => bytes,
        Err(clipboard_image::ClipboardImageError::NoImage) => {
            warn(io, "⚠ No image in clipboard");
            return;
        }
        Err(clipboard_image::ClipboardImageError::NoClipboard) => {
            warn(io, "⚠ No clipboard backend available");
            return;
        }
        Err(clipboard_image::ClipboardImageError::Backend(msg)) => {
            warn(io, &format!("⚠ Clipboard backend error: {msg}"));
            return;
        }
        Err(clipboard_image::ClipboardImageError::Unsupported) => {
            warn(io, "⚠ Clipboard image paste is not enabled in this build");
            return;
        }
        Err(err) => {
            warn(io, &format!("⚠ Clipboard read failed: {err}"));
            return;
        }
    };

    let Some(dir) = clipboard_image::default_cache_dir() else {
        warn(io, "⚠ Could not resolve clipboard cache dir");
        return;
    };
    let path = match clipboard_image::save_to_dir(&dir, &bytes) {
        Ok(p) => p,
        Err(err) => {
            warn(io, &format!("⚠ Could not save clipboard image: {err}"));
            return;
        }
    };

    let mention = quote_path_for_mention(&path.to_string_lossy());
    let token = format!("@{mention} ");
    io.input.insert_str(&token);
    let _ = io.draw();
}

/// Normalize text arriving via bracketed paste: collapse `\r\n` and bare
/// `\r` to `\n`, then drop ASCII control chars other than `\n` and `\t`.
/// Bracketed paste already filters most terminal escapes, but Windows-
/// formatted clipboards routinely include CR and stray controls would
/// corrupt the input buffer or terminal state.
fn sanitize_pasted_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    normalized
        .chars()
        .filter(|&c| c == '\n' || c == '\t' || !c.is_control())
        .collect()
}

/// Display width of a single character in terminal columns, using the same
/// `unicode-width` tables ratatui uses to lay out buffer cells. Control
/// characters and other zero-width code points report 0.
pub(crate) fn char_width(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

/// Display width of a string in terminal columns. Matches the cell count
/// ratatui reserves when it renders the string, so wrapping and cursor
/// placement stay aligned with what is drawn.
fn unicode_display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Ask the model for a short session title given the conversation so far.
/// Returns `Ok(Some(title))` on success, `Ok(None)` when the conversation is
/// too short to title or the model reply sanitizes to an empty string.
///
/// Pure compute — no UI state is touched, so callers can spawn this without
/// holding the IO mutex across the model round-trip.
pub async fn compute_title(
    model: &Arc<dyn Model>,
    items: &[sweet_core::MemoryItem],
) -> sweet_core::Result<Option<String>> {
    if items.len() < 2 {
        return Ok(None);
    }
    let mut context = String::new();
    for item in items.iter().take(8) {
        let sweet_core::MemoryItem::Message(msg) = item;
        if msg.role == sweet_core::Role::Tool {
            continue;
        }
        let preview: String = msg.text_content().chars().take(200).collect();
        context.push_str(&format!("{}: {}\n", msg.role.as_str(), preview));
    }
    let prompt = format!(
        "Generate a short (3-6 word) title for this coding session based on the \
         conversation below. Return ONLY the title text, nothing else. \
         No quotes, no punctuation at the end.\n\n{}",
        context
    );
    let reply = model
        .complete(&[sweet_core::Message::user(prompt)], &[])
        .await?;
    let title = sanitize_title(&reply.text_content());
    Ok(if title.is_empty() { None } else { Some(title) })
}

/// Clean up a model-generated title: trim, strip wrapping quotes, drop
/// trailing punctuation, collapse to the first line. Models routinely return
/// `"Fixed bug."` or multi-line responses despite the prompt asking otherwise.
fn sanitize_title(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or("").trim();
    let unquoted = first_line
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| {
            first_line
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
        })
        .unwrap_or(first_line);
    unquoted
        .trim_end_matches(['.', '!', '?', ',', ';', ':'])
        .trim()
        .to_string()
}

pub(crate) fn io_err<E: std::fmt::Display>(e: E) -> sweet_core::Error {
    sweet_core::Error::Io(io::Error::other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_filter_basic_trigger() {
        assert_eq!(mention_filter("@foo", 4), Some("foo".to_string()));
        assert_eq!(mention_filter("hello @bar", 10), Some("bar".to_string()));
        // Just `@` with cursor right after — empty filter, picker opens.
        assert_eq!(mention_filter("@", 1), Some(String::new()));
    }

    #[test]
    fn mention_filter_skips_mid_token() {
        // `name@host` is not a mention — `@` must follow whitespace or SOL.
        assert_eq!(mention_filter("name@host", 9), None);
        assert_eq!(mention_filter("foo@bar.com", 11), None);
    }

    #[test]
    fn mention_filter_stops_at_whitespace_after_at() {
        assert_eq!(mention_filter("@foo bar", 8), None);
        assert_eq!(mention_filter("@foo\tbar", 8), None);
    }

    #[test]
    fn mention_filter_boundary_cursor() {
        // Cursor at 0 — no trigger.
        assert_eq!(mention_filter("@foo", 0), None);
        // Cursor past end — no trigger (guard against bad callers).
        assert_eq!(mention_filter("@foo", 99), None);
    }

    #[test]
    fn mention_filter_utf8_filter() {
        // Multi-byte chars in the filter must slice safely.
        assert_eq!(mention_filter("@ñ", 2), Some("ñ".to_string()));
        assert_eq!(mention_filter("hi @café", 8), Some("café".to_string()));
    }

    #[test]
    fn splice_file_mention_replaces_filter_token() {
        let (out, cur) = splice_file_mention("@ma", 3, "src/main.rs").expect("splice");
        assert_eq!(out, "@src/main.rs ");
        assert_eq!(cur, "@src/main.rs ".chars().count());
    }

    #[test]
    fn splice_file_mention_preserves_surrounding_text() {
        let (out, _) = splice_file_mention("hello @ma world", 9, "src/main.rs").expect("splice");
        // The replacement spans @ to cursor; existing trailing text is kept verbatim.
        assert_eq!(out, "hello @src/main.rs  world");
    }

    #[test]
    fn splice_file_mention_utf8_path() {
        let (out, cur) = splice_file_mention("@c", 2, "café/menu.md").expect("splice");
        assert_eq!(out, "@café/menu.md ");
        assert_eq!(cur, "@café/menu.md ".chars().count());
    }

    #[test]
    fn splice_file_mention_no_anchor_returns_none() {
        assert!(splice_file_mention("hello", 5, "x").is_none());
        // Cursor at 0 — no `@` to anchor onto.
        assert!(splice_file_mention("@foo", 0, "x").is_none());
    }

    #[test]
    fn quote_path_for_mention_simple_path() {
        assert_eq!(quote_path_for_mention("src/main.rs"), "src/main.rs");
    }

    #[test]
    fn quote_path_for_mention_path_with_spaces() {
        assert_eq!(
            quote_path_for_mention("my screenshots/photo.png"),
            r#""my screenshots/photo.png""#
        );
    }

    #[test]
    fn quote_path_for_mention_path_with_quotes() {
        assert_eq!(
            quote_path_for_mention(r#"she said "hi".png"#),
            r#""she said \"hi\".png""#
        );
    }

    #[test]
    fn quote_path_for_mention_path_with_backslash() {
        // Backslashes alone don't trigger quoting — they're not ambiguous
        // in the whitespace-delimited @token parser. Only whitespace and
        // double-quotes trigger quoting.
        assert_eq!(
            quote_path_for_mention(r"path\to\file.png"),
            r"path\to\file.png"
        );
    }

    #[test]
    fn quote_path_for_mention_spaces_and_backslash() {
        // When quoting IS triggered (by spaces), backslashes are escaped.
        assert_eq!(
            quote_path_for_mention(r"path\to\file with spaces.png"),
            r#""path\\to\\file with spaces.png""#
        );
    }

    #[test]
    fn breath_phase_smooth_cycle() {
        let period = BREATH_PERIOD_MS as u64;
        // Dimmest at the cycle boundary.
        assert!(breath_phase(Duration::from_millis(0)) < 0.01);
        assert!(breath_phase(Duration::from_millis(period)) < 0.01);
        // Brightest at the midpoint.
        assert!(breath_phase(Duration::from_millis(period / 2)) > 0.99);
        // Quarter and three-quarter points sit near 0.5 (cosine cross).
        let q = breath_phase(Duration::from_millis(period / 4));
        let tq = breath_phase(Duration::from_millis(period * 3 / 4));
        assert!((q - 0.5).abs() < 0.02);
        assert!((tq - 0.5).abs() < 0.02);
        // Stays in range across many cycles.
        for ms in [0u64, 100, 500, 1234, 5000, 12345, 999_999] {
            let v = breath_phase(Duration::from_millis(ms));
            assert!((0.0..=1.0).contains(&v), "phase {v} out of range at {ms}ms");
        }
    }

    #[test]
    fn breath_color_endpoints_match_lerp() {
        let dim = breath_color(Duration::from_millis(0));
        let bright = breath_color(Duration::from_millis(BREATH_PERIOD_MS as u64 / 2));
        // Dim: theme-agnostic perceptual mid-grey.
        assert_eq!(dim, Color::Rgb(90, 90, 90));
        // Peak: exactly the accent, so the dot matches the static text.
        assert_eq!(bright, Color::Rgb(217, 119, 87));
        assert_eq!(bright, ACCENT);
    }

    #[test]
    fn active_tools_render_count_fits_or_truncates() {
        // No tools or no room → nothing.
        assert_eq!(active_tools_render_count(0, 4), 0);
        assert_eq!(active_tools_render_count(3, 0), 0);
        // Fits exactly: render all.
        assert_eq!(active_tools_render_count(3, 4), 3);
        assert_eq!(active_tools_render_count(4, 4), 4);
        // Overflows: caps at available (caller reserves last row for "+N more").
        assert_eq!(active_tools_render_count(7, 4), 4);
    }

    #[test]
    fn sanitize_title_strips_common_model_decorations() {
        // Plain title passes through.
        assert_eq!(sanitize_title("Fix login bug"), "Fix login bug");
        // Surrounding quotes stripped.
        assert_eq!(sanitize_title("\"Fix login bug\""), "Fix login bug");
        assert_eq!(sanitize_title("'Fix login bug'"), "Fix login bug");
        // Trailing punctuation stripped.
        assert_eq!(sanitize_title("Fix login bug."), "Fix login bug");
        assert_eq!(sanitize_title("Fix login bug!"), "Fix login bug");
        // Combination: quotes + trailing punct.
        assert_eq!(sanitize_title("\"Fix login bug.\""), "Fix login bug");
        // Whitespace trimmed.
        assert_eq!(sanitize_title("  Fix login bug  "), "Fix login bug");
        // Multi-line: only first line kept.
        assert_eq!(
            sanitize_title("Fix login bug\nExtra explanation"),
            "Fix login bug"
        );
        // Empty input stays empty.
        assert_eq!(sanitize_title(""), "");
        assert_eq!(sanitize_title("   "), "");
    }

    fn recap_items(msgs: Vec<Message>) -> Vec<MemoryItem> {
        msgs.into_iter().map(MemoryItem::Message).collect()
    }

    #[test]
    fn recap_entries_skips_empty_and_non_conversational() {
        let items = recap_items(vec![
            Message::system("system prompt"),
            Message::user("hello"),
            Message::assistant(""),
            Message::tool_result("call-1", "tool output"),
            Message::assistant("hi there"),
        ]);
        let (omitted, entries) = recap_entries(&items);
        assert_eq!(omitted, 0);
        // Only the non-empty user + assistant messages survive.
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].role, Role::User);
        assert_eq!(entries[0].lines, vec!["hello"]);
        assert_eq!(entries[1].role, Role::Assistant);
        assert_eq!(entries[1].lines, vec!["hi there"]);
    }

    #[test]
    fn recap_entries_keeps_only_last_max_messages() {
        let msgs: Vec<Message> = (0..RESUME_MAX_MESSAGES + 4)
            .map(|i| Message::user(format!("msg {i}")))
            .collect();
        let items = recap_items(msgs);
        let (omitted, entries) = recap_entries(&items);
        assert_eq!(omitted, 4);
        assert_eq!(entries.len(), RESUME_MAX_MESSAGES);
        // The recap window starts after the four dropped messages.
        assert_eq!(entries[0].lines, vec!["msg 4"]);
    }

    #[test]
    fn recap_entries_truncates_long_messages() {
        let body: String = (0..RESUME_LINES_PER_MESSAGE + 2)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let items = recap_items(vec![Message::assistant(body)]);
        let (_, entries) = recap_entries(&items);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lines.len(), RESUME_LINES_PER_MESSAGE);
        assert_eq!(entries[0].omitted_lines, 2);
    }

    #[test]
    fn recap_entries_empty_session_yields_nothing() {
        let (omitted, entries) = recap_entries(&[]);
        assert_eq!(omitted, 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn wrap_line_splits_on_newlines() {
        // Two short lines — no wrapping needed.
        let lines = wrap_line("hello\nworld", 80);
        assert_eq!(lines, vec!["hello", "world"]);

        // Three lines with an empty one in the middle.
        let lines = wrap_line("a\n\nb", 80);
        assert_eq!(lines, vec!["a", "", "b"]);

        // Long line with embedded newline wraps the long part.
        let long = "abcdefghij".repeat(10);
        let input = format!("first\n{long}");
        let lines = wrap_line(&input, 40);
        assert_eq!(lines[0], "first");
        // The long segment wraps at 40 chars.
        assert_eq!(lines[1].chars().count(), 40);
        assert_eq!(lines[2].chars().count(), 40);
        assert_eq!(lines[3].chars().count(), 20);

        // Empty string produces a single empty line.
        let lines = wrap_line("", 80);
        assert_eq!(lines, vec![""]);
    }

    #[test]
    fn cursor_at_end_of_full_soft_wrapped_line_stays_on_line() {
        // Text fills the line exactly with no trailing chars — cursor must
        // stay at col=width on the current line, not advance to a non-
        // existent next line (which would render below the input area).
        assert_eq!(cursor_position("abcdefghij", 10, 10, false), (0, 10));
        assert_eq!(cursor_position("a", 1, 1, false), (0, 1));

        // When another char follows, the soft wrap is real — by default the
        // cursor moves to the start of the next line.
        assert_eq!(cursor_position("abcdefghijk", 10, 10, false), (1, 0));

        // Hard newline always wraps, even at end of text.
        assert_eq!(cursor_position("a\n", 2, 80, false), (1, 0));

        // A full line immediately followed by a hard newline must not add a
        // phantom row: "abcde\nf" at width 5 lays out as "abcde" | "f", so
        // the char after the newline is on line 1, not line 2.
        assert_eq!(cursor_position("abcde\nf", 6, 5, false), (1, 0));

        // Cursor in the middle, before any wrap.
        assert_eq!(cursor_position("abc", 2, 80, false), (0, 2));

        // Cursor across an explicit newline.
        assert_eq!(cursor_position("a\nb", 3, 80, false), (1, 1));
    }

    #[test]
    fn spinner_word_is_fixed_per_seed() {
        // A given seed always maps to the same word (stable for the turn).
        assert_eq!(spinner_word(3), spinner_word(3));
        // Different seeds generally pick different words.
        assert_ne!(spinner_word(0), spinner_word(1));
        // Any seed lands inside the list, including huge ones (no panic).
        assert!(SPINNER_WORDS.iter().any(|w| w.0 == spinner_word(u64::MAX)));
        // Shirl gets her wink, in both tenses.
        assert!(SPINNER_WORDS.contains(&("Shirling", "Shirled")));
    }

    #[test]
    fn spinner_word_past_matches_present_for_same_seed() {
        // The summary word must be the past tense of the very word shown while
        // working — both keyed by the same seed.
        for seed in 0..SPINNER_WORDS.len() as u64 {
            let present = spinner_word(seed);
            let past = spinner_word_past(seed);
            assert!(SPINNER_WORDS.contains(&(present, past)));
        }
    }

    #[test]
    fn cursor_affinity_at_soft_wrap_boundary() {
        // The boundary index 10 closes row 0 and opens row 1. Ctrl+E parks
        // there with prefer_row_end=true and must render at the row-0 edge;
        // the default (Ctrl+A, typing) renders at the start of row 1.
        assert_eq!(cursor_position("abcdefghijk", 10, 10, true), (0, 10));
        assert_eq!(cursor_position("abcdefghijk", 10, 10, false), (1, 0));

        // A hard newline ignores the affinity — the cursor is always on the
        // new row.
        assert_eq!(cursor_position("abcde\nf", 6, 5, true), (1, 0));

        // Affinity only matters on a boundary; one column in it is irrelevant.
        assert_eq!(cursor_position("abcdefghijk", 11, 10, true), (1, 1));

        // Wide glyphs: two kanji fill a width-5 row, the third opens row 1.
        assert_eq!(cursor_position("一二三四", 2, 5, true), (0, 4));
        assert_eq!(cursor_position("一二三四", 2, 5, false), (1, 0));
    }

    #[test]
    fn cursor_position_counts_wide_glyphs_as_two_columns() {
        // Reported bug: "俳句asd" — each kanji is two display cells, so the
        // cursor at the end sits at column 7 (2+2+1+1+1), not char-count 5.
        assert_eq!(cursor_position("俳句asd", 5, 80, false), (0, 7));
        // Cursor between the two kanji is at column 2.
        assert_eq!(cursor_position("俳句asd", 1, 80, false), (0, 2));
        // An emoji is also two cells wide.
        assert_eq!(cursor_position("a😀b", 2, 80, false), (0, 3));
    }

    #[test]
    fn wide_glyphs_soft_wrap_on_column_width() {
        // Four kanji at width 5: two fit per visual line (2+2=4, a third
        // would overflow to column 6).
        let lines = wrap_line("一二三四", 5);
        assert_eq!(lines, vec!["一二", "三四"]);
        // The cursor layout agrees: the third kanji starts visual line 1.
        assert_eq!(cursor_position("一二三四", 2, 5, false), (1, 0));
        // A glyph wider than the whole line still renders (never wrapped to
        // a phantom empty line).
        assert_eq!(wrap_line("一", 1), vec!["一"]);
    }

    #[test]
    fn unicode_display_width_measures_cells_not_chars() {
        assert_eq!(unicode_display_width("abc"), 3);
        assert_eq!(unicode_display_width("俳句"), 4);
        assert_eq!(unicode_display_width("a句"), 3);
    }

    #[test]
    fn sanitize_pasted_text_normalizes_line_endings() {
        // Windows CRLF collapses to LF.
        assert_eq!(sanitize_pasted_text("hello\r\nworld"), "hello\nworld");
        // Bare CR (classic Mac / stray) collapses to LF.
        assert_eq!(sanitize_pasted_text("hello\rworld"), "hello\nworld");
        // Mixed input.
        assert_eq!(sanitize_pasted_text("a\r\nb\rc\nd"), "a\nb\nc\nd");
        // \n and \t pass through; other control chars (BEL, ESC) are dropped.
        assert_eq!(
            sanitize_pasted_text("ok\ttab\nline\x07bell\x1bend"),
            "ok\ttab\nlinebellend"
        );
        // Plain text untouched.
        assert_eq!(sanitize_pasted_text("plain text"), "plain text");
    }

    #[test]
    fn format_elapsed_basic() {
        assert_eq!(format_elapsed(Duration::from_secs(0)), "0s");
        assert_eq!(format_elapsed(Duration::from_secs(5)), "5s");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
    }

    #[test]
    fn format_elapsed_minutes() {
        assert_eq!(format_elapsed(Duration::from_secs(60)), "1m 0s");
        assert_eq!(format_elapsed(Duration::from_secs(95)), "1m 35s");
        assert_eq!(format_elapsed(Duration::from_secs(3599)), "59m 59s");
    }

    #[test]
    fn format_elapsed_hours() {
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "1h 0s");
        assert_eq!(format_elapsed(Duration::from_secs(3723)), "1h 2m 3s");
    }

    #[test]
    fn format_elapsed_days() {
        assert_eq!(format_elapsed(Duration::from_secs(86400)), "1d 0s");
        assert_eq!(format_elapsed(Duration::from_secs(90123)), "1d 1h 2m 3s");
    }
}
