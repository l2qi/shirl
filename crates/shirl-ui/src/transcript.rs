// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Full-conversation transcript view (Ctrl+O).
//!
//! Opens an alternate-screen popup that shows the entire conversation
//! history read from the session store. User messages, assistant replies,
//! tool calls, and full (untruncated) tool results are all visible.
//! The view scrolls with arrow keys, j/k, PgUp/PgDn, and Home/End.

use std::io;

use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

use sweet_core::{MemoryItem, Message, Role};

use super::{io_err, truncate_chars};

/// Two rows reserved at the bottom of the popup: one for the horizontal rule
/// and one for the footer hint.
const FOOTER_ROWS: usize = 2;

/// Accent color matching the parent module's `ACCENT`.
const ACCENT: Color = Color::Rgb(217, 119, 87);

/// Maximum characters per line before truncation. Lines are pre-truncated
/// so that 1 logical line = 1 visual row and scroll offsets stay aligned.
const MAX_LINE_WIDTH: usize = 300;

/// Build styled lines for the full transcript.
///
/// Each output [`Line`] corresponds to exactly one visual row (no wrapping).
/// Long lines are truncated to a fixed width so the scroll offset, footer
/// position indicator, and viewport fill stay consistent.
fn build_transcript_lines(items: &[MemoryItem]) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for item in items {
        match item {
            MemoryItem::Message(Message {
                role,
                content,
                tool_calls,
                ..
            }) => {
                // Use Display so image blocks render as placeholders.
                let text = content
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join("");
                append_message_lines(&mut lines, role, &text, tool_calls)
            }
        }
    }

    lines
}

fn append_message_lines(
    lines: &mut Vec<Line<'static>>,
    role: &Role,
    content: &str,
    tool_calls: &[sweet_core::ToolCall],
) {
    match role {
        Role::User => {
            lines.push(Line::from(""));
            for line in content.lines() {
                lines.push(Line::from(Span::styled(
                    truncate_chars(&format!("› {line}"), MAX_LINE_WIDTH),
                    Style::default().add_modifier(Modifier::BOLD),
                )));
            }
        }
        Role::Assistant => {
            if !content.is_empty() {
                lines.push(Line::from(""));
                for line in content.lines() {
                    lines.push(Line::from(Span::from(truncate_chars(line, MAX_LINE_WIDTH))));
                }
            }
            for tc in tool_calls {
                let args_str = serde_json::to_string(&tc.arguments)
                    .unwrap_or_else(|_| tc.arguments.to_string());
                let args_preview = truncate_chars(&args_str, 80);
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    truncate_chars(&format!("⏺ {}({})", tc.name, args_preview), MAX_LINE_WIDTH),
                    Style::default().fg(ACCENT),
                )));
            }
        }
        Role::Tool => {
            for line in content.lines() {
                lines.push(Line::from(Span::styled(
                    truncate_chars(&format!("  ↳ {}", line), MAX_LINE_WIDTH),
                    Style::default().fg(super::MUTED),
                )));
            }
        }
        Role::System => {}
    }
}

/// Render the transcript into the popup terminal.
pub fn render_transcript(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    lines: &[Line<'static>],
    scroll_offset: usize,
) -> sweet_core::Result<()> {
    let (width, height) = terminal
        .size()
        .map(|r| (r.width, r.height as usize))
        .unwrap_or((80, 24));

    let content_height = content_height(height);
    let visible_count = lines
        .len()
        .saturating_sub(scroll_offset)
        .min(content_height);
    let mut widget_lines: Vec<Line<'_>> =
        lines[scroll_offset..scroll_offset + visible_count].to_vec();

    // Pad with blank lines if content is shorter than the viewport.
    for _ in 0..content_height.saturating_sub(visible_count) {
        widget_lines.push(Line::from(""));
    }

    // Horizontal rule separator above the footer.
    widget_lines.push(Line::from(Span::styled(
        "─".repeat(width as usize),
        Style::default().fg(super::MUTED),
    )));

    // Footer hint.
    widget_lines.push(Line::from(Span::styled(
        format!(
            " {}",
            footer_hint(scroll_offset, visible_count, lines.len())
        ),
        Style::default().fg(super::MUTED),
    )));

    let paragraph = Paragraph::new(widget_lines).block(Block::default().borders(Borders::NONE));

    terminal
        .draw(|f| {
            let area = Rect::new(0, 0, width, height as u16);
            f.render_widget(paragraph, area);
        })
        .map_err(io_err)?;

    Ok(())
}

/// Number of rows available for transcript content given the full popup height.
///
/// One row is reserved at the bottom for the footer hint.
fn content_height(popup_height: usize) -> usize {
    popup_height.saturating_sub(FOOTER_ROWS)
}

/// Clamp a scroll offset so the last line of content stays visible.
///
/// `content_viewport` is the number of rows available for content
/// (i.e. popup height minus the footer). Passing `usize::MAX` for
/// `offset` snaps to the bottom.
fn clamp_scroll(offset: usize, total: usize, content_viewport: usize) -> usize {
    let max = total.saturating_sub(content_viewport);
    offset.min(max)
}

/// In-memory state for an open transcript view.
///
/// Holds the pre-built styled lines and the current scroll offset so the
/// caller can drive scroll updates without re-building the line list on
/// every keystroke.
pub struct TranscriptView {
    lines: Vec<Line<'static>>,
    scroll_offset: usize,
}

impl TranscriptView {
    /// Build a fresh view from session items, with the scroll snapped to
    /// the bottom (most recent messages) given the supplied popup height.
    pub fn new(items: &[MemoryItem], popup_height: usize) -> Self {
        let lines = build_transcript_lines(items);
        let viewport = content_height(popup_height);
        let scroll_offset = clamp_scroll(usize::MAX, lines.len(), viewport);
        Self {
            lines,
            scroll_offset,
        }
    }

    /// Apply a signed scroll delta. `i32::MIN` snaps to the top,
    /// `i32::MAX` snaps to the bottom.
    pub fn scroll(&mut self, delta: i32, popup_height: usize) {
        let viewport = content_height(popup_height);
        let new_offset = match delta.cmp(&0) {
            std::cmp::Ordering::Less => {
                let abs = (delta as i64).unsigned_abs() as usize;
                self.scroll_offset.saturating_sub(abs)
            }
            std::cmp::Ordering::Greater => self.scroll_offset.saturating_add(delta as usize),
            std::cmp::Ordering::Equal => self.scroll_offset,
        };
        self.scroll_offset = clamp_scroll(new_offset, self.lines.len(), viewport);
    }

    pub fn lines(&self) -> &[Line<'static>] {
        &self.lines
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset
    }
}

fn footer_hint(scroll_offset: usize, visible_count: usize, total: usize) -> String {
    if total == 0 {
        return "Empty session · Esc to close".to_string();
    }
    let last_visible = scroll_offset + visible_count;
    let pct = (last_visible * 100) / total;
    format!(
        "Line {}-{} of {} ({}%) · ↑↓jk PgUp/PgDn scroll · q/Esc/Ctrl+O to close",
        scroll_offset + 1,
        last_visible,
        total,
        pct.min(100)
    )
}
