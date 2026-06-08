// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Inline file-path picker rendering.
//!
//! Renders a floating list of file-path matches directly above the input line
//! inside the inline viewport — no alternate screen, no popup.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::MUTED;

/// Maximum number of visible entries in the picker list.
pub const MAX_VISIBLE: usize = 5;

/// A single selectable entry in the file picker.
///
/// The display form encodes directory-ness with a trailing `/`; `is_dir`
/// is the source of truth and `path` is rendered verbatim.
#[derive(Clone, Debug)]
pub struct FileEntry {
    /// Path as it should be inserted into the input buffer. Directories
    /// carry a trailing `/`.
    pub path: String,
    /// True if this entry represents a directory.
    pub is_dir: bool,
}

/// Rendering state for the inline file picker.
#[derive(Clone, Debug)]
pub struct FilePickerState {
    /// All matching entries.
    pub entries: Vec<FileEntry>,
    /// Index into `entries` of the currently highlighted item.
    pub selected: usize,
    /// The user's current filter text (after the `@`).
    pub filter: String,
    /// Scroll offset — the index of the first visible entry.
    pub scroll: usize,
}

impl FilePickerState {
    pub fn new(filter: String, entries: Vec<FileEntry>) -> Self {
        Self {
            entries,
            selected: 0,
            filter,
            scroll: 0,
        }
    }

    /// Number of rows the picker needs to render (capped at MAX_VISIBLE).
    /// When the picker is open with no matches we still reserve 1 row so
    /// the user gets feedback rather than silent dismissal.
    pub fn height(&self) -> u16 {
        if self.entries.is_empty() {
            1
        } else {
            self.entries.len().min(MAX_VISIBLE) as u16
        }
    }

    /// Move the selection by `delta`, clamping and adjusting scroll.
    pub fn move_selection(&mut self, delta: i32) {
        if self.entries.is_empty() {
            return;
        }
        let max = self.entries.len() - 1;
        let new = (self.selected as i32 + delta).clamp(0, max as i32) as usize;
        self.selected = new;
        // Keep selection visible.
        if new < self.scroll {
            self.scroll = new;
        } else if new >= self.scroll + MAX_VISIBLE {
            self.scroll = new - MAX_VISIBLE + 1;
        }
    }

    /// The currently selected entry, if any.
    pub fn selected_entry(&self) -> Option<&FileEntry> {
        self.entries.get(self.selected)
    }
}

/// Render the file picker as a list of entries above the input area.
///
/// Takes the full-width area the picker should occupy. Each row shows a
/// file/directory path, with the selected row highlighted.
pub fn render_file_picker(f: &mut ratatui::Frame, state: &FilePickerState, area: Rect) {
    if area.height == 0 {
        return;
    }
    if state.entries.is_empty() {
        let hint = Line::from(Span::styled(
            "  no matching files",
            Style::default().fg(MUTED),
        ));
        f.render_widget(Paragraph::new(hint), area);
        return;
    }

    let visible_count = state
        .entries
        .len()
        .min(MAX_VISIBLE)
        .min(area.height as usize);
    let visible = &state.entries[state.scroll..state.scroll + visible_count];

    // Two-space leading indent so paths align with the input row; no glyph.
    const INDENT: &str = "  ";
    let lines: Vec<Line> = visible
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let abs_idx = state.scroll + i;
            let is_selected = abs_idx == state.selected;
            let (fg, bg) = if is_selected {
                (Color::White, Color::DarkGray)
            } else if entry.is_dir {
                (Color::Cyan, Color::default())
            } else {
                (Color::default(), Color::default())
            };
            let path_style = Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD);
            let mut spans = vec![
                Span::styled(INDENT, Style::default().bg(bg)),
                Span::styled(&entry.path, path_style),
            ];
            if is_selected {
                // Pad to full width for highlight bar effect.
                let used = INDENT.chars().count() + entry.path.chars().count();
                let padding = (area.width as usize).saturating_sub(used);
                spans.push(Span::styled(" ".repeat(padding), Style::default().bg(bg)));
            }
            Line::from(spans)
        })
        .collect();

    f.render_widget(Paragraph::new(lines), area);
}
