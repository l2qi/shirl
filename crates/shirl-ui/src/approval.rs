// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Inline approval rendering for the permission system.
//!
//! Renders directly into the viewport's input-area chunk — no popup, no
//! alternate screen. Two rows: a summary line and a key-hint line.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use sweet_core::permission::{approval_scope, ToolRisk};

/// State for the currently displayed approval prompt.
#[derive(Clone, Debug)]
pub struct ApprovalRenderState {
    pub tool_name: String,
    pub risk: ToolRisk,
    /// The call's scope — bash command, file path, or serialized args — the
    /// same value used as the session-approval key, so the prompt shows
    /// exactly what an "Always" grant will cover.
    pub preview: String,
}

impl ApprovalRenderState {
    pub fn new(tool_name: &str, risk: ToolRisk, args: &serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            risk,
            preview: approval_scope(args),
        }
    }
}

fn truncate_preview(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

fn risk_label(risk: ToolRisk) -> (&'static str, Color) {
    match risk {
        ToolRisk::ReadOnly => ("read-only", Color::Green),
        ToolRisk::FileWrite => ("file-write", Color::Yellow),
        ToolRisk::Dangerous => ("dangerous", Color::Red),
    }
}

/// Render the approval prompt inline into the given area (the input chunk).
///
/// Requires at least 2 rows. Row 0: tool + risk + preview. Row 1: key hints.
pub fn render_approval_inline(f: &mut ratatui::Frame, state: &ApprovalRenderState, area: Rect) {
    let (risk_text, risk_color) = risk_label(state.risk);

    // Truncate preview to fit the available width after the tool/risk prefix.
    // Prefix: "⚠ tool_name · risk_label  " ≈ tool_name.len() + risk_text.len() + 7
    let prefix_overhead = state.tool_name.chars().count() + risk_text.chars().count() + 7;
    let available = (area.width as usize).saturating_sub(prefix_overhead);
    let preview = truncate_preview(&state.preview, available);

    let row1 = Line::from(vec![
        Span::styled("⚠ ", Style::default().fg(risk_color)),
        Span::styled(
            &state.tool_name,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" · "),
        Span::styled(
            risk_text,
            Style::default().fg(risk_color).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(preview, Style::default().fg(super::MUTED)),
    ]);

    let row2 = Line::from(vec![
        Span::styled("[y] Yes", Style::default().fg(Color::Green)),
        Span::styled("  ", Style::default()),
        Span::styled("[a] Always (session)", Style::default().fg(Color::Yellow)),
        Span::styled("  ", Style::default()),
        Span::styled("[n] No", Style::default().fg(Color::Red)),
        Span::styled("  ", Style::default()),
        Span::styled("[Esc] Cancel", Style::default().fg(super::MUTED)),
    ]);

    let lines = vec![row1, row2];
    f.render_widget(Paragraph::new(lines), area);
}
