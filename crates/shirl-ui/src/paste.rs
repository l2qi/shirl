// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Clipboard image paste and bracketed-paste text sanitization.

use ratatui::style::{Color, Style};

use crate::mention::quote_path_for_mention;

/// Shared muted color for secondary UI text (overflow hints, ghost text,
/// scroll indicators, cancelled labels). Chosen for ~4.5:1 contrast on
/// dark backgrounds (~#1e1e1e) and ~3.5:1 on light (~#ffffff), readable
/// on both themes without being visually prominent.
pub(crate) const MUTED: Color = Color::Rgb(120, 120, 120);

/// Handle clipboard image paste: read an image from the system clipboard,
/// persist it under `~/.shirl/cache/clipboard/`, and splice the
/// resulting `@"path"` token into the input buffer at the cursor. The
/// existing `image_input::resolve_images` pass in shirl-cli picks the
/// `@"..."` token up at submit time and turns it into a `ContentBlock::Image`.
///
/// Failures (no image present, no clipboard backend, decode error) surface
/// as a single muted line in scrollback so the user knows the paste was
/// seen but produced nothing.
pub(crate) fn handle_paste_image(io: &mut crate::ReplIo) {
    let muted = Style::default().fg(MUTED);
    let warn = |io: &mut crate::ReplIo, msg: &str| {
        let _ = io.insert_styled_line(msg, muted);
        let _ = io.draw();
    };

    let bytes = match crate::clipboard_image::read_clipboard_png() {
        Ok(bytes) => bytes,
        Err(crate::clipboard_image::ClipboardImageError::NoImage) => {
            warn(io, "⚠ No image in clipboard");
            return;
        }
        Err(crate::clipboard_image::ClipboardImageError::NoClipboard) => {
            warn(io, "⚠ No clipboard backend available");
            return;
        }
        Err(crate::clipboard_image::ClipboardImageError::Backend(msg)) => {
            warn(io, &format!("⚠ Clipboard backend error: {msg}"));
            return;
        }
        Err(crate::clipboard_image::ClipboardImageError::Unsupported) => {
            warn(io, "⚠ Clipboard image paste is not enabled in this build");
            return;
        }
        Err(err) => {
            warn(io, &format!("⚠ Clipboard read failed: {err}"));
            return;
        }
    };

    let Some(dir) = crate::clipboard_image::default_cache_dir() else {
        warn(io, "⚠ Could not resolve clipboard cache dir");
        return;
    };
    let path = match crate::clipboard_image::save_to_dir(&dir, &bytes) {
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
pub(crate) fn sanitize_pasted_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    normalized
        .chars()
        .filter(|&c| c == '\n' || c == '\t' || !c.is_control())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
