// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Text layout utilities for the inline viewport.
//!
//! Soft-wrap, cursor positioning, and display-width helpers shared by the
//! input editor, rendering, and the file-picker. All measurements are in
//! terminal display cells (a CJK glyph or emoji is two cells wide), matching
//! the `unicode-width` tables ratatui uses for buffer allocation.

use unicode_width::UnicodeWidthChar;

/// Maximum number of preview lines before truncation.
pub(crate) const PREVIEW_LINE_CAP: usize = 200;
/// Maximum character count for tool-args previews.
pub(crate) const TOOL_ARGS_PREVIEW_CHARS: usize = 80;

/// Display width of a single character in terminal columns, using the same
/// `unicode-width` tables ratatui uses to lay out buffer cells. Control
/// characters and other zero-width code points report 0.
pub(crate) fn char_width(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

/// Display width of a string in terminal columns. Matches the cell count
/// ratatui reserves when it renders the string, so wrapping and cursor
/// placement stay aligned with what is drawn.
pub(crate) fn unicode_display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
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
pub(crate) fn cursor_position(
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

/// Wrap text to fit within `width` columns. Splits on explicit `\n`
/// newlines first, then wraps each resulting segment at character boundaries.
pub(crate) fn wrap_line(line: &str, width: usize) -> Vec<String> {
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

/// Truncate `s` to `max` chars, appending `…` when truncated.
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Condense a JSON tool-args value into a short preview string.
pub(crate) fn summarize_args(args: &serde_json::Value) -> String {
    let s = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
    truncate_chars(&s, TOOL_ARGS_PREVIEW_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
