// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! File-mention helpers: quoting, splicing, and @-trigger detection.

/// Quote and escape a path for insertion after `@`.
///
/// If the path contains whitespace or `"` characters, wrap it in double
/// quotes and escape any internal `"` as `\"` and `\` as `\\`. Otherwise
/// return the path as-is (no quoting needed).
pub(crate) fn quote_path_for_mention(path: &str) -> String {
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
pub(crate) fn splice_file_mention(
    input: &str,
    cursor: usize,
    path: &str,
) -> Option<(String, usize)> {
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

/// Scan backward from `cursor` in `input` looking for a `@` that begins a
/// mention token. Returns the text between `@` and cursor (the fuzzy-search
/// filter), or `None` if no trigger is active.
///
/// A `@` only counts as a trigger when it sits at the start of input or
/// directly after whitespace — this avoids false triggers mid-token (e.g.
/// while typing `name@host`).
///
/// `cursor` is a **char** index (from `InputState::cursor`).
pub(crate) fn mention_filter(input: &str, cursor: usize) -> Option<String> {
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
}
