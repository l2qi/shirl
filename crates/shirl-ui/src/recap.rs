// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Session recap: select and truncate messages for the resumed-session banner.

use sweet_core::{MemoryItem, Message, Role};

/// Maximum number of user/assistant messages shown in the resumed-session recap.
pub(crate) const RESUME_MAX_MESSAGES: usize = 10;
/// Maximum lines rendered per message in the resumed-session recap.
pub(crate) const RESUME_LINES_PER_MESSAGE: usize = 3;

/// One message prepared for the resumed-session recap.
pub(crate) struct RecapEntry {
    /// `User` or `Assistant` - the recap skips every other role.
    pub role: Role,
    /// The first [`RESUME_LINES_PER_MESSAGE`] lines of the message content.
    pub lines: Vec<String>,
    /// How many further content lines were dropped after `lines`.
    pub omitted_lines: usize,
}

/// Select and truncate the messages shown in a resumed-session recap.
///
/// Keeps the last [`RESUME_MAX_MESSAGES`] non-empty user/assistant messages,
/// each truncated to [`RESUME_LINES_PER_MESSAGE`] lines. Returns the count of
/// older messages dropped from the front alongside the per-message entries.
pub(crate) fn recap_entries(items: &[MemoryItem]) -> (usize, Vec<RecapEntry>) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
