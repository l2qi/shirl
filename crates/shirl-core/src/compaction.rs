// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_agent::{Agent, Capability, CommandContext, HookEvent, ProcedureSpec};
use sweet_core::{MemoryItem, Message, Model, Result, Role};

use crate::hooks::AutoCompactionProcedure;

/// Default number of recent items to keep verbatim when compacting.
pub const DEFAULT_PRESERVE_RECENT: usize = 6;

const AUTO_COMPACTION_PROCEDURE_ID: &str = "shirl:compaction:auto";

/// Configuration for automatic compaction.
pub struct CompactionConfig {
    pub threshold: f32,
    pub preserve_recent: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            preserve_recent: DEFAULT_PRESERVE_RECENT,
        }
    }
}

/// Install Shirl's automatic compaction on `agent`. The procedure runs as a
/// `BeforeModelCall` hook and compacts session history when the model's
/// context window is breached.
pub fn install_auto_compaction<M: Model>(agent: Agent<M>, config: CompactionConfig) -> Agent<M> {
    agent.with_capabilities([
        Capability::Procedure(ProcedureSpec::new(
            AUTO_COMPACTION_PROCEDURE_ID,
            "Automatically compact Shirl session history before model calls",
            AutoCompactionProcedure::new(config),
        )),
        Capability::hook(HookEvent::BeforeModelCall, AUTO_COMPACTION_PROCEDURE_ID),
    ])
}

/// Adjust `preserve_recent` so the compaction boundary does not split an
/// assistant-with-tool-calls / tool-result pair.
///
/// If the boundary lands between an assistant message carrying `tool_calls`
/// and its `Role::Tool` result, the preserved tail would start with a tool
/// result whose carrier was summarized away — a dangling reference that
/// strict providers reject. This function nudges `preserve_recent` upward
/// until the boundary no longer splits any such pair.
pub(crate) fn adjusted_preserve_count(items: &[MemoryItem], preserve_recent: usize) -> usize {
    let len = items.len();
    if preserve_recent >= len || preserve_recent == 0 {
        return preserve_recent;
    }

    let mut preserve = preserve_recent;
    loop {
        let boundary = len - preserve;
        if boundary == 0 {
            break;
        }

        let split = match (items.get(boundary - 1), items.get(boundary)) {
            (Some(MemoryItem::Message(before)), Some(MemoryItem::Message(after))) => {
                // Case 1: the last summarized item is an assistant with
                // tool_calls, and the first preserved item is a tool result
                // for those calls → split.
                (before.role == Role::Assistant && !before.tool_calls.is_empty())
                // Case 2: the first preserved item is a tool result whose
                // carrier is in the summarized range → split.
                || after.role == Role::Tool
            }
            _ => false,
        };

        if split && preserve < len {
            preserve += 1;
        } else {
            break;
        }
    }

    preserve
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use sweet_core::ToolCall;

    #[test]
    fn no_adjustment_when_no_split() {
        let items = vec![
            MemoryItem::Message(Message::user("hello")),
            MemoryItem::Message(Message::assistant("world")),
            MemoryItem::Message(Message::user("keep")),
            MemoryItem::Message(Message::assistant("this")),
        ];
        assert_eq!(adjusted_preserve_count(&items, 2), 2);
    }

    #[test]
    fn nudges_past_tool_result_at_boundary() {
        let items = vec![
            MemoryItem::Message(Message::user("hello")),
            MemoryItem::Message(Message {
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({}),
                }],
                ..Message::assistant("thinking")
            }),
            MemoryItem::Message(Message::tool_result("c1", "result")),
            MemoryItem::Message(Message::user("keep")),
        ];
        // boundary=2 lands on tool_result whose carrier is at index 1.
        // Should nudge to preserve=3 so the tool result stays with its carrier.
        assert_eq!(adjusted_preserve_count(&items, 2), 3);
    }

    #[test]
    fn nudges_past_assistant_with_tool_calls() {
        let items = vec![
            MemoryItem::Message(Message::user("hello")),
            MemoryItem::Message(Message {
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({}),
                }],
                ..Message::assistant("thinking")
            }),
            MemoryItem::Message(Message::tool_result("c1", "result")),
            MemoryItem::Message(Message::user("keep")),
        ];
        // preserve_recent=2 → boundary=2, items[1] is assistant with tool_calls
        // (last summarized), items[2] is tool_result (first preserved) → split.
        // Should nudge to preserve=3.
        assert_eq!(adjusted_preserve_count(&items, 2), 3);
    }

    #[test]
    fn no_adjustment_when_preserve_equals_len() {
        let items = vec![MemoryItem::Message(Message::user("only"))];
        assert_eq!(adjusted_preserve_count(&items, 1), 1);
    }
}

pub async fn compact_session(
    ctx: &mut dyn CommandContext,
    preserve_recent: usize,
    hint: Option<&str>,
) -> Result<()> {
    let items = ctx.session().items().to_vec();
    if items.len() <= preserve_recent {
        return Ok(());
    }
    let preserve_recent = crate::compaction::adjusted_preserve_count(&items, preserve_recent);
    let range = 0..(items.len() - preserve_recent);

    let summary_prompt = build_compaction_prompt(&items[range.clone()], hint);
    let summary_reply = ctx.model().complete(&[summary_prompt], &[]).await?;

    ctx.session_mut()
        .replace_range(range, compaction_pair(summary_reply.text_content(), hint))?;
    Ok(())
}

/// Build the user+assistant `MemoryItem` pair that replaces a compacted range.
/// Both messages carry `compacted = true` so they round-trip through
/// persistence as compaction-generated entries.
pub(crate) fn compaction_pair(summary: String, hint: Option<&str>) -> Vec<MemoryItem> {
    let mut user_msg = Message::user(compaction_user_content(hint));
    user_msg.compacted = true;
    let mut assistant_msg = Message::assistant(summary);
    assistant_msg.compacted = true;
    vec![
        MemoryItem::Message(user_msg),
        MemoryItem::Message(assistant_msg),
    ]
}

fn compaction_user_content(hint: Option<&str>) -> String {
    let mut s =
        String::from("[System: This is a compaction summary of the preceding conversation.]");
    if let Some(hint) = hint {
        s.push_str(&format!("\nUser hint: {}", hint));
    }
    s
}

pub(crate) fn build_compaction_prompt(
    items: &[sweet_core::MemoryItem],
    hint: Option<&str>,
) -> Message {
    let mut text = String::from(
        "Summarize the following conversation history concisely. Follow these guidelines:\n\
         - Preserve: architectural decisions, unresolved bugs, key facts, file paths discussed, \
         tool results that informed decisions\n\
         - Drop: redundant tool outputs, repeated explorations, intermediate reasoning\n\
         - Keep: the most recent exchange verbatim if possible\n\
         - Format: write a coherent narrative, not a bullet list\n\n",
    );
    for item in items {
        let MemoryItem::Message(msg) = item;
        text.push_str(&format!("{:?}: {}\n", msg.role, msg.text_content()));
    }
    if let Some(hint) = hint {
        text.push_str(&format!("\nAdditional context from user: {}", hint));
    }
    Message::user(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_pair_marks_compacted() {
        let pair = compaction_pair("Summary text".to_string(), None);
        assert_eq!(pair.len(), 2);

        let MemoryItem::Message(user) = &pair[0];
        assert_eq!(user.role, sweet_core::Role::User);
        assert!(user.compacted);
        assert!(user.text_content().contains("compaction summary"));

        let MemoryItem::Message(assistant) = &pair[1];
        assert_eq!(assistant.role, sweet_core::Role::Assistant);
        assert!(assistant.compacted);
        assert_eq!(assistant.text_content(), "Summary text");
    }

    #[test]
    fn compaction_pair_includes_hint() {
        let pair = compaction_pair("Summary".to_string(), Some("focus on auth"));
        let MemoryItem::Message(user) = &pair[0];
        assert!(user.text_content().contains("focus on auth"));
    }

    #[test]
    fn compaction_pair_no_hint_omits_hint_line() {
        let pair = compaction_pair("Summary".to_string(), None);
        let MemoryItem::Message(user) = &pair[0];
        assert!(!user.text_content().contains("User hint:"));
    }

    #[test]
    fn build_compaction_prompt_includes_history() {
        let items = vec![
            MemoryItem::Message(Message::user("hello")),
            MemoryItem::Message(Message::assistant("world")),
        ];
        let prompt = build_compaction_prompt(&items, None);
        let text = prompt.text_content();
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
        assert!(text.contains("Summarize"));
    }

    #[test]
    fn build_compaction_prompt_with_hint() {
        let items = vec![MemoryItem::Message(Message::user("fix bug"))];
        let prompt = build_compaction_prompt(&items, Some("keep file paths"));
        let text = prompt.text_content();
        assert!(text.contains("keep file paths"));
    }

    #[test]
    fn default_config_values() {
        let config = CompactionConfig::default();
        assert!((config.threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.preserve_recent, DEFAULT_PRESERVE_RECENT);
    }
}
