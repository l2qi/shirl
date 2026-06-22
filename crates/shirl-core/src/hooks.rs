// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use sweet_agent::{CommandContext, HookInvocation, ProcedureHandler};
use sweet_core::{async_trait, MemoryItem, Message, Result, Session};

use crate::compaction::CompactionConfig;

pub struct AutoCompactionProcedure {
    threshold: f32,
    preserve_recent: usize,
}

impl AutoCompactionProcedure {
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            threshold: config.threshold,
            preserve_recent: config.preserve_recent,
        }
    }

    /// Cheap compaction pass: replace tool-result messages older than the
    /// `preserve_recent` window with a short placeholder. Returns the estimated
    /// number of tokens freed - `context_size()` cannot observe this edit (the
    /// cleared messages predate the provider's most recent `prompt_tokens`
    /// measurement), so the caller subtracts this to decide whether the
    /// costlier summarization pass is still needed.
    fn clear_old_tool_results(&self, session: &mut dyn Session) -> Result<usize> {
        let items = session.items().to_vec();
        let cutoff = items.len().saturating_sub(self.preserve_recent);
        let placeholder = "[Result cleared - re-run tool if needed]";
        let mut freed = 0;
        for (i, item) in items.iter().enumerate() {
            if i >= cutoff {
                break;
            }
            let MemoryItem::Message(msg) = item;
            if msg.role == sweet_core::Role::Tool {
                freed += (msg.text_content().chars().count() / 4)
                    .saturating_sub(placeholder.chars().count() / 4);
                let mut cleared =
                    Message::tool_result(msg.tool_call_id.clone().unwrap_or_default(), placeholder);
                // Compaction artifact: lets full-transcript views (Ctrl+O)
                // hide the placeholder and show the archived original.
                cleared.compacted = true;
                session.replace_range(i..(i + 1), vec![MemoryItem::Message(cleared)])?;
            }
        }
        Ok(freed)
    }

    async fn summarize_old_messages(&self, ctx: &mut dyn CommandContext) -> Result<()> {
        let items = ctx.session().items().to_vec();
        if items.len() <= self.preserve_recent {
            return Ok(());
        }
        let preserve_recent =
            crate::compaction::adjusted_preserve_count(&items, self.preserve_recent);
        let range = 0..(items.len() - preserve_recent);

        let summary_prompt =
            crate::compaction::build_compaction_prompt(&items[range.clone()], None);
        let summary_reply = ctx.model().complete(&[summary_prompt], &[]).await?;

        let pair = crate::compaction::compaction_pair(summary_reply.text_content(), None);
        ctx.session_mut().replace_range(range, pair)?;
        Ok(())
    }
}

#[async_trait]
impl ProcedureHandler for AutoCompactionProcedure {
    async fn handle(
        &self,
        _invocation: &HookInvocation,
        ctx: &mut dyn CommandContext,
    ) -> Result<()> {
        let Some(max) = ctx.model().context_window() else {
            return Ok(());
        };
        let limit = (max as f32 * self.threshold) as usize;
        let used = ctx.session().context_size();
        if used <= limit {
            return Ok(());
        }
        // First, clear old tool results - cheap, no model call.
        let freed = self.clear_old_tool_results(ctx.session_mut())?;
        // If that did not reclaim enough, summarize old messages - costs a model call.
        if used.saturating_sub(freed) > limit {
            self.summarize_old_messages(ctx).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{install_auto_compaction, CompactionConfig};
    use sweet_agent::test_util::MockModel;
    use sweet_agent::{Agent, TurnResult};
    use sweet_core::{InMemorySession, MemoryItem, Message, Session};

    #[tokio::test]
    async fn auto_compaction_runs_as_before_model_procedure() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("old user message")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant(
                "old assistant message",
            )))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::user("newer user message")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant(
                "newer assistant message",
            )))
            .unwrap();
        let model = MockModel::with_replies(["summary", "final"]).with_context_window(1);
        let agent = Agent::new(model).with_session(session);
        let mut agent = install_auto_compaction(
            agent,
            CompactionConfig {
                threshold: 0.1,
                preserve_recent: 2,
            },
        );

        let reply = match agent.step("current request").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };

        assert_eq!(reply.text_content(), "final");
        let items = agent.session().items();
        // First two items should be the compaction user+assistant pair.
        assert_eq!(items.len(), 5);
        let MemoryItem::Message(msg) = &items[0];
        assert!(msg.compacted);
        assert_eq!(msg.role, sweet_core::Role::User);
        let MemoryItem::Message(msg) = &items[1];
        assert!(msg.compacted);
        assert_eq!(msg.role, sweet_core::Role::Assistant);
        assert_eq!(msg.text_content(), "summary");
    }

    #[tokio::test]
    async fn clearing_tool_results_can_skip_summarization() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("u0")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::tool_result(
                "call-1",
                "x".repeat(800),
            )))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::user("u2")))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant("a3")))
            .unwrap();

        // One reply only: if summarization ran it would need a second reply.
        let model = MockModel::with_replies(["final"]).with_context_window(100);
        let agent = Agent::new(model).with_session(session);
        let mut agent = install_auto_compaction(
            agent,
            CompactionConfig {
                threshold: 0.5,
                preserve_recent: 2,
            },
        );

        let reply = match agent.step("current request").await.unwrap() {
            TurnResult::Message(m) => m,
            TurnResult::Handoff { .. } => panic!("unexpected handoff"),
        };

        assert_eq!(reply.text_content(), "final");
        let items = agent.session().items();
        // Clearing the 800-char tool result drops usage below threshold, so no
        // summary is produced - every item is still a plain message.
        assert!(items.iter().all(|i| matches!(i, MemoryItem::Message(_))));
        // The old tool result was cleared in place (and marked as a
        // compaction artifact).
        let MemoryItem::Message(msg) = &items[1];
        assert!(msg.text_content().starts_with("[Result cleared"));
        assert!(msg.compacted);
    }
}
