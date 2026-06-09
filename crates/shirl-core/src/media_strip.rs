// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Post-turn media stripping.
//!
//! Replaces `ContentBlock::Image` and `ContentBlock::File` blocks with compact
//! text placeholders once the turn that introduced them has fully resolved.
//! Binary attachments are expensive tokens on every subsequent API call
//! (they're re-sent with the full conversation history), so stripping them
//! after the turn ends keeps context costs bounded while still allowing the
//! model to consult the media across multiple intermediate model calls within
//! the same turn (e.g. while looping through tool calls).
//!
//! The stripping runs as an `AfterTurn` hook so it fires exactly once per
//! turn, after the final assistant message or handoff has been settled.

use sweet_agent::{CommandContext, ProcedureHandler};
use sweet_core::{async_trait, ContentBlock, MemoryItem, Message, Result, Session};

const STRIP_PROCEDURE_ID: &str = "shirl:media-strip";

/// Install the single-use media stripping hook on an agent.
pub fn install_media_strip<M: sweet_core::Model>(
    agent: sweet_agent::Agent<M>,
) -> sweet_agent::Agent<M> {
    agent.with_capabilities([
        sweet_agent::Capability::Procedure(sweet_agent::ProcedureSpec::new(
            STRIP_PROCEDURE_ID,
            "Strip image and file blocks from session history once the turn has ended",
            MediaStripProcedure,
        )),
        sweet_agent::Capability::hook(sweet_agent::HookEvent::AfterTurn, STRIP_PROCEDURE_ID),
    ])
}

struct MediaStripProcedure;

#[async_trait]
impl ProcedureHandler for MediaStripProcedure {
    async fn handle(
        &self,
        _invocation: &sweet_agent::HookInvocation,
        ctx: &mut dyn CommandContext,
    ) -> Result<()> {
        strip_media_from_session(ctx.session_mut())
    }
}

/// Replace all `ContentBlock::Image` and `ContentBlock::File` blocks in the
/// session with text placeholders. Only modifies messages that contain
/// attachments.
fn strip_media_from_session(session: &mut dyn Session) -> Result<()> {
    let items = session.items().to_vec();
    let mut stripped_count = 0usize;
    for (i, item) in items.iter().enumerate() {
        let MemoryItem::Message(msg) = item;
        if !msg.has_attachments() {
            continue;
        }
        let (stripped, n) = strip_media_from_message(msg);
        stripped_count += n;
        session.replace_range(i..(i + 1), vec![MemoryItem::Message(stripped)])?;
    }
    if stripped_count > 0 {
        tracing::debug!(stripped_count, "stripped media blocks from session history");
    }
    Ok(())
}

/// Replace `ContentBlock::Image` and `ContentBlock::File` blocks with text
/// placeholders, returning the updated message and the number of blocks
/// replaced.
fn strip_media_from_message(msg: &Message) -> (Message, usize) {
    let mut out = msg.clone();
    let mut count = 0usize;
    out.content = msg
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Image { .. } | ContentBlock::File { .. } => {
                count += 1;
                ContentBlock::text(block.to_string())
            }
            other => other.clone(),
        })
        .collect();
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::InMemorySession;

    #[test]
    fn strips_image_blocks_into_text_placeholders() {
        let msg = Message::user_blocks(vec![
            ContentBlock::text("here's a picture:"),
            ContentBlock::Image {
                data: vec![0u8; 1024],
                media_type: "image/png".to_string(),
            },
        ]);
        let (stripped, count) = strip_media_from_message(&msg);
        assert_eq!(count, 1);
        assert!(!stripped.has_attachments());
        assert_eq!(
            stripped.text_content(),
            "here's a picture:[image: image/png, 1 KB]"
        );
        // Original unchanged
        assert!(msg.has_images());
    }

    #[test]
    fn strips_file_blocks_into_text_placeholders() {
        let msg = Message::user_blocks(vec![
            ContentBlock::text("review this:"),
            ContentBlock::File {
                data: vec![0u8; 2048],
                media_type: "application/pdf".to_string(),
                filename: "report.pdf".to_string(),
            },
        ]);
        let (stripped, count) = strip_media_from_message(&msg);
        assert_eq!(count, 1);
        assert!(!stripped.has_attachments());
        assert_eq!(
            stripped.text_content(),
            "review this:[file: report.pdf, application/pdf, 2 KB]"
        );
        // Original unchanged
        assert!(msg.has_files());
    }

    #[test]
    fn strips_mixed_image_and_file_blocks() {
        let msg = Message::user_blocks(vec![
            ContentBlock::Image {
                data: vec![1],
                media_type: "image/png".to_string(),
            },
            ContentBlock::text(" and "),
            ContentBlock::File {
                data: vec![2],
                media_type: "application/pdf".to_string(),
                filename: "doc.pdf".to_string(),
            },
        ]);
        let (stripped, count) = strip_media_from_message(&msg);
        assert_eq!(count, 2);
        assert!(!stripped.has_attachments());
    }

    #[test]
    fn leaves_text_only_messages_untouched() {
        let msg = Message::user("just text");
        let (stripped, count) = strip_media_from_message(&msg);
        assert_eq!(count, 0);
        assert_eq!(stripped.text_content(), "just text");
    }

    #[test]
    fn session_strip_replaces_in_place() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user_blocks(vec![
                ContentBlock::text("look"),
                ContentBlock::Image {
                    data: vec![1, 2, 3],
                    media_type: "image/jpeg".to_string(),
                },
            ])))
            .unwrap();
        session
            .push(MemoryItem::Message(Message::assistant("I see it")))
            .unwrap();

        strip_media_from_session(&mut session).unwrap();

        let msgs = session.messages();
        assert!(!msgs[0].has_attachments());
        assert!(msgs[0].text_content().contains("[image: image/jpeg,"));
        assert_eq!(msgs[1].text_content(), "I see it");
    }

    #[test]
    fn skips_session_without_attachments() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("text only")))
            .unwrap();
        // No panic, no error
        strip_media_from_session(&mut session).unwrap();
        assert_eq!(session.messages()[0].text_content(), "text only");
    }

    #[test]
    fn only_touches_messages_with_attachments() {
        let mut session = InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("clean")))
            .unwrap();
        let msg = Message::user_blocks(vec![
            ContentBlock::text("img"),
            ContentBlock::Image {
                data: vec![42],
                media_type: "image/png".to_string(),
            },
        ]);
        session.push(MemoryItem::Message(msg)).unwrap();

        strip_media_from_session(&mut session).unwrap();

        // First message untouched
        assert_eq!(session.messages()[0].text_content(), "clean");
        // Second message stripped
        assert!(!session.messages()[1].has_attachments());
    }
}
