// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use shirl_core::PersistedSession;
use sweet_agent::Agent;
use sweet_core::{MemoryItem, Model, Session};
use tokio::sync::Mutex;

use crate::file_picker::{self, FileListCache};
use crate::SharedIo;
use shirl_ui::transcript::TranscriptView;
use shirl_ui::Command;

/// Open the transcript popup, snapshotting the current session.
pub(crate) async fn open_transcript(
    state: &mut Option<TranscriptView>,
    agent: &Arc<Mutex<Agent<Arc<dyn Model>>>>,
    shared_io: &SharedIo,
) -> Result<()> {
    if state.is_some() {
        return Ok(());
    }
    let items: Vec<MemoryItem> = {
        let guard = agent.lock().await;
        transcript_items(guard.session())
    };
    let mut io = shared_io.lock().await;
    io.open_popup()?;
    let popup_height = io.popup_height().unwrap_or(24);
    let view = TranscriptView::new(&items, popup_height);
    io.render_transcript(&view)?;
    *state = Some(view);
    Ok(())
}

/// The items the transcript popup shows: the full history including rows
/// archived by compaction, with the synthetic compaction artifacts (summary
/// pairs, cleared-tool-result placeholders) hidden since the originals are
/// shown in their place. Sessions without archived history (or non-persisted
/// ones) fall back to the live view.
fn transcript_items(session: &dyn Session) -> Vec<MemoryItem> {
    let full = session
        .as_any()
        .downcast_ref::<PersistedSession>()
        .and_then(|s| s.full_items().ok());
    match full {
        Some(full) => full
            .into_iter()
            // The archived flag is intentionally unused: archived originals
            // carry `compacted = false` and their synthetic replacements
            // `compacted = true`, so filtering on `compacted` alone shows the
            // originals in place (filtering out archived rows would just
            // reproduce the live view).
            .filter(|(MemoryItem::Message(m), _)| !m.compacted)
            .map(|(item, _)| item)
            .collect(),
        None => session.items().to_vec(),
    }
}

/// Close the transcript popup if it is open.
pub(crate) async fn close_transcript(
    state: &mut Option<TranscriptView>,
    shared_io: &SharedIo,
) -> Result<()> {
    if state.take().is_some() {
        let mut io = shared_io.lock().await;
        io.close_popup()?;
    }
    Ok(())
}

/// Apply a relative scroll delta to the open transcript and re-render.
pub(crate) async fn scroll_transcript(
    view: &mut TranscriptView,
    delta: i32,
    shared_io: &SharedIo,
) -> Result<()> {
    let mut io = shared_io.lock().await;
    let popup_height = io.popup_height().unwrap_or(24);
    view.scroll(delta, popup_height);
    io.render_transcript(view)?;
    Ok(())
}

/// Route a `SelectMove` to the file picker if it's open, otherwise
/// scroll the transcript. Returns `true` if the picker consumed the
/// command (caller should `continue`).
pub(crate) async fn route_select_move(
    delta: i32,
    shared_io: &SharedIo,
    cache: &mut FileListCache,
    cwd: &Path,
    transcript_view: &mut Option<TranscriptView>,
) -> Result<bool> {
    let picker_open = shared_io.lock().await.file_picker.is_some();
    if picker_open {
        let _ = file_picker::dispatch(shared_io, cache, cwd, &Command::SelectMove(delta)).await?;
        Ok(true)
    } else {
        if let Some(ref mut view) = transcript_view {
            scroll_transcript(view, delta, shared_io).await?;
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::{Message, SessionId};

    #[test]
    fn transcript_shows_archived_originals_and_hides_summaries() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = PersistedSession::open_in(dir.path(), SessionId::new()).unwrap();
        for text in ["first question", "first answer", "second question"] {
            session
                .push(MemoryItem::Message(Message::user(text)))
                .unwrap();
        }
        let mut summary = Message::user("summary of the start");
        summary.compacted = true;
        session
            .replace_range(0..2, vec![MemoryItem::Message(summary)])
            .unwrap();

        // Live view lost the originals...
        let live: Vec<String> = session
            .messages()
            .iter()
            .map(|m| m.text_content())
            .collect();
        assert_eq!(live, ["summary of the start", "second question"]);

        // ...the transcript shows them, without the synthetic summary.
        let shown: Vec<String> = transcript_items(&session)
            .iter()
            .map(|MemoryItem::Message(m)| m.text_content())
            .collect();
        assert_eq!(shown, ["first question", "first answer", "second question"]);
    }

    #[test]
    fn transcript_falls_back_to_live_view_for_plain_sessions() {
        let mut session = sweet_core::InMemorySession::new();
        session
            .push(MemoryItem::Message(Message::user("hello")))
            .unwrap();
        let shown = transcript_items(&session);
        assert_eq!(shown.len(), 1);
    }
}
