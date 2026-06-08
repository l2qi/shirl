// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use sweet_agent::Agent;
use sweet_core::{MemoryItem, Model};
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
        guard.session().items().to_vec()
    };
    let mut io = shared_io.lock().await;
    io.open_popup()?;
    let popup_height = io.popup_height().unwrap_or(24);
    let view = TranscriptView::new(&items, popup_height);
    io.render_transcript(&view)?;
    *state = Some(view);
    Ok(())
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
