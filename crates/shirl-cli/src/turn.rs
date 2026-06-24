// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use shirl_agents::agents::AgentKind;
use shirl_core::Resolved;
use sweet_agent::{Agent, AgentIo, TurnResult};
use sweet_core::Model;
use sweet_core::{FinishReason, PermissionState};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::RuntimeCtx;

/// Resolve `@image` tokens in `input`, surface any warnings (size cap,
/// non-vision model), call `on_turn_start`, and spawn the model task.
///
/// Re-resolves the agent's model on every call so a `/model` switch between
/// turns is reflected in the vision check.
pub(crate) async fn spawn_user_turn(
    ctx: &RuntimeCtx<'_>,
    input: &str,
    agent_kind: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<TurnResult>>>,
) -> Result<()> {
    let cwd = std::env::current_dir().context("get cwd")?;
    let Resolved {
        blocks,
        mut warnings,
    } = shirl_core::resolve_media(input, &cwd).with_context(|| "resolving media in input")?;
    let has_images = shirl_core::has_images(&blocks);
    let has_files = shirl_core::has_files(&blocks);
    if has_images || has_files {
        if let Some(w) = attachment_check(ctx, agent_kind, has_images).await {
            warnings.push(w);
        }
    }

    {
        let mut io_guard = ctx.shared_io.lock().await;
        if !warnings.is_empty() {
            io_guard.insert_lines(&warnings)?;
        }
        io_guard.on_turn_start().await?;
    }

    let agent_clone = ctx.agent.clone();
    let mut io_clone = ctx.shared_io.clone();
    *model_handle = Some(tokio::spawn(async move {
        let mut agent = agent_clone.lock().await;
        agent.step_stream(blocks, &mut io_clone).await
    }));
    Ok(())
}

/// Check whether the model bound to `agent_kind` supports the attached
/// media types. `has_images` reports whether the input actually carries
/// image attachments. Returns `None` if everything attached is supported
/// (or if the model can't be found in the catalog), `Some(warning)`
/// otherwise.
async fn attachment_check(
    ctx: &RuntimeCtx<'_>,
    agent_kind: AgentKind,
    has_images: bool,
) -> Option<String> {
    let name = {
        let models = ctx.models.lock().await;
        models.name(agent_kind)?
    };
    let (provider_id, model_id) = name.split_once('/')?;
    let model = crate::model::find_model(ctx.catalog, provider_id, model_id)?;

    // Only warn about media types that are actually attached. The catalog
    // exposes a `vision` flag for images but no document-support flag yet,
    // so file attachments (PDFs) are never flagged - extend here once
    // providers add the field.
    if has_images && !model.vision {
        Some(format!(
            "⚠ Model {name} does not support image input - attachments may be ignored"
        ))
    } else {
        None
    }
}

/// A user-facing scrollback warning for a non-normal finish reason, or `None`
/// for a clean stop / tool-call turn. Surfaces truncation and refusals the
/// model would otherwise hide (incl. Fable 5 / Opus 4.8 HTTP-200 refusals).
pub(crate) fn finish_reason_warning(reason: &FinishReason) -> Option<String> {
    match reason {
        FinishReason::Length => Some(
            "⚠ Response was cut off at the model's output limit \
             (raise max_tokens or ask it to continue)."
                .into(),
        ),
        FinishReason::ContentFilter => Some("⚠ Response was stopped by a content filter.".into()),
        FinishReason::Refusal => Some("⚠ The model declined to complete this request.".into()),
        FinishReason::Stop | FinishReason::ToolCalls | FinishReason::Other(_) => None,
    }
}

/// Abort the in-flight turn (if any), repair the session, and report it.
pub(crate) async fn cancel_turn(
    agent: &Arc<Mutex<Agent<Arc<dyn Model>>>>,
    shared_io: &crate::SharedIo,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<TurnResult>>>,
) -> Result<()> {
    if let Some(h) = model_handle.take() {
        h.abort();
        let _ = h.await;
        let repaired = {
            let mut agent_guard = agent.lock().await;
            agent_guard.repair_orphaned_tool_calls()?
        };
        let mut io_guard = shared_io.lock().await;
        io_guard.show_cancelled(repaired)?;
        io_guard.clear_working();
        io_guard.draw()?;
    }
    Ok(())
}

/// Cycle the permission mode and update the UI.
pub(crate) async fn cycle_permission_mode(
    handle: &Arc<PermissionState>,
    shared_io: &crate::SharedIo,
    sandbox_enabled: bool,
    sandbox_warning_shown: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    let next = handle.cycle_mode();
    let label = match next {
        sweet_core::PermissionMode::Normal => "normal",
        sweet_core::PermissionMode::AutoEdit => "accept edits",
        sweet_core::PermissionMode::FullAuto => "auto",
    };
    let mut io_guard = shared_io.lock().await;
    io_guard.set_permission_mode(next)?;
    io_guard.insert_lines(&[format!("Permission mode: {}", label)])?;

    // One-time warning when entering FullAuto with sandbox off
    if next == sweet_core::PermissionMode::FullAuto
        && !sandbox_enabled
        && !sandbox_warning_shown.load(std::sync::atomic::Ordering::Relaxed)
    {
        sandbox_warning_shown.store(true, std::sync::atomic::Ordering::Relaxed);
        io_guard.insert_lines(&[
            "⚠ Full-auto mode with no sandbox - the agent can run any command without restriction."
                .to_string(),
            "  Start with --sandbox for OS-level isolation.".to_string(),
        ])?;
    }

    Ok(())
}
