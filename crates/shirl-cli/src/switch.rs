// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use shirl_agents::agents::{self, AgentKind, ModeSwitch};
use sweet_agent::Agent;
use sweet_core::Model;
use tokio::task::JoinHandle;

use crate::mcp;
use crate::model::resolve_web_search;
use crate::RuntimeCtx;
use shirl_core::ReasoningPref;
use shirl_llm::factory::build_model;
use std::sync::Arc;

/// Build a fresh agent: resolve web search -> build -> auto-compaction ->
/// media-strip -> tracker.
///
/// Called from both [`apply_mode_switch`] and [`apply_model_switch`].
/// The returned agent still needs its permission handle set by the caller
/// under the agent mutex - this function performs **no** async work while
/// holding any lock.
async fn rebuild_agent(
    ctx: &RuntimeCtx<'_>,
    kind: AgentKind,
    model: Arc<dyn Model>,
    session: Box<dyn sweet_core::Session>,
) -> Agent<Arc<dyn Model>> {
    let web_search = resolve_web_search(kind, ctx.config, ctx.auth).await;
    let mcp_specs = mcp::flatten_mcp_specs(ctx.mcp_providers);
    let tracker = crate::tracking::load_tracker(session.id());
    let new_agent = agents::build_agent(
        kind,
        model,
        ctx.extensions,
        web_search,
        session,
        &mcp_specs,
        ctx.sandbox.clone(),
        ctx.memory.as_ref(),
    );
    let new_agent =
        shirl_core::install_auto_compaction(new_agent, shirl_core::CompactionConfig::default());
    let new_agent = shirl_core::install_media_strip(new_agent);
    match (kind, &tracker) {
        (AgentKind::Main, Some(tracker)) => crate::tracking::attach(new_agent, tracker),
        _ => new_agent,
    }
}

pub(crate) async fn apply_mode_switch(
    mut switch: ModeSwitch,
    ctx: &RuntimeCtx<'_>,
    active_agent: &mut AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<sweet_agent::TurnResult>>>,
) -> Result<()> {
    if let Some(h) = model_handle.take() {
        h.abort();
        let _ = h.await;
    }

    let model = {
        let store = ctx.models.lock().await;
        store
            .get(switch.target)
            .context("no model configured for this agent")?
    };

    let (session, session_id) = {
        let mut agent_guard = ctx.agent.lock().await;
        // Repair any orphaned tool calls left by the aborted turn before
        // taking the session for the new agent. Without this, the session
        // may contain an assistant message with unanswered tool calls that
        // strict providers reject on the next request.
        agent_guard.repair_orphaned_tool_calls()?;
        let session = agent_guard.take_session();
        let session_id = session.id().to_owned();
        (session, session_id)
    };
    // Lock is released here - the async rebuild (resolve_web_search)
    // runs without blocking other tasks that need ctx.agent.

    // Entering Main from Plan/Review with a directive means a report is being
    // handed over. Persist it to disk so it survives Main's history compaction,
    // then point Main at the file. The report is the outgoing agent's last
    // substantial message; in that case `step_with` is a *separate* user
    // instruction (`/fix only item 3`, `/approve`) that carries the selection
    // and MUST be preserved. Only when there's no assistant text (a model-driven
    // handoff whose payload itself is the report) do we treat step_with as the
    // report and let the model choose.
    let tracker = crate::tracking::load_tracker(&session_id);
    if switch.target == AgentKind::Main
        && matches!(*active_agent, AgentKind::Plan | AgentKind::Review)
        && switch.step_with.is_some()
    {
        if let Some(tracker) = &tracker {
            let session_report = crate::tracking::last_assistant_text(&*session);
            if let Some(handover) =
                crate::tracking::resolve_handover(session_report, switch.step_with.clone())
            {
                let saved = if *active_agent == AgentKind::Plan {
                    tracker.save_plan(&handover.report)
                } else {
                    tracker.save_review(&handover.report)
                };
                if let Ok(path) = saved {
                    switch.step_with = Some(crate::tracking::report_directive(
                        *active_agent,
                        &path,
                        handover.instruction.as_deref(),
                    ));
                }
            }
        }
    }

    let new_agent = rebuild_agent(ctx, switch.target, model, session).await;
    {
        let mut agent_guard = ctx.agent.lock().await;
        let handle = agent_guard.permission_handle();
        *agent_guard = new_agent.with_permission_handle(handle);
    }
    *active_agent = switch.target;

    let mode_label = match switch.target {
        AgentKind::Main => None,
        AgentKind::Plan => Some("plan".to_string()),
        AgentKind::Review => Some("review".to_string()),
    };
    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.set_mode(mode_label)?;
    let (model_name, context_window) = {
        let store = ctx.models.lock().await;
        (
            store.name(switch.target).unwrap_or_default(),
            store.context_window(switch.target),
        )
    };
    io_guard.set_context_window(context_window)?;
    io_guard.set_model(model_name)?;
    io_guard.insert_lines(&[format!(
        "-> switched to {} mode",
        switch.target.display_name()
    )])?;
    drop(io_guard);
    if let Some(input) = switch.step_with {
        crate::turn::spawn_user_turn(ctx, &input, switch.target, model_handle).await?;
    }
    Ok(())
}

/// Rebuild the active agent's model from its **current** config (provider,
/// model, and any reasoning override), store it, and swap in a fresh agent. The
/// shared tail of both `/model` (which first changes provider/model) and
/// `/reasoning` (which first changes the reasoning override). Updates the status
/// line's context window and model name, but prints no scrollback message -
/// callers add their own.
pub(crate) async fn rebuild_active_model(
    provider_id: &str,
    model_id: &str,
    ctx: &RuntimeCtx<'_>,
    active_agent: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<sweet_agent::TurnResult>>>,
) -> Result<()> {
    let crate::model::ResolvedParams {
        protocol,
        base_url,
        api_key,
        context_window,
        max_output_tokens,
        reasoning,
        reasoning_replay,
        sampling,
    } = {
        let config = ctx.config.lock().await;
        let auth = ctx.auth.lock().await;
        let pref = config.reasoning_for(active_agent.display_name());
        let sampling_pref = config.sampling_for(active_agent.display_name());
        crate::model::resolve_provider_params(
            provider_id,
            &config,
            &auth,
            ctx.catalog,
            model_id,
            pref,
            sampling_pref,
        )?
    };

    let model = build_model(
        protocol,
        model_id,
        &base_url,
        &api_key,
        context_window,
        max_output_tokens,
        &reasoning,
        reasoning_replay,
        &sampling,
    )?;
    let full_name = format!("{}/{}", provider_id, model_id);

    {
        let mut store = ctx.models.lock().await;
        store.insert(
            active_agent,
            model.clone(),
            full_name.clone(),
            context_window,
        );
    }

    if let Some(h) = model_handle.take() {
        h.abort();
        let _ = h.await;
    }

    let session = {
        let mut agent_guard = ctx.agent.lock().await;
        // Repair any orphaned tool calls left by the aborted turn.
        agent_guard.repair_orphaned_tool_calls()?;
        agent_guard.take_session()
    };
    // Lock released before async rebuild.
    let new_agent = rebuild_agent(ctx, active_agent, model, session).await;
    {
        let mut agent_guard = ctx.agent.lock().await;
        let handle = agent_guard.permission_handle();
        *agent_guard = new_agent.with_permission_handle(handle);
    }

    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.set_context_window(context_window)?;
    io_guard.set_model(full_name)?;
    Ok(())
}

pub(crate) async fn apply_model_switch(
    provider_id: &str,
    model_id: &str,
    ctx: &RuntimeCtx<'_>,
    active_agent: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<sweet_agent::TurnResult>>>,
) -> Result<()> {
    {
        let mut config = ctx.config.lock().await;
        match active_agent {
            AgentKind::Main => config.set_default(provider_id, model_id),
            AgentKind::Plan => config.set_agent_model("plan", provider_id, model_id),
            AgentKind::Review => config.set_agent_model("review", provider_id, model_id),
        }
        let config_path = shirl_core::ShirlConfig::default_path()?;
        config.save(&config_path)?;
    }

    rebuild_active_model(provider_id, model_id, ctx, active_agent, model_handle).await?;

    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&[format!("⏺ Switched to {}/{}", provider_id, model_id)])?;
    Ok(())
}

/// Apply a `/reasoning` change: persist the override for the active agent, then
/// rebuild its model in place. The new model picks up the override via
/// [`rebuild_active_model`]. Returns the resolved `provider/model` for the
/// caller's status line.
pub(crate) async fn apply_reasoning_switch(
    pref: Option<ReasoningPref>,
    ctx: &RuntimeCtx<'_>,
    active_agent: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<sweet_agent::TurnResult>>>,
) -> Result<()> {
    let (provider_id, model_id) = {
        let mut config = ctx.config.lock().await;
        let agent = active_agent.display_name();
        config.set_reasoning(agent, pref);
        let config_path = shirl_core::ShirlConfig::default_path()?;
        config.save(&config_path)?;
        (
            config.provider_for(agent).to_string(),
            config.model_for(agent).to_string(),
        )
    };

    rebuild_active_model(&provider_id, &model_id, ctx, active_agent, model_handle).await
}
