// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use shirl_agents::agents::{self, AgentKind, ModeSwitch};
use tokio::task::JoinHandle;

use crate::mcp;
use crate::model::resolve_web_search;
use crate::RuntimeCtx;
use shirl_llm::factory::build_model;

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

    let web_search = resolve_web_search(switch.target, ctx.config, ctx.auth).await;
    let mcp_specs = mcp::flatten_mcp_specs(ctx.mcp_providers);
    let mut agent_guard = ctx.agent.lock().await;
    let session = agent_guard.take_session();

    // Entering Main from Plan/Review with a directive means a report is being
    // handed over. Persist it to disk so it survives Main's history compaction,
    // then point Main at the file. The report is the outgoing agent's last
    // substantial message; in that case `step_with` is a *separate* user
    // instruction (`/fix only item 3`, `/approve`) that carries the selection
    // and MUST be preserved. Only when there's no assistant text (a model-driven
    // handoff whose payload itself is the report) do we treat step_with as the
    // report and let the model choose.
    let tracker = crate::tracking::load_tracker(session.id());
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

    let new_agent = agents::build_agent(
        switch.target,
        model,
        ctx.extensions,
        web_search,
        session,
        &mcp_specs,
        ctx.sandbox.clone(),
    );
    let new_agent =
        shirl_core::install_auto_compaction(new_agent, shirl_core::CompactionConfig::default());
    let new_agent = shirl_core::install_media_strip(new_agent);
    let new_agent = match (switch.target, &tracker) {
        (AgentKind::Main, Some(tracker)) => crate::tracking::attach(new_agent, tracker),
        _ => new_agent,
    };
    // Preserve the permission handle across agent switches.
    let handle = agent_guard.permission_handle();
    let new_agent = new_agent.with_permission_handle(handle);
    *agent_guard = new_agent;
    *active_agent = switch.target;
    drop(agent_guard);

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
        "→ switched to {} mode",
        switch.target.display_name()
    )])?;
    drop(io_guard);
    if let Some(input) = switch.step_with {
        crate::turn::spawn_user_turn(ctx, &input, switch.target, model_handle).await?;
    }
    Ok(())
}

pub(crate) async fn apply_model_switch(
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
        reasoning,
    } = {
        let config = ctx.config.lock().await;
        let auth = ctx.auth.lock().await;
        crate::model::resolve_provider_params(provider_id, &config, &auth, ctx.catalog, model_id)?
    };

    let model = build_model(
        protocol,
        model_id,
        &base_url,
        &api_key,
        context_window,
        reasoning,
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

    if let Some(h) = model_handle.take() {
        h.abort();
        let _ = h.await;
    }

    {
        let web_search = resolve_web_search(active_agent, ctx.config, ctx.auth).await;
        let mcp_specs = mcp::flatten_mcp_specs(ctx.mcp_providers);
        let mut agent_guard = ctx.agent.lock().await;
        let session = agent_guard.take_session();
        let tracker = crate::tracking::load_tracker(session.id());
        let new_agent = agents::build_agent(
            active_agent,
            model,
            ctx.extensions,
            web_search,
            session,
            &mcp_specs,
            ctx.sandbox.clone(),
        );
        let new_agent =
            shirl_core::install_auto_compaction(new_agent, shirl_core::CompactionConfig::default());
        let new_agent = shirl_core::install_media_strip(new_agent);
        let new_agent = match (active_agent, &tracker) {
            (AgentKind::Main, Some(tracker)) => crate::tracking::attach(new_agent, tracker),
            _ => new_agent,
        };
        // Preserve the permission handle.
        let handle = agent_guard.permission_handle();
        let new_agent = new_agent.with_permission_handle(handle);
        *agent_guard = new_agent;
    }

    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.set_context_window(context_window)?;
    io_guard.set_model(full_name)?;
    io_guard.insert_lines(&[format!("⏺ Switched to {}/{}", provider_id, model_id)])?;
    Ok(())
}
