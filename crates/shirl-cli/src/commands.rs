// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use shirl_agents::agents::{self, AgentKind, ModeCommand};
use shirl_core::{AuthStore, ReasoningPref};
use shirl_llm::catalog::ReasoningOption;
use sweet_agent::{Agent, AgentIo, CommandRouter, TurnResult};
use sweet_core::{Message, Model};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::switch;
use crate::turn;
use crate::RuntimeCtx;

const MAX_REVIEW_DIFF_BYTES: usize = 30_000;

/// Viewport redraw cadence while a turn (model call or slow slash command) is
/// in flight. 150 ms ~ 17 frames per breath cycle for the `⏺` indicator -
/// smooth without burning cycles.
pub(crate) const REDRAW_INTERVAL: std::time::Duration = std::time::Duration::from_millis(150);

pub(crate) async fn default_review_instruction(cwd: &Path) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(["diff", "HEAD"])
        .current_dir(cwd)
        .output()
        .await;
    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => {
            // No commits yet (or not a git repo) - diff against the empty
            // tree. Fails harmlessly if we're outside a repo.
            tokio::process::Command::new("git")
                .args(["diff"])
                .current_dir(cwd)
                .output()
                .await
                .ok()?
        }
    };
    if !output.status.success() {
        return None;
    }
    let diff = String::from_utf8(output.stdout)
        .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
    if diff.trim().is_empty() || diff.len() > MAX_REVIEW_DIFF_BYTES {
        return None;
    }
    Some(format!("Review the following git changes:\n\n{diff}"))
}

pub(crate) async fn handle_chat_input(
    line: &str,
    ctx: &RuntimeCtx<'_>,
    active_agent: &mut AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<TurnResult>>>,
    commands: &CommandRouter,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<crate::Command>,
    titled_session: &mut Option<sweet_core::SessionId>,
) -> Result<()> {
    let agent = ctx.agent;
    let trimmed = line.trim();
    if trimmed.starts_with('/') {
        // Echo slash commands immediately - no model to cancel first.
        {
            let mut io_guard = ctx.shared_io.lock().await;
            io_guard.echo_prompt(line)?;
        }
        if let Some((name, args)) = shirl_core::parse_slash_command(trimmed) {
            match name {
                "model" => {
                    if args.trim().is_empty() {
                        open_model_picker(ctx, *active_agent, model_handle, cmd_rx).await?;
                    } else {
                        handle_model_command(args, ctx, *active_agent, model_handle).await?;
                    }
                }
                "provider" => {
                    handle_provider_command(args, ctx, cmd_rx).await?;
                }
                "reasoning" => {
                    handle_reasoning_command(args, ctx, *active_agent, model_handle).await?;
                }
                "capabilities" => {
                    let lines = capability_lines(*active_agent, agent, commands).await;
                    let mut io_guard = ctx.shared_io.lock().await;
                    io_guard.insert_lines(&lines)?;
                }
                "memory" => {
                    crate::memory_cmd::handle_memory_command(args, ctx).await?;
                }
                "help" => {
                    let mut io_guard = ctx.shared_io.lock().await;
                    io_guard.insert_lines(&[
                        "Keyboard shortcuts:".to_string(),
                        "  Shift+Tab  cycle permission mode (normal -> accept edits -> auto)"
                            .to_string(),
                        "  Ctrl+J     insert newline".to_string(),
                        "  Ctrl+C     cancel current turn".to_string(),
                        "  Ctrl+D     exit (on empty input)".to_string(),
                    ])?;
                }
                _ => match agents::resolve_mode_command(name, args, *active_agent) {
                    ModeCommand::Switch(mut switch) => {
                        if switch.target == AgentKind::Review && switch.step_with.is_none() {
                            let cwd = std::env::current_dir()
                                .unwrap_or_else(|_| std::path::PathBuf::from("."));
                            switch.step_with = default_review_instruction(&cwd).await;
                            if switch.step_with.is_none() {
                                let mut io_guard = ctx.shared_io.lock().await;
                                io_guard.insert_lines(&[
                                    "Review mode: waiting for instructions.".to_string(),
                                ])?;
                            }
                        }
                        switch::apply_mode_switch(switch, ctx, active_agent, model_handle).await?;
                    }
                    ModeCommand::Invalid(msg) => {
                        let mut io_guard = ctx.shared_io.lock().await;
                        io_guard.insert_lines(&[format!("Error: {msg}")])?;
                    }
                    ModeCommand::NotModeCommand => {
                        // A custom command is a prompt template: render it and
                        // submit the result as a user turn. Templates can never
                        // shadow a built-in (the router drops colliding names),
                        // so this check is safe before the action path.
                        if let Some(template) = commands.template(name) {
                            // Cancel any in-flight turn before spawning the
                            // template-driven turn - same sequence as plain
                            // text input. Without this the old task keeps
                            // running detached, holding the agent mutex and
                            // interleaving output.
                            if let Some(h) = model_handle.take() {
                                h.abort();
                                let _ = h.await;
                                let repaired = {
                                    let mut agent_guard = agent.lock().await;
                                    agent_guard.repair_orphaned_tool_calls()?
                                };
                                {
                                    let mut io_guard = ctx.shared_io.lock().await;
                                    io_guard.show_cancelled(repaired)?;
                                    io_guard.abort_cleanup()?;
                                }
                            }
                            let rendered =
                                shirl_core::custom_commands::render_template(template, args);
                            turn::spawn_user_turn(ctx, &rendered, *active_agent, model_handle)
                                .await?;
                        } else {
                            // `/new` rotates the session away: distill the
                            // outgoing transcript's pending span on a detached
                            // task. The span is claimed from the pre-rotation
                            // snapshot before the command runs, so nothing is
                            // lost or double-distilled - and `/new` itself
                            // returns instantly. (Called before taking the
                            // agent lock: spawn_distill locks it briefly.)
                            if name == "new" {
                                if let Some(wiring) = ctx.memory.as_ref() {
                                    crate::memory_cmd::spawn_distill(
                                        wiring,
                                        agent,
                                        ctx.shared_io,
                                        ctx.distill_task,
                                        1,
                                    )
                                    .await;
                                }
                            }
                            let mut agent_guard = agent.lock().await;
                            // Snapshot the session id before invoking the command;
                            // `/new` swaps it, `/clear` keeps it but wipes items.
                            // Either case invalidates the current title - we
                            // detect both by checking the id changed or the
                            // session is now empty.
                            let prev_id = agent_guard.session().id().clone();
                            // Show the working indicator.
                            {
                                let mut io_guard = ctx.shared_io.lock().await;
                                io_guard.on_turn_start().await?;
                            }
                            // Run the command handler in a select! with a local
                            // redraw tick so the breathing ⏺ and elapsed timer
                            // keep animating while e.g. /compact calls the model.
                            let mut tick = tokio::time::interval(REDRAW_INTERVAL);
                            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                            let result = {
                                let cmd_fut = commands.handle(name, args, &mut *agent_guard);
                                tokio::pin!(cmd_fut);
                                loop {
                                    tokio::select! {
                                        r = &mut cmd_fut => break r,
                                        _ = tick.tick() => {
                                            let mut io_guard =
                                                ctx.shared_io.lock().await;
                                            let _ = io_guard.draw();
                                        }
                                    }
                                }
                            };
                            let mut io_guard = ctx.shared_io.lock().await;
                            io_guard.clear_working();
                            io_guard.draw()?;
                            // Check for session reset regardless of whether the
                            // command produced a message - a future handler may
                            // return Ok(None) after resetting the session.
                            let session = agent_guard.session();
                            let session_reset =
                                session.id() != &prev_id || session.items().is_empty();
                            if session_reset {
                                *titled_session = None;
                                io_guard.clear_title()?;
                            }
                            if let Some(msg) = result? {
                                io_guard.write_reply(&Message::system(msg), session).await?;
                            }
                        }
                    }
                },
            }
        }
    } else {
        if let Some(h) = model_handle.take() {
            h.abort();
            let _ = h.await;
            let repaired = {
                let mut agent_guard = agent.lock().await;
                agent_guard.repair_orphaned_tool_calls()?
            };
            {
                let mut io_guard = ctx.shared_io.lock().await;
                io_guard.show_cancelled(repaired)?;
                io_guard.abort_cleanup()?;
            }
        }

        // Echo the prompt AFTER any running model is cancelled so that
        // `⏺ Cancelled` appears above `› prompt`.
        {
            let mut io_guard = ctx.shared_io.lock().await;
            io_guard.echo_prompt(line)?;
        }

        turn::spawn_user_turn(ctx, line, *active_agent, model_handle).await?;
    }
    Ok(())
}

/// Run the interactive model picker and apply the selection to `active_agent`.
async fn open_model_picker(
    ctx: &RuntimeCtx<'_>,
    active_agent: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<TurnResult>>>,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<crate::Command>,
) -> Result<()> {
    let current = {
        let store = ctx.models.lock().await;
        store.name(active_agent).unwrap_or_default()
    };

    let connected: Vec<String> = {
        let auth = ctx.auth.lock().await;
        ctx.catalog
            .providers_with_auth(|id| auth.contains(id))
            .iter()
            .map(|p| p.id.clone())
            .collect()
    };

    if connected.is_empty() {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard
            .insert_lines(&["No providers connected. Use /provider to connect one.".to_string()])?;
        return Ok(());
    }

    {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[format!("Current model: {}", current)])?;
    }

    // The main loop is parked here for the picker's lifetime, so config cannot
    // change underneath us - a snapshot keeps the lock from being held across
    // the (potentially long) interaction.
    let config = ctx.config.lock().await.clone();
    let selection = crate::picker::run_model_picker(
        ctx.shared_io,
        cmd_rx,
        ctx.catalog,
        &config,
        connected,
        current,
    )
    .await?;

    match selection {
        Some((provider, model_id)) => {
            if let Err(e) =
                switch::apply_model_switch(&provider, &model_id, ctx, active_agent, model_handle)
                    .await
            {
                let mut io_guard = ctx.shared_io.lock().await;
                io_guard.insert_lines(&[format!("Error switching model: {e}")])?;
            }
        }
        None => {
            let mut io_guard = ctx.shared_io.lock().await;
            io_guard.insert_lines(&["Model selection cancelled.".to_string()])?;
        }
    }
    Ok(())
}

async fn handle_model_command(
    args: &str,
    ctx: &RuntimeCtx<'_>,
    active_agent: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<TurnResult>>>,
) -> Result<()> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        let current = {
            let store = ctx.models.lock().await;
            store.name(active_agent).unwrap_or_default()
        };
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[format!("Current model: {}", current)])?;
        return Ok(());
    }

    if let Some((provider, model_id)) = trimmed.split_once('/') {
        switch::apply_model_switch(provider, model_id, ctx, active_agent, model_handle).await?;
        return Ok(());
    }

    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&[format!(
        "Ambiguous model '{trimmed}'. Use provider/model_id (e.g. anthropic/claude-sonnet-4.5)"
    )])?;
    Ok(())
}

/// Describe a reasoning override for display (`/reasoning` with no args, and the
/// confirmation line). `None` is the catalog default.
fn describe_reasoning_pref(pref: Option<&ReasoningPref>) -> String {
    let Some(p) = pref else {
        return "default (catalog)".to_string();
    };
    let mut parts = Vec::new();
    if let Some(enabled) = p.enabled {
        parts.push(if enabled { "on" } else { "off" }.to_string());
    }
    if let Some(effort) = &p.effort {
        parts.push(format!("effort={effort}"));
    }
    if let Some(budget) = p.budget_tokens {
        parts.push(format!("budget={budget}"));
    }
    if parts.is_empty() {
        "default (catalog)".to_string()
    } else {
        parts.join(", ")
    }
}

/// Parse a `/reasoning` argument against the model's supported dialects. Returns
/// `Ok(None)` to clear the override, `Ok(Some(pref))` to set it, or `Err(msg)`
/// when the argument doesn't fit what the model supports.
fn parse_reasoning_arg(
    arg: &str,
    options: &[ReasoningOption],
) -> std::result::Result<Option<ReasoningPref>, String> {
    let lower = arg.to_ascii_lowercase();
    match lower.as_str() {
        "default" | "reset" | "clear" => return Ok(None),
        "on" => {
            return Ok(Some(ReasoningPref {
                enabled: Some(true),
                ..Default::default()
            }))
        }
        "off" => {
            return Ok(Some(ReasoningPref {
                enabled: Some(false),
                ..Default::default()
            }))
        }
        _ => {}
    }

    if let Ok(budget) = arg.parse::<u32>() {
        let bounds = options.iter().find_map(|o| match o {
            ReasoningOption::BudgetTokens { min, max } => Some((*min, *max)),
            _ => None,
        });
        let Some((min, max)) = bounds else {
            return Err("this model does not support a thinking budget".to_string());
        };
        if let Some(min) = min.filter(|m| budget < *m) {
            return Err(format!(
                "budget {budget} is below this model's minimum of {min}"
            ));
        }
        if let Some(max) = max.filter(|m| budget > *m) {
            return Err(format!(
                "budget {budget} is above this model's maximum of {max}"
            ));
        }
        return Ok(Some(ReasoningPref {
            budget_tokens: Some(budget),
            ..Default::default()
        }));
    }

    let effort_values: Vec<&str> = options
        .iter()
        .filter_map(|o| match o {
            ReasoningOption::Effort { values } => Some(values.iter().map(|s| s.as_str())),
            _ => None,
        })
        .flatten()
        .collect();
    if effort_values.is_empty() {
        return Err(format!("unknown reasoning option '{arg}'"));
    }
    if !effort_values.iter().any(|v| v.eq_ignore_ascii_case(arg)) {
        return Err(format!(
            "'{arg}' is not a valid effort level (expected one of {})",
            effort_values.join("/")
        ));
    }
    Ok(Some(ReasoningPref {
        effort: Some(lower),
        ..Default::default()
    }))
}

/// `/reasoning [on|off|<effort>|<budget>|default]` - view or change how the
/// active agent's model reasons, validated against its catalog dialect, then
/// rebuild the model in place and persist the override.
async fn handle_reasoning_command(
    args: &str,
    ctx: &RuntimeCtx<'_>,
    active_agent: AgentKind,
    model_handle: &mut Option<JoinHandle<sweet_core::Result<TurnResult>>>,
) -> Result<()> {
    let (provider_id, model_id, current, protocol) = {
        let config = ctx.config.lock().await;
        let agent = active_agent.display_name();
        let provider_id = config.provider_for(agent).to_string();
        (
            provider_id.clone(),
            config.model_for(agent).to_string(),
            config.reasoning_for(agent).cloned(),
            crate::model::lookup_protocol(ctx.catalog, &config, &provider_id),
        )
    };
    let options = crate::model::lookup_reasoning_options(ctx.catalog, &provider_id, &model_id);
    let summary = crate::model::reasoning_capability_summary(&options);

    let arg = args.trim();
    if arg.is_empty() {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[
            format!(
                "Reasoning for {}/{}: {}",
                provider_id,
                model_id,
                describe_reasoning_pref(current.as_ref())
            ),
            format!("Supported: {summary}"),
            "Usage: /reasoning <on|off|effort-level|budget-tokens|default>".to_string(),
        ])?;
        return Ok(());
    }

    let pref = match parse_reasoning_arg(arg, &options) {
        Ok(pref) => pref,
        Err(msg) => {
            let mut io_guard = ctx.shared_io.lock().await;
            io_guard.insert_lines(&[format!("Error: {msg}"), format!("Supported: {summary}")])?;
            return Ok(());
        }
    };

    // A bare `off` on a model that reasons by default with no off-switch can't
    // actually disable reasoning. Don't persist a misleading `off` override - a
    // later status view would report `off` while reasoning stays on - so reject
    // it with an explanation and make no change.
    let turning_off = matches!(&pref, Some(p)
        if p.enabled == Some(false) && p.effort.is_none() && p.budget_tokens.is_none());
    if turning_off && protocol.is_some_and(|p| !shirl_llm::can_disable_reasoning(p, &options)) {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[
            format!(
                "{provider_id}/{model_id} reasons by default and exposes no off-switch; \
                 reasoning stays on. No change made."
            ),
            format!("Supported: {summary}"),
        ])?;
        return Ok(());
    }

    let label = describe_reasoning_pref(pref.as_ref());
    switch::apply_reasoning_switch(pref, ctx, active_agent, model_handle).await?;
    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&[format!(
        "⏺ Reasoning set to {label} for {provider_id}/{model_id}"
    )])?;
    Ok(())
}

/// Connect a provider interactively: pick from the provider list, then prompt
/// for its API key, then persist it to `auth.toml`.
async fn connect_provider_interactive(
    ctx: &RuntimeCtx<'_>,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<crate::Command>,
) -> Result<()> {
    // Snapshots: the main loop is parked here for the pickers' lifetime, so
    // config/auth cannot change underneath us.
    let (config, auth) = {
        let config = ctx.config.lock().await;
        let auth = ctx.auth.lock().await;
        (config.clone(), auth.clone())
    };

    let provider_id = match crate::picker::run_provider_picker(
        ctx.shared_io,
        cmd_rx,
        ctx.catalog,
        &config,
        &auth,
    )
    .await?
    {
        Some(id) => id,
        None => {
            let mut io_guard = ctx.shared_io.lock().await;
            io_guard.insert_lines(&["Provider selection cancelled.".to_string()])?;
            return Ok(());
        }
    };

    let api_key = match crate::picker::prompt_api_key(
        ctx.shared_io,
        cmd_rx,
        &provider_id,
        ctx.catalog,
        &config,
    )
    .await?
    {
        Some(key) => key,
        None => {
            let mut io_guard = ctx.shared_io.lock().await;
            io_guard.insert_lines(&["Provider connection cancelled.".to_string()])?;
            return Ok(());
        }
    };

    {
        let mut auth = ctx.auth.lock().await;
        crate::picker::save_provider_key(&mut auth, &provider_id, &api_key)?;
    }
    let display = crate::picker::provider_display_name(&provider_id, ctx.catalog, &config);
    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&[format!("Connected: {} ({})", display, provider_id)])?;
    Ok(())
}

async fn handle_provider_command(
    args: &str,
    ctx: &RuntimeCtx<'_>,
    cmd_rx: &mut tokio::sync::mpsc::Receiver<crate::Command>,
) -> Result<()> {
    let sub = args.trim();
    if sub.is_empty() {
        connect_provider_interactive(ctx, cmd_rx).await?;
        return Ok(());
    }

    if let Some(rest) = sub.strip_prefix("--remove ") {
        let provider_id = rest.trim();
        {
            let mut auth = ctx.auth.lock().await;
            auth.remove(provider_id);
            auth.save(&AuthStore::default_path()?)?;
        }
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[format!("Disconnected provider: {}", provider_id)])?;
        return Ok(());
    }

    let parts: Vec<&str> = sub.splitn(2, ' ').collect();
    if parts.len() != 2 {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[
            "Usage: /provider <provider-id> <api-key>".to_string(),
            "       /provider --remove <provider-id>".to_string(),
        ])?;
        return Ok(());
    }

    let provider_id = parts[0];
    let api_key = parts[1];

    let config = ctx.config.lock().await;
    let exists = ctx.catalog.get_provider(provider_id).is_some()
        || config.providers.contains_key(provider_id);
    let display = crate::picker::provider_display_name(provider_id, ctx.catalog, &config);
    drop(config);

    if !exists {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[format!(
            "Unknown provider: {}. Use a provider id from the catalog or define it in config.toml [providers.{}]",
            provider_id, provider_id
        )])?;
        return Ok(());
    }

    {
        let mut auth = ctx.auth.lock().await;
        crate::picker::save_provider_key(&mut auth, provider_id, api_key)?;
    }

    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&[format!("Connected provider: {} ({})", display, provider_id)])?;
    Ok(())
}

/// Build the `/capabilities` output: the active agent's tools and handoffs,
/// plus the slash-commands routed through command capabilities.
async fn capability_lines(
    active_agent: AgentKind,
    agent: &Arc<Mutex<Agent<Arc<dyn Model>>>>,
    commands: &CommandRouter,
) -> Vec<String> {
    let (mut tools, mut handoffs) = {
        let agent_guard = agent.lock().await;
        let tools: Vec<String> = agent_guard.tools().iter().map(|t| t.name.clone()).collect();
        let handoffs: Vec<String> = agent_guard
            .handoffs()
            .iter()
            .map(|h| h.name().to_string())
            .collect();
        (tools, handoffs)
    };
    tools.sort_unstable();
    handoffs.sort_unstable();
    let mut cmds: Vec<&str> = commands
        .commands()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    cmds.sort_unstable();

    let mut lines = vec![format!(
        "Capabilities - {} agent",
        active_agent.display_name()
    )];
    lines.push(format!("  tools ({}):", tools.len()));
    lines.extend(tools.iter().map(|t| format!("    {t}")));
    if !handoffs.is_empty() {
        lines.push(format!("  handoffs ({}):", handoffs.len()));
        lines.extend(handoffs.iter().map(|h| format!("    {h}")));
    }
    let mut custom: Vec<&str> = commands.template_names().collect();
    custom.sort_unstable();
    let total_cmds = cmds.len() + custom.len();
    if total_cmds > 0 {
        lines.push(format!("  commands ({total_cmds}):"));
        lines.extend(cmds.iter().map(|c| format!("    /{c}")));
        if !cmds.is_empty() && !custom.is_empty() {
            lines.push("    ── custom ──".to_string());
        }
        lines.extend(custom.iter().map(|c| format!("    /{c}")));
    }
    lines
}

// ---------------------------------------------------------------------------
// Test helper
// ---------------------------------------------------------------------------

/// Trivial [`CommandHandler`] for tests.
#[cfg(test)]
struct DummyCmd;

#[cfg(test)]
use sweet_agent::{CommandContext, CommandHandler};

#[cfg(test)]
#[sweet_agent::async_trait]
impl CommandHandler for DummyCmd {
    async fn handle(
        &self,
        _args: &str,
        _ctx: &mut dyn CommandContext,
    ) -> sweet_core::Result<Option<String>> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    async fn run_git(cwd: &Path, args: &[&str]) {
        let status = tokio::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .await
            .expect("git invocation failed")
            .status;
        assert!(status.success(), "git {args:?} failed");
    }

    async fn init_repo(cwd: &Path) {
        run_git(cwd, &["init", "--quiet"]).await;
        run_git(cwd, &["config", "user.email", "test@example.com"]).await;
        run_git(cwd, &["config", "user.name", "Test"]).await;
        run_git(cwd, &["config", "commit.gpgsign", "false"]).await;
    }

    #[tokio::test]
    async fn default_review_instruction_returns_none_outside_git_repo() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(default_review_instruction(tmp.path()).await.is_none());
    }

    #[tokio::test]
    async fn default_review_instruction_returns_none_for_empty_diff() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path()).await;
        std::fs::write(tmp.path().join("hello.txt"), "original\n").unwrap();
        run_git(tmp.path(), &["add", "hello.txt"]).await;
        run_git(tmp.path(), &["commit", "--quiet", "-m", "initial"]).await;
        assert!(default_review_instruction(tmp.path()).await.is_none());
    }

    #[tokio::test]
    async fn default_review_instruction_returns_some_for_nonempty_diff() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path()).await;
        let file_path = tmp.path().join("hello.txt");
        std::fs::write(&file_path, "original\n").unwrap();
        run_git(tmp.path(), &["add", "hello.txt"]).await;
        run_git(tmp.path(), &["commit", "--quiet", "-m", "initial"]).await;
        std::fs::write(&file_path, "modified\n").unwrap();
        let prompt = default_review_instruction(tmp.path()).await.unwrap();
        assert!(prompt.contains("Review the following git changes"));
        assert!(prompt.contains("modified"));
    }

    #[tokio::test]
    async fn default_review_instruction_includes_staged_changes() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path()).await;
        let file_path = tmp.path().join("hello.txt");
        std::fs::write(&file_path, "original\n").unwrap();
        run_git(tmp.path(), &["add", "hello.txt"]).await;
        run_git(tmp.path(), &["commit", "--quiet", "-m", "initial"]).await;
        std::fs::write(&file_path, "staged change\n").unwrap();
        run_git(tmp.path(), &["add", "hello.txt"]).await;
        let prompt = default_review_instruction(tmp.path()).await.unwrap();
        assert!(prompt.contains("staged change"));
    }

    #[tokio::test]
    async fn capability_lines_merged_commands_with_divider() {
        use sweet_agent::test_util::MockModel;
        use sweet_agent::{Capability, CommandSpec, PromptSpec};
        use sweet_core::InMemorySession;

        let model: Arc<dyn Model> = Arc::new(MockModel::with_replies(["ok"]));
        let agent = Arc::new(Mutex::new(
            Agent::new(model).with_session(InMemorySession::new()),
        ));

        // Built-in commands (via CommandSpec) + custom templates.
        let router = CommandRouter::from_capabilities([
            Capability::Command(CommandSpec {
                name: "clear".into(),
                description: "clear session".into(),
                usage: String::new(),
                handler: Arc::new(DummyCmd),
            }),
            Capability::Prompt(PromptSpec::command("check", "run checks")),
            Capability::Prompt(PromptSpec::command("pr", "create a PR")),
        ]);

        let lines = capability_lines(AgentKind::Main, &agent, &router).await;
        let joined = lines.join("\n");

        // Single merged heading with total count.
        assert!(joined.contains("  commands (3):"));
        // Built-ins appear before the divider.
        assert!(joined.contains("    /clear"));
        // Divider separates built-ins from custom.
        assert!(joined.contains("    ── custom ──"));
        // Custom commands appear after the divider.
        assert!(joined.contains("    /check"));
        assert!(joined.contains("    /pr"));
        // No separate "custom commands" heading.
        assert!(!joined.contains("custom commands"));
    }

    #[tokio::test]
    async fn capability_lines_no_divider_when_only_builtins() {
        use sweet_agent::test_util::MockModel;
        use sweet_agent::{Capability, CommandSpec};
        use sweet_core::InMemorySession;

        let model: Arc<dyn Model> = Arc::new(MockModel::with_replies(["ok"]));
        let agent = Arc::new(Mutex::new(
            Agent::new(model).with_session(InMemorySession::new()),
        ));

        let router = CommandRouter::from_capabilities([Capability::Command(CommandSpec {
            name: "new".into(),
            description: "new session".into(),
            usage: String::new(),
            handler: Arc::new(DummyCmd),
        })]);

        let lines = capability_lines(AgentKind::Main, &agent, &router).await;
        let joined = lines.join("\n");

        assert!(joined.contains("  commands (1):"));
        assert!(joined.contains("    /new"));
        assert!(!joined.contains("── custom ──"));
    }

    #[tokio::test]
    async fn capability_lines_no_divider_when_only_custom() {
        use sweet_agent::test_util::MockModel;
        use sweet_agent::{Capability, PromptSpec};
        use sweet_core::InMemorySession;

        let model: Arc<dyn Model> = Arc::new(MockModel::with_replies(["ok"]));
        let agent = Arc::new(Mutex::new(
            Agent::new(model).with_session(InMemorySession::new()),
        ));

        let router = CommandRouter::from_capabilities([Capability::Prompt(PromptSpec::command(
            "deploy",
            "deploy the app",
        ))]);

        let lines = capability_lines(AgentKind::Main, &agent, &router).await;
        let joined = lines.join("\n");

        assert!(joined.contains("  commands (1):"));
        assert!(joined.contains("    /deploy"));
        assert!(!joined.contains("── custom ──"));
    }

    #[tokio::test]
    async fn capability_lines_no_commands_heading_when_empty() {
        use sweet_agent::test_util::MockModel;
        use sweet_core::InMemorySession;

        let model: Arc<dyn Model> = Arc::new(MockModel::with_replies(["ok"]));
        let agent = Arc::new(Mutex::new(
            Agent::new(model).with_session(InMemorySession::new()),
        ));
        let router = CommandRouter::new();

        let lines = capability_lines(AgentKind::Main, &agent, &router).await;
        let joined = lines.join("\n");

        assert!(!joined.contains("commands"));
    }

    #[test]
    fn parse_reasoning_clear_and_toggle() {
        assert_eq!(parse_reasoning_arg("default", &[]), Ok(None));
        assert_eq!(parse_reasoning_arg("reset", &[]), Ok(None));
        assert_eq!(
            parse_reasoning_arg("on", &[]),
            Ok(Some(ReasoningPref {
                enabled: Some(true),
                ..Default::default()
            }))
        );
        assert_eq!(
            parse_reasoning_arg("off", &[]),
            Ok(Some(ReasoningPref {
                enabled: Some(false),
                ..Default::default()
            }))
        );
    }

    #[test]
    fn parse_reasoning_budget_validates_bounds() {
        let opts = [ReasoningOption::BudgetTokens {
            min: Some(1024),
            max: Some(32000),
        }];
        // In range.
        assert_eq!(
            parse_reasoning_arg("4096", &opts),
            Ok(Some(ReasoningPref {
                budget_tokens: Some(4096),
                ..Default::default()
            }))
        );
        // Below the advertised minimum / above the maximum are rejected.
        assert!(parse_reasoning_arg("10", &opts)
            .unwrap_err()
            .contains("minimum of 1024"));
        assert!(parse_reasoning_arg("99999", &opts)
            .unwrap_err()
            .contains("maximum of 32000"));
    }

    #[test]
    fn parse_reasoning_budget_unbounded_and_unsupported() {
        // No bounds advertised: any positive budget is accepted.
        let unbounded = [ReasoningOption::BudgetTokens {
            min: None,
            max: None,
        }];
        assert_eq!(
            parse_reasoning_arg("1", &unbounded),
            Ok(Some(ReasoningPref {
                budget_tokens: Some(1),
                ..Default::default()
            }))
        );
        // A budget on an effort-only model is rejected.
        assert!(parse_reasoning_arg(
            "2048",
            &[ReasoningOption::Effort {
                values: vec!["low".into(), "high".into()]
            }]
        )
        .unwrap_err()
        .contains("does not support a thinking budget"));
    }

    #[test]
    fn parse_reasoning_effort_validated_against_values() {
        let opts = [ReasoningOption::Effort {
            values: vec!["low".into(), "high".into()],
        }];
        assert_eq!(
            parse_reasoning_arg("HIGH", &opts),
            Ok(Some(ReasoningPref {
                effort: Some("high".into()),
                ..Default::default()
            }))
        );
        assert!(parse_reasoning_arg("medium", &opts)
            .unwrap_err()
            .contains("not a valid effort level"));
    }
}
