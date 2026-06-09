// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use shirl_core::ShirlConfig;
use shirl_llm::catalog::Catalog;
use sweet_agent::{Agent, TurnResult};
use sweet_core::{Model, Session, SessionId};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use shirl_agents::agents::{self, AgentKind};
use shirl_ui::transcript::TranscriptView;
use shirl_ui::{Command, ReplIo, SharedIo};
use sweet_agent::AgentIo;
use sweet_core::sandbox::{DirectSandbox, Sandbox, SandboxPolicy};
use sweet_sandbox::OsSandbox;

mod approval;
mod cli;
mod commands;
mod file_picker;
mod headless;
mod mcp;
mod model;
mod picker;
mod switch;
mod tracking;
mod transcript;
mod turn;

use file_picker::FileListCache;
use model::ModelStore;

/// Viewport redraw cadence while a turn (model call or slow slash command) is
/// in flight. 150 ms ≈ 17 frames per breath cycle for the `⏺` indicator —
/// smooth without burning cycles.
const REDRAW_INTERVAL: Duration = Duration::from_millis(150);

struct RuntimeCtx<'a> {
    agent: &'a Arc<Mutex<Agent<Arc<dyn Model>>>>,
    shared_io: &'a SharedIo,
    extensions: &'a sweet_agent::ExtensionRegistry,
    models: &'a Mutex<ModelStore>,
    config: &'a Mutex<ShirlConfig>,
    auth: &'a Mutex<shirl_core::AuthStore>,
    catalog: &'a Catalog,
    mcp_providers: &'a [sweet_mcp::McpProvider],
    sandbox: &'a Arc<dyn Sandbox>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    cli::install_panic_hook();
    let cli_args = match cli::parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(2);
        }
    };

    if cli_args.headless {
        // Headless mode: no terminal setup needed.
        let prompt = match cli_args.prompt.context(
            "headless mode requires a prompt. \
             Pass one as an argument to -p or pipe to stdin.",
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e:#}");
                std::process::exit(2);
            }
        };
        let exit_code = headless::run_headless(
            prompt,
            cli_args.resume,
            if cli_args.json {
                headless::OutputFormat::Json
            } else {
                headless::OutputFormat::Text
            },
            cli_args.diff,
            cli_args.permission_mode,
            cli_args.sandbox_policy,
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("{e:#}");
            1
        });
        std::process::exit(exit_code);
    }

    let result = run(cli_args).await;
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::cursor::Show,
        crossterm::terminal::Clear(crossterm::terminal::ClearType::FromCursorDown),
    );
    result
}

async fn run(cli_args: cli::CliArgs) -> Result<()> {
    let resume_id = cli_args.resume.clone();
    let config_path = ShirlConfig::default_path()?;
    let auth_path = shirl_core::AuthStore::default_path()?;

    // Garbage-collect old clipboard pastes. Best-effort: any error here is
    // ignored so a broken or unreadable cache dir never blocks startup.
    if let Some(dir) = shirl_ui::clipboard_image::default_cache_dir() {
        let _ = shirl_ui::clipboard_image::sweep_dir(
            &dir,
            std::time::Duration::from_secs(7 * 24 * 60 * 60),
        );
    }

    let mut auth = shirl_core::AuthStore::load(&auth_path)?;

    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel(16);
    let io = Arc::new(Mutex::new(ReplIo::new(
        "not configured".to_string(),
        None,
        cmd_tx.clone(),
    )?));
    let (shared_io, mut approval_rx) = SharedIo::new(io.clone());
    ReplIo::spawn_input_thread(io.clone(), tokio::runtime::Handle::current());

    // A catalog fetch failure with no cache is non-fatal: shirl still starts,
    // and custom providers defined in config.toml remain usable.
    let http = reqwest::Client::new();
    let catalog = match Catalog::load(&http).await {
        Ok(c) => c,
        Err(e) => {
            let mut io_guard = io.lock().await;
            io_guard.insert_lines(&[format!(
                "Warning: could not load the model catalog ({e}). \
                 Only custom providers defined in config.toml are available."
            )])?;
            Catalog::default()
        }
    };

    let config = match ShirlConfig::load(&config_path)? {
        Some(c) if c.is_complete() => c,
        _ => {
            {
                let mut io_guard = io.lock().await;
                io_guard.insert_lines(&[
                    "Welcome to shirl! Let's set up your default model.".to_string()
                ])?;
            }
            let (provider_name, model_id) = picker::run_setup_picker(
                &shared_io,
                &mut cmd_rx,
                &catalog,
                &ShirlConfig::default(),
                &mut auth,
            )
            .await?;
            let mut config = ShirlConfig::default();
            config.set_default(provider_name, model_id);
            config.save(&config_path)?;
            config
        }
    };

    let default_provider = config.default.provider.clone();
    let default_model = config.default.model.clone();

    let mut store = ModelStore::new();

    let main_ctx = model::load_agent_model(
        &mut store,
        AgentKind::Main,
        &default_provider,
        &default_model,
        &config,
        &auth,
        &catalog,
    )
    .await?;

    for (kind, agent) in [(AgentKind::Plan, "plan"), (AgentKind::Review, "review")] {
        let provider = config.provider_for(agent).to_string();
        let model = config.model_for(agent).to_string();
        model::load_agent_model(
            &mut store, kind, &provider, &model, &config, &auth, &catalog,
        )
        .await?;
    }

    let models = Arc::new(Mutex::new(store));
    let config = Arc::new(Mutex::new(config));
    let auth = Arc::new(Mutex::new(auth));
    let catalog = Arc::new(catalog);

    let web_search = model::resolve_web_search(AgentKind::Main, &config, &auth).await;
    let session = match resume_id {
        Some(ref id) => shirl_core::PersistedSession::resume(id.clone())?,
        None => shirl_core::PersistedSession::new()?,
    };
    let _observability_guard = cli::init_observability(session.id())?;

    let mcp_providers = mcp::load_mcp_providers(&io, &auth).await;
    let mcp_specs = mcp::flatten_mcp_specs(&mcp_providers);

    let mut extensions = sweet_agent::ExtensionRegistry::new();
    extensions.register(shirl_core::agents_md::load());
    extensions.register(shirl_core::New);
    extensions.register(shirl_core::Clear);
    extensions.register(shirl_core::Compact);
    extensions.register(shirl_core::CustomCommandsProvider::load(
        cli::RESERVED_COMMANDS,
    ));
    extensions.register(shirl_core::SkillsProvider::load());

    let mut sandbox_enabled = cli_args.sandbox_policy != SandboxPolicy::Off;
    let sandbox: Arc<dyn Sandbox> = if sandbox_enabled {
        match OsSandbox::new(
            std::env::current_dir()
                .context("current directory does not exist — cd into a valid directory first")?,
            cli_args.sandbox_policy,
            // Let the agent read back plan/review files under ~/.shirl/sessions.
            tracking::sandbox_read_roots(),
            // Hide ~/.shirl (auth.toml holds API keys) from the sandbox.
            vec![".shirl".to_string()],
        ) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                let mut io_guard = shared_io.lock().await;
                io_guard.insert_lines(&[
                    format!("\u{26a0} Failed to create sandbox: {e}"),
                    "  Falling back to unsandboxed mode. Remove --sandbox to suppress this warning."
                        .to_string(),
                ])?;
                sandbox_enabled = false;
                Arc::new(DirectSandbox::new())
            }
        }
    } else {
        Arc::new(DirectSandbox::new())
    };

    let main_model = {
        let store = models.lock().await;
        store
            .get(AgentKind::Main)
            .context("no model configured for main agent — run /model to set one")?
    };

    let session_id = session.id().clone();
    let agent = agents::build_agent(
        AgentKind::Main,
        main_model,
        &extensions,
        web_search,
        Box::new(session),
        &mcp_specs,
        sandbox.clone(),
    );
    let agent = shirl_core::install_auto_compaction(agent, shirl_core::CompactionConfig::default());
    let agent = shirl_core::install_media_strip(agent);
    // Resuming a session restores any persisted plan/review + todo list.
    let mut agent = match tracking::load_tracker(&session_id) {
        Some(tracker) => tracking::attach(agent, &tracker),
        None => agent,
    };

    if resume_id.is_some() {
        let repaired = agent.repair_orphaned_tool_calls()?;
        if repaired {
            let mut io_guard = io.lock().await;
            io_guard.show_session_repaired()?;
        }
    }

    {
        let mut io_guard = io.lock().await;
        io_guard.set_context_window(main_ctx)?;
        io_guard.set_model(format!("{}/{}", default_provider, default_model))?;
        io_guard.set_permission_mode(cli_args.permission_mode)?;
        io_guard.print_banner(&agent.session().id().to_string())?;
        if resume_id.is_some() {
            io_guard.print_resumed_messages(agent.session().items())?;
        }
    }

    let agent = Arc::new(Mutex::new(agent));
    let permission_handle = agent.lock().await.permission_handle();
    permission_handle.set_mode(cli_args.permission_mode);
    let sandbox_warning_shown = std::sync::atomic::AtomicBool::new(false);
    let commands = sweet_agent::CommandRouter::from_extension_registry(&extensions);

    // Build custom command descriptors from the template entries discovered
    // by CustomCommandsProvider. Derive a short description from the first
    // non-empty line of the template text.
    {
        let custom_cmds: Vec<shirl_ui::CommandInfo> = commands
            .template_entries()
            .map(|(name, template)| {
                let desc = template
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| {
                        let s = l.trim();
                        if s.len() > 60 {
                            format!("{}…", &s[..s.ceil_char_boundary(60)])
                        } else {
                            s.to_string()
                        }
                    })
                    .unwrap_or_else(|| "custom command".to_string());
                shirl_ui::CommandInfo {
                    name: name.to_string(),
                    description: desc,
                }
            })
            .collect();
        let mut io_guard = io.lock().await;
        io_guard.set_custom_commands(custom_cmds);
    }

    let mut active_agent = AgentKind::Main;
    let mut model_handle: Option<JoinHandle<sweet_core::Result<TurnResult>>> = None;
    // Tracks which session has had a title generated. `/new` swaps the session
    // id (so we'll regenerate); `/clear` keeps the id but wipes history (we
    // detect that separately by emptying this on the next command).
    let mut titled_session: Option<SessionId> = None;
    // Transcript-view state. `Some` iff the alternate-screen popup is open
    // and the main loop should route SelectMove/Resize/ToggleTranscript to it.
    let mut transcript_view: Option<TranscriptView> = None;
    let ctx = RuntimeCtx {
        agent: &agent,
        shared_io: &shared_io,
        extensions: &extensions,
        models: &models,
        config: &config,
        auth: &auth,
        catalog: &catalog,
        mcp_providers: &mcp_providers,
        sandbox: &sandbox,
    };

    // Drives viewport redraws while a turn is in flight: animates the
    // breathing `⏺` glyph and refreshes the elapsed-seconds counter.
    // Idle iterations don't poll this branch, so there's no off-turn churn.
    let mut redraw_tick = tokio::time::interval(REDRAW_INTERVAL);
    redraw_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut file_list_cache = FileListCache::default();
    let cwd = std::env::current_dir().context("get cwd")?;

    loop {
        if let Some(cmd) = {
            let mut io_guard = shared_io.lock().await;
            io_guard.pending_command.take()
        } {
            match cmd {
                Command::Submit(line) => {
                    commands::handle_chat_input(
                        &line,
                        &ctx,
                        &mut active_agent,
                        &mut model_handle,
                        &commands,
                        &mut cmd_rx,
                        &mut titled_session,
                    )
                    .await?;
                }
                Command::SelectMove(delta) => {
                    if transcript::route_select_move(
                        delta,
                        &shared_io,
                        &mut file_list_cache,
                        &cwd,
                        &mut transcript_view,
                    )
                    .await?
                    {
                        continue;
                    }
                }
                Command::Partial(_) | Command::Resize | Command::ApprovalKey(_) => {}
                Command::Cancel => {
                    turn::cancel_turn(&agent, &shared_io, &mut model_handle).await?;
                }
                Command::Exit => {
                    transcript::close_transcript(&mut transcript_view, &shared_io).await?;
                    if let Some(h) = model_handle.take() {
                        h.abort();
                        let _ = h.await;
                    }
                    break;
                }
                Command::CycleMode => {
                    turn::cycle_permission_mode(
                        &permission_handle,
                        &shared_io,
                        sandbox_enabled,
                        &sandbox_warning_shown,
                    )
                    .await?;
                }
                Command::ToggleTranscript => {
                    if transcript_view.is_some() {
                        transcript::close_transcript(&mut transcript_view, &shared_io).await?;
                    } else {
                        transcript::open_transcript(&mut transcript_view, &agent, &shared_io)
                            .await?;
                    }
                }
                fp_cmd @ (Command::FilePickerFilter(_)
                | Command::FilePickerAccept
                | Command::FilePickerClose) => {
                    file_picker::dispatch(&shared_io, &mut file_list_cache, &cwd, &fp_cmd).await?;
                }
            }
            continue;
        }

        if let Some(ref mut handle) = model_handle {
            tokio::select! {
                result = handle => {
                    model_handle = None;
                    match result {
                        Ok(Ok(turn_result)) => {
                            match turn_result {
                                TurnResult::Message(_) => {
                                    // End the turn and capture the session id
                                    // so we can decide whether to title.
                                    let current_session_id = {
                                        let mut io_guard = shared_io.lock().await;
                                        let agent_guard = agent.lock().await;
                                        let session = agent_guard.session();
                                        io_guard.on_turn_end(session).await?;
                                        session.id().clone()
                                    };

                                    // Generate a session title after the first
                                    // successful main-agent exchange for this
                                    // session. Runs in the background so the
                                    // user can submit the next prompt without
                                    // waiting on the model round-trip. The
                                    // model call runs without holding the IO
                                    // mutex; we only reacquire it briefly to
                                    // set the title.
                                    let needs_title = active_agent == AgentKind::Main
                                        && titled_session.as_ref() != Some(&current_session_id);
                                    if needs_title {
                                        let snapshot: Vec<_> = {
                                            let agent_guard = agent.lock().await;
                                            agent_guard
                                                .session()
                                                .items()
                                                .iter()
                                                .take(8)
                                                .cloned()
                                                .collect()
                                        };
                                        let model = {
                                            let store = models.lock().await;
                                            store
                                                .get(AgentKind::Main)
                                                .context("no model for main agent")?
                                        };
                                        // Mark before spawning so the next turn won't
                                        // spawn a second attempt while this one is in
                                        // flight. Best-effort: failures don't retry.
                                        titled_session = Some(current_session_id.clone());
                                        let io_clone = shared_io.clone();
                                        tokio::spawn(async move {
                                            if let Ok(Some(title)) =
                                                shirl_ui::compute_title(&model, &snapshot)
                                                    .await
                                            {
                                                let mut io_guard = io_clone.lock().await;
                                                let _ = io_guard.set_title(title);
                                            }
                                        });
                                    }
                                }
                                TurnResult::Handoff { target, payload } => {
                                    {
                                        let mut io_guard = shared_io.lock().await;
                                        let agent_guard = agent.lock().await;
                                        let session = agent_guard.session();
                                        io_guard.on_turn_end(session).await?;
                                    }
                                    match AgentKind::from_target(&target) {
                                        Some(kind) => {
                                            switch::apply_mode_switch(
                                                shirl_agents::agents::ModeSwitch {
                                                    target: kind,
                                                    step_with: payload,
                                                },
                                                &ctx,
                                                &mut active_agent,
                                                &mut model_handle,
                                            )
                                            .await?;
                                        }
                                        None => {
                                            let mut io_guard = shared_io.lock().await;
                                            io_guard.insert_lines(&[format!(
                                                "Error: handoff requested unknown agent `{target}`"
                                            )])?;
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            let mut io_guard = shared_io.lock().await;
                            let agent_guard = agent.lock().await;
                            let session = agent_guard.session();
                            io_guard.on_turn_end(session).await?;
                            let msg = format!("Error: {e}");
                            io_guard
                                .write_reply(&sweet_core::Message::system(&msg), session)
                                .await?;
                        }
                        Err(join_err) => {
                            let mut io_guard = shared_io.lock().await;
                            let agent_guard = agent.lock().await;
                            let session = agent_guard.session();
                            io_guard.on_turn_end(session).await?;
                            let msg = format!("Error: {join_err}");
                            io_guard
                                .write_reply(&sweet_core::Message::system(&msg), session)
                                .await?;
                        }
                    }
                }
                cmd = cmd_rx.recv() => {
                    match cmd {
                        Some(Command::Submit(line)) => {
                            let mut io_guard = shared_io.lock().await;
                            io_guard.pending_command = Some(Command::Submit(line));
                        }
                        Some(Command::SelectMove(delta)) => {
                            transcript::route_select_move(
                                delta,
                                &shared_io,
                                &mut file_list_cache,
                                &cwd,
                                &mut transcript_view,
                            )
                            .await?;
                        }
                        Some(Command::Partial(_))
                        | Some(Command::Resize)
                        | Some(Command::ApprovalKey(_))
                        | Some(Command::FilePickerFilter(_))
                        | Some(Command::FilePickerAccept)
                        | Some(Command::FilePickerClose) => {}
                        Some(Command::Cancel) => {
                            let mut io_guard = shared_io.lock().await;
                            io_guard.pending_command = Some(Command::Cancel);
                        }
                        Some(Command::Exit) => {
                            transcript::close_transcript(&mut transcript_view, &shared_io).await?;
                            if let Some(h) = model_handle.take() {
                                h.abort();
                                let _ = h.await;
                            }
                            break;
                        }
                        Some(Command::CycleMode) => {
                            turn::cycle_permission_mode(&permission_handle, &shared_io, sandbox_enabled, &sandbox_warning_shown).await?;
                        }
                        Some(Command::ToggleTranscript) => {
                            // Swallow — transcript view is only available when idle.
                            // Opening during a turn causes an abrupt switch
                            // when the handle resolves and the main loop
                            // re-enters Phase 1.
                        }
                        None => {}
                    }
                }
                _ = redraw_tick.tick() => {
                    let mut io_guard = shared_io.lock().await;
                    let _ = io_guard.draw();
                }
                approval = approval_rx.recv() => {
                    if let Some(req) = approval {
                        // Approvals render in the inline viewport; hide the
                        // alternate-screen transcript first so the prompt is
                        // visible to the user.
                        transcript::close_transcript(&mut transcript_view, &shared_io).await?;
                        let outcome = approval::run_approval_dialog(
                            &shared_io,
                            &mut cmd_rx,
                            &req.call,
                            req.risk,
                            req.response_tx,
                        )
                        .await?;
                        // Esc / Ctrl+C in the prompt cancels the whole turn.
                        if outcome == approval::ApprovalOutcome::Cancelled {
                            turn::cancel_turn(&agent, &shared_io, &mut model_handle).await?;
                        }
                    }
                    // None: channel closed — agent task gone, nothing to do.
                }
            }
        } else {
            let cmd = cmd_rx.recv().await;
            match cmd {
                Some(Command::Submit(line)) => {
                    let mut io_guard = shared_io.lock().await;
                    io_guard.pending_command = Some(Command::Submit(line));
                }
                Some(Command::SelectMove(delta)) => {
                    transcript::route_select_move(
                        delta,
                        &shared_io,
                        &mut file_list_cache,
                        &cwd,
                        &mut transcript_view,
                    )
                    .await?;
                }
                Some(Command::Partial(_))
                | Some(Command::Resize)
                | Some(Command::ApprovalKey(_)) => {}
                Some(Command::Cancel) => {
                    let mut io_guard = shared_io.lock().await;
                    io_guard.pending_command = Some(Command::Cancel);
                }
                Some(Command::CycleMode) => {
                    turn::cycle_permission_mode(
                        &permission_handle,
                        &shared_io,
                        sandbox_enabled,
                        &sandbox_warning_shown,
                    )
                    .await?;
                }
                Some(Command::ToggleTranscript) => {
                    if transcript_view.is_some() {
                        transcript::close_transcript(&mut transcript_view, &shared_io).await?;
                    } else {
                        transcript::open_transcript(&mut transcript_view, &agent, &shared_io)
                            .await?;
                    }
                }
                Some(
                    fp_cmd @ (Command::FilePickerFilter(_)
                    | Command::FilePickerAccept
                    | Command::FilePickerClose),
                ) => {
                    file_picker::dispatch(&shared_io, &mut file_list_cache, &cwd, &fp_cmd).await?;
                }
                Some(Command::Exit) => {
                    transcript::close_transcript(&mut transcript_view, &shared_io).await?;
                    break;
                }
                None => break,
            }
        }
    }

    let mut io_guard = shared_io.lock().await;
    io_guard.print_resume_hint(&agent.lock().await.session().id().to_string())?;
    Ok(())
}
