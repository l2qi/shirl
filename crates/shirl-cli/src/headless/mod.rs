// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use shirl_agents::headless;
use shirl_core::{AuthStore, ShirlConfig};
use shirl_llm::catalog::Catalog;
use sweet_agent::{AgentIo, ExtensionRegistry, TurnResult};
use sweet_core::sandbox::{DirectSandbox, Sandbox, SandboxPolicy};
use sweet_core::{Message, Session, SessionId};
use sweet_sandbox::OsSandbox;

use crate::mcp;
use crate::model::{self, ModelStore};
use shirl_agents::agents::AgentKind;

/// Exit codes for headless mode.
pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_ERROR: i32 = 1;
pub const EXIT_CONFIG_INCOMPLETE: i32 = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Non-interactive I/O surface for headless mode.
///
/// Headless drives a single turn through `Agent::step_stream` and reads the
/// final assistant content from its `TurnResult::Message(msg)`. Streaming
/// deltas are deliberately ignored here: `step_stream`'s inner loop emits
/// `on_content_delta` for every model call (including intermediate rounds
/// before tool calls), so concatenating them would interleave pre-tool
/// reasoning with the final answer. Only tool activity surfaces, on stderr.
struct HeadlessIo;

#[sweet_core::async_trait]
impl AgentIo for HeadlessIo {
    async fn read_input(&mut self) -> sweet_core::error::Result<Option<String>> {
        // Headless drives a single turn through `Agent::step_stream`, which
        // injects the prompt directly. The agent runloop (`run`) is the only
        // caller of `read_input`, and we never enter it.
        unreachable!("headless mode drives Agent::step_stream, not Agent::run")
    }

    async fn write_reply(
        &mut self,
        _message: &Message,
        _session: &dyn Session,
    ) -> sweet_core::error::Result<()> {
        // Required by the trait but never invoked: `step_stream` does not call
        // `write_reply`. The final reply is read from `TurnResult::Message`.
        Ok(())
    }

    async fn on_tool_call(&mut self, call: &sweet_core::ToolCall) -> sweet_core::error::Result<()> {
        let args_str = match &call.arguments {
            serde_json::Value::String(s) => s.clone(),
            other => {
                let s = other.to_string();
                // Trim to first 80 chars of the JSON representation.
                truncate(&s, 80)
            }
        };
        eprintln!("· {}({})", call.name, args_str);
        Ok(())
    }

    async fn on_tool_result(
        &mut self,
        _call: &sweet_core::ToolCall,
        result: &str,
    ) -> sweet_core::error::Result<()> {
        let lines: Vec<&str> = result.lines().take(3).collect();
        for line in lines {
            eprintln!("  ↳ {}", truncate(line, 120));
        }
        Ok(())
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let mut out = String::with_capacity(s.len().min(max * 4));
    let mut iter = s.chars();
    for _ in 0..max.saturating_sub(1) {
        match iter.next() {
            Some(c) => out.push(c),
            None => return out,
        }
    }
    match (iter.next(), iter.next()) {
        (None, _) => out,
        (Some(c), None) => {
            out.push(c);
            out
        }
        (Some(_), Some(_)) => {
            out.push('…');
            out
        }
    }
}

/// Run shirl in headless mode. Returns an exit code.
pub async fn run_headless(
    prompt: String,
    resume_id: Option<SessionId>,
    format: OutputFormat,
    include_diff: bool,
    permission_mode: sweet_core::PermissionMode,
    sandbox_policy: SandboxPolicy,
) -> Result<i32> {
    let config_path = ShirlConfig::default_path()?;
    let auth_path = AuthStore::default_path()?;

    let auth = AuthStore::load(&auth_path)?;

    // Load config — fail if incomplete (headless can't launch the picker).
    let config = match ShirlConfig::load(&config_path)? {
        Some(c) if c.is_complete() => c,
        _ => {
            eprintln!(
                "Config incomplete. Run shirl interactively first to set up a default model."
            );
            return Ok(EXIT_CONFIG_INCOMPLETE);
        }
    };

    // Catalog fetch failure is non-fatal.
    let http = reqwest::Client::new();
    let catalog = match Catalog::load(&http).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Warning: could not load model catalog ({e}).");
            Catalog::default()
        }
    };

    let default_provider = config.default.provider.clone();
    let default_model = config.default.model.clone();

    // Headless workers all share the orchestrator's model via
    // `SubagentContext::parent_model`, so only the Main slot needs loading.
    // (Per-agent models for headless would require subagent handlers to
    // consult the store — separate change.)
    let mut store = ModelStore::new();
    model::load_agent_model(
        &mut store,
        AgentKind::Main,
        &default_provider,
        &default_model,
        &config,
        &auth,
        &catalog,
    )
    .await
    .context("failed to load model for headless mode")?;

    let models = Arc::new(tokio::sync::Mutex::new(store));
    let config = Arc::new(tokio::sync::Mutex::new(config));
    let auth = Arc::new(tokio::sync::Mutex::new(auth));

    let main_model = {
        let store = models.lock().await;
        store.get(AgentKind::Main).context("no model configured")?
    };

    let web_search = model::resolve_web_search(AgentKind::Main, &config, &auth).await;

    let session = match resume_id {
        Some(ref id) => shirl_core::PersistedSession::resume(id.clone())?,
        None => shirl_core::PersistedSession::new()?,
    };

    let session_id = session.id().clone();
    let _observability_guard = crate::cli::init_observability(&session_id)?;

    let auth_guard = auth.lock().await;
    let mcp_providers = mcp::load_mcp_providers_headless(&auth_guard).await;
    drop(auth_guard);
    let mcp_specs = mcp::flatten_mcp_specs(&mcp_providers);

    let mut extensions = ExtensionRegistry::new();
    extensions.register(shirl_core::agents_md::load());
    extensions.register(shirl_core::New);
    extensions.register(shirl_core::Clear);
    extensions.register(shirl_core::Compact);
    // Custom commands never fire in headless (no slash parsing), but the skills
    // catalog reaches the orchestrator and its children through the registry.
    extensions.register(shirl_core::CustomCommandsProvider::load(
        crate::cli::RESERVED_COMMANDS,
    ));
    extensions.register(shirl_core::SkillsProvider::load());
    let extensions = Arc::new(extensions);

    let sandbox: Arc<dyn Sandbox> = if sandbox_policy != SandboxPolicy::Off {
        match OsSandbox::new(
            std::env::current_dir().context("current directory does not exist")?,
            sandbox_policy,
            // Let the agent read back plan/review files under ~/.shirl/sessions.
            crate::tracking::sandbox_read_roots(),
            // Hide ~/.shirl (auth.toml holds API keys) from the sandbox.
            vec![".shirl".to_string()],
        ) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                eprintln!("Warning: failed to create sandbox: {e}. Falling back to unsandboxed.");
                Arc::new(DirectSandbox::new())
            }
        }
    } else {
        Arc::new(DirectSandbox::new())
    };

    let mut tracking = crate::tracking::load_tracker(&session_id)
        .map(crate::tracking::headless_tracking)
        .context("could not resolve session directory for workflow tracker")?;

    let worker_post_build: shirl_agents::headless::WorkerPostBuild =
        std::sync::Arc::new(|agent| {
            shirl_core::install_auto_compaction(agent, shirl_core::CompactionConfig::default())
        });
    tracking.worker_post_build = Some(worker_post_build);

    let mut agent = headless::build_orchestrator(
        main_model,
        extensions,
        web_search,
        Box::new(session),
        &mcp_specs,
        sandbox,
        tracking,
    );
    agent = shirl_core::install_auto_compaction(agent, shirl_core::CompactionConfig::default());
    agent = shirl_core::install_media_strip(agent);
    agent.set_permission_mode(permission_mode);

    // Run a single turn with the user's prompt.
    let mut io = HeadlessIo;
    let turn_result = agent.step_stream(prompt, &mut io).await;

    // The model's authoritative final reply lives on `TurnResult::Message`;
    // streaming deltas span multiple inner-loop iterations and would
    // concatenate intermediate pre-tool-call content with the final answer.
    let final_message = match turn_result {
        Ok(TurnResult::Message(msg)) => msg.text_content(),
        Ok(TurnResult::Handoff { .. }) => {
            eprintln!("Error: unexpected handoff in headless mode");
            return Ok(EXIT_ERROR);
        }
        Err(e) => {
            eprintln!("Error: {e}");
            return Ok(EXIT_ERROR);
        }
    };

    // Output results.
    match format {
        OutputFormat::Text => {
            if !final_message.is_empty() {
                println!("{}", final_message);
            }

            // Append git diff --stat
            let (stat, diff) = git_changes(include_diff).await;
            if let Some(stat) = stat {
                println!();
                println!("── changes ──");
                println!("{}", stat.trim_end());
            }
            if let Some(diff) = diff {
                println!("{}", diff);
            }
        }
        OutputFormat::Json => {
            let (stat, diff) = git_changes(include_diff).await;
            let files_touched = git_files_touched().await;
            let mut obj = serde_json::json!({
                "session_id": session_id.to_string(),
                "result": final_message,
                "files_touched": files_touched,
            });
            if let Some(stat) = stat {
                obj["diff_stat"] = serde_json::Value::String(stat.trim_end().to_string());
            }
            if let Some(diff) = diff {
                obj["diff"] = serde_json::Value::String(diff);
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&obj)
                    .expect("serializing constructed JSON cannot fail")
            );
        }
    }

    eprintln!("session: {}", session_id);
    Ok(EXIT_SUCCESS)
}

async fn git_changes(include_diff: bool) -> (Option<String>, Option<String>) {
    let stat = git_output(&["diff", "--stat", "HEAD"]).await;
    let diff = if include_diff {
        git_output(&["diff", "HEAD"]).await
    } else {
        None
    };
    (stat, diff)
}

async fn git_files_touched() -> Vec<String> {
    git_output(&["diff", "--name-only", "HEAD"])
        .await
        .map(|s| s.lines().map(|l| l.to_string()).collect())
        .unwrap_or_default()
}

async fn git_output(args: &[&str]) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).into_owned();
    if s.trim().is_empty() {
        None
    } else {
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::truncate;

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact_length_unchanged() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_long_string_gets_ellipsis() {
        assert_eq!(truncate("hello world", 8), "hello w…");
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        // Each `é` is multi-byte; the truncation must not slice mid-codepoint.
        let s = "éééééééé"; // 8 chars
        assert_eq!(truncate(s, 4), "ééé…");
    }

    #[test]
    fn truncate_zero_max_is_empty() {
        assert_eq!(truncate("anything", 0), "");
    }
}
