// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Startup wiring and the `/memory` slash command for long-term memory.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use shirl_agents::MemoryWiring;
use shirl_core::{AuthStore, ShirlConfig};
use shirl_llm::catalog::Catalog;
// AgentIo is a trait import: it brings on_turn_start/on_turn_end into scope
// for ReplIo in start_spinner and run_distill_command (method resolution requires
// the trait to be in scope, not just the concrete type).
use sweet_agent::{Agent, AgentIo, DistillConfig, DistillReport, MemoryDistiller};
use sweet_core::{MemoryItem, MemoryQuery, Model};
use tokio::sync::Mutex;

use crate::model::resolve_provider_params;
use crate::RuntimeCtx;

/// Ceiling on one distillation pass (a model call plus embedding calls; the
/// provider clients set no HTTP timeout, so a stalled connection would
/// otherwise pin the pass forever).
const DISTILL_TIMEOUT: Duration = Duration::from_secs(120);

/// Slot holding the most recent background distill pass, so an explicit
/// `/memory distill` can wait it out (the background pass holds the span
/// claim - without joining it the command would report "nothing new" while
/// a pass is visibly in flight).
pub(crate) type DistillTask = Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>;

/// Lock config + auth, resolve the cwd, and build the run's memory wiring.
/// Both entry points (interactive `run` and `run_headless`) need this same
/// setup; they differ only in how they surface `warnings` (REPL scrollback
/// vs stderr), so that stays at the call site.
pub(crate) async fn resolve_wiring(
    config: &Mutex<ShirlConfig>,
    auth: &Mutex<AuthStore>,
    catalog: &Catalog,
    session_id: &str,
) -> Result<(Option<MemoryWiring>, Vec<String>)> {
    let config_guard = config.lock().await;
    let auth_guard = auth.lock().await;
    let cwd = std::env::current_dir().context("get cwd")?;
    Ok(build_wiring(
        &config_guard,
        &auth_guard,
        catalog,
        session_id,
        &cwd,
    ))
}

/// Build the run's memory wiring from config. Returns `None` (memory
/// disabled) when the config says so or when the store cannot be opened;
/// problems surface as warnings rather than failing startup.
pub(crate) fn build_wiring(
    config: &ShirlConfig,
    auth: &AuthStore,
    catalog: &Catalog,
    session_id: &str,
    cwd: &Path,
) -> (Option<MemoryWiring>, Vec<String>) {
    let mut warnings = Vec::new();
    if !config.memory.enabled {
        return (None, warnings);
    }

    let embedder = match &config.memory.embedder {
        Some(spec) => match build_embedder_from_spec(spec, config, auth, catalog) {
            Ok(e) => Some(e),
            Err(e) => {
                warnings.push(format!(
                    "Warning: memory embedder `{spec}` unavailable ({e}); \
                     falling back to keyword-only recall."
                ));
                None
            }
        },
        None => None,
    };

    let store = match shirl_core::memory::open_store(embedder) {
        Ok(s) => s,
        Err(e) => {
            warnings.push(format!("Warning: long-term memory disabled: {e}"));
            return (None, warnings);
        }
    };
    let project_scope = match shirl_core::memory::project_scope(cwd) {
        Ok(s) => s,
        Err(e) => {
            warnings.push(format!("Warning: long-term memory disabled: {e}"));
            return (None, warnings);
        }
    };

    // Built unconditionally: explicit `/memory distill` works even when the
    // automatic passes are configured off.
    let distiller = Arc::new(MemoryDistiller::new(
        Arc::clone(&store),
        project_scope.clone(),
        DistillConfig::default(),
    ));

    (
        Some(MemoryWiring {
            store,
            user_scope: shirl_core::memory::user_scope(),
            project_scope,
            session_id: session_id.to_string(),
            recall_limit: config.memory.recall_limit,
            distiller,
            auto_distill: config.memory.auto_distill,
        }),
        warnings,
    )
}

/// Resolve a `"provider/model-id"` embedder spec through the same provider
/// machinery models use (catalog or custom provider, key from auth.toml).
fn build_embedder_from_spec(
    spec: &str,
    config: &ShirlConfig,
    auth: &AuthStore,
    catalog: &Catalog,
) -> Result<Arc<dyn sweet_core::Embedder>> {
    let (provider_id, model_id) = spec
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("expected `provider/model-id`"))?;
    // Embedders have no reasoning or sampling, so no overrides are threaded here.
    let params = resolve_provider_params(provider_id, config, auth, catalog, model_id, None, None)?;
    if params.api_key.is_empty() {
        anyhow::bail!("no API key for provider `{provider_id}` in auth.toml");
    }
    shirl_llm::build_embedder(params.protocol, model_id, &params.base_url, &params.api_key)
}

/// Everything one distillation pass needs, snapshotted under a brief agent
/// lock so the pass itself can run without the agent.
async fn snapshot_agent(
    agent: &Mutex<Agent<Arc<dyn Model>>>,
) -> (Arc<dyn Model>, Vec<MemoryItem>, String) {
    let guard = agent.lock().await;
    (
        guard.model().clone(),
        guard.session().items().to_vec(),
        guard.session().id().to_string(),
    )
}

/// Automatic pass: claim the pending span (when it has reached `min_items`)
/// and distill it on a detached task, so the UI never waits on the model
/// call. Gated on `auto_distill`. The outcome lands in scrollback - silent
/// when nothing was written, one warning line on failure.
pub(crate) async fn spawn_distill(
    wiring: &MemoryWiring,
    agent: &Mutex<Agent<Arc<dyn Model>>>,
    io: &crate::SharedIo,
    task_slot: &DistillTask,
    min_items: usize,
) {
    if !wiring.auto_distill {
        return;
    }
    let (model, items, session_id) = snapshot_agent(agent).await;
    // Claimed before spawning: a concurrent pass can't grab the same span,
    // and an aborted pass (quit mid-flight) is skipped, not retried.
    let Some(claim) = wiring.distiller.claim_span(items.len(), min_items) else {
        return;
    };
    let distiller = Arc::clone(&wiring.distiller);
    let io = io.clone();
    let handle = tokio::spawn(async move {
        let result = tokio::time::timeout(
            DISTILL_TIMEOUT,
            distiller.distill_span(model.as_ref(), claim, &items, &session_id),
        )
        .await;
        let line = match result {
            Ok(Ok(report)) => match distill_summary(&report) {
                Some(line) => line,
                None => return,
            },
            Ok(Err(e)) => format!("⚠ memory distillation failed: {e}"),
            Err(_) => "⚠ memory distillation timed out".to_string(),
        };
        let mut io_guard = io.lock().await;
        let _ = io_guard.insert_lines(&[line]);
    });
    *task_slot.lock().await = Some(handle);
}

/// One-line summary of a pass, `None` when it wrote nothing.
fn distill_summary(report: &DistillReport) -> Option<String> {
    let mut parts = Vec::new();
    if !report.saved.is_empty() {
        parts.push(format!("{} saved", report.saved.len()));
    }
    if report.updated > 0 {
        parts.push(format!("{} updated", report.updated));
    }
    (!parts.is_empty()).then(|| format!("✦ long-term memory: {}", parts.join(", ")))
}

/// What `/memory distill` prints: the saved facts themselves (the user asked
/// for the pass, show them what it wrote).
fn distill_result_lines(report: &DistillReport) -> Vec<String> {
    if report.saved.is_empty() && report.updated == 0 {
        return vec!["Nothing qualified for long-term memory.".to_string()];
    }
    let mut lines: Vec<String> = report
        .saved
        .iter()
        .map(|r| format!("  ({}) {}", r.id, r.content))
        .collect();
    if report.updated > 0 {
        lines.push(format!("  {} existing updated", report.updated));
    }
    lines
}

/// Explicit `/memory distill`: first waits out any in-flight background
/// pass (which holds the span claim - its `✦` line lands in scrollback as
/// it finishes), then claims and distills whatever remains, inline under
/// the working indicator. Blocking is right when the user asked for the
/// pass; works regardless of `auto_distill`.
async fn run_distill_command(ctx: &RuntimeCtx<'_>, wiring: &MemoryWiring) -> Result<()> {
    let pending = match ctx.distill_task.lock().await.take() {
        Some(handle) if !handle.is_finished() => Some(handle),
        _ => None,
    };
    let joined = pending.is_some();
    let mut spinning = false;

    // Keep the breathing indicator animating while we wait - same pattern
    // as the other slow slash commands.
    let mut tick = tokio::time::interval(crate::commands::REDRAW_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    if let Some(mut handle) = pending {
        start_spinner(ctx, &mut spinning).await?;
        loop {
            tokio::select! {
                _ = &mut handle => break,
                _ = tick.tick() => redraw(ctx).await,
            }
        }
    }

    let (model, items, session_id) = snapshot_agent(ctx.agent).await;
    let lines = match wiring.distiller.claim_span(items.len(), 1) {
        None if joined => vec!["Background distill pass finished; nothing further.".to_string()],
        None => vec!["Nothing new to distill.".to_string()],
        Some(claim) => {
            start_spinner(ctx, &mut spinning).await?;
            let result = {
                let fut = tokio::time::timeout(
                    DISTILL_TIMEOUT,
                    wiring
                        .distiller
                        .distill_span(model.as_ref(), claim, &items, &session_id),
                );
                tokio::pin!(fut);
                loop {
                    tokio::select! {
                        r = &mut fut => break r,
                        _ = tick.tick() => redraw(ctx).await,
                    }
                }
            };
            match result {
                Ok(Ok(report)) => distill_result_lines(&report),
                Ok(Err(e)) => vec![format!("Error: distillation failed: {e}")],
                Err(_) => vec!["Error: distillation timed out.".to_string()],
            }
        }
    };

    let mut io_guard = ctx.shared_io.lock().await;
    if spinning {
        io_guard.clear_working();
        io_guard.draw()?;
    }
    io_guard.insert_lines(&lines)?;
    Ok(())
}

/// Show the working indicator once; later calls are no-ops.
async fn start_spinner(ctx: &RuntimeCtx<'_>, spinning: &mut bool) -> Result<()> {
    if !*spinning {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.on_turn_start().await?;
        *spinning = true;
    }
    Ok(())
}

async fn redraw(ctx: &RuntimeCtx<'_>) {
    let mut io_guard = ctx.shared_io.lock().await;
    let _ = io_guard.draw();
}

/// Handle the `/memory` subcommands (see [`MEMORY_USAGE`]).
pub(crate) async fn handle_memory_command(args: &str, ctx: &RuntimeCtx<'_>) -> Result<()> {
    let Some(wiring) = ctx.memory else {
        let mut io_guard = ctx.shared_io.lock().await;
        io_guard.insert_lines(&[
            "Long-term memory is disabled ([memory] in config.toml).".to_string()
        ])?;
        return Ok(());
    };
    if args.split_whitespace().next() == Some("distill") {
        return run_distill_command(ctx, wiring).await;
    }
    let lines = run_subcommand(args, wiring).await;
    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&lines)?;
    Ok(())
}

const MEMORY_USAGE: &str = "Usage: /memory [list [user|project] | add <text> | search <query> | \
     forget <id> | distill | help]";

const MEMORY_HELP: &[&str] = &[
    "  list [user|project]        show saved memories (default: both scopes)",
    "  add <text>                 save a memory to the project scope",
    "  search <query>             search memories in both scopes",
    "  forget <id>                delete a memory by id",
    "  distill                    extract durable facts from this session now",
    "  help                       show this help",
];

/// Dev-only escape hatch for cleaning up after distillation experiments;
/// too destructive to ship, so it exists (and is listed) only in debug
/// builds.
#[cfg(debug_assertions)]
const MEMORY_HELP_DEBUG: &[&str] =
    &["  forget all <user|project>  delete every memory in that scope (debug builds only)"];

async fn run_subcommand(args: &str, wiring: &MemoryWiring) -> Vec<String> {
    let (sub, rest) = match args.trim().split_once(char::is_whitespace) {
        Some((s, r)) => (s, r.trim()),
        None => (args.trim(), ""),
    };

    let scopes = vec![wiring.project_scope.clone(), wiring.user_scope.clone()];
    match sub {
        "" | "list" => {
            let scopes = match rest {
                "" => scopes,
                "user" => vec![wiring.user_scope.clone()],
                "project" => vec![wiring.project_scope.clone()],
                other => return vec![format!("Unknown scope `{other}`. {MEMORY_USAGE}")],
            };
            let query = MemoryQuery::new().with_scopes(scopes).with_limit(20);
            match wiring.store.search(&query).await {
                Ok(hits) if hits.is_empty() => vec!["No memories saved yet.".to_string()],
                Ok(hits) => hits.iter().map(|h| render_hit(h, wiring)).collect(),
                Err(e) => vec![format!("Error listing memories: {e}")],
            }
        }
        "add" => {
            if rest.is_empty() {
                return vec![MEMORY_USAGE.to_string()];
            }
            match wiring
                .store
                .save(
                    wiring.project_scope.clone(),
                    rest,
                    &[],
                    Some(&wiring.session_id),
                )
                .await
            {
                Ok(record) => vec![format!("Saved memory ({})", record.id)],
                Err(e) => vec![format!("Error saving memory: {e}")],
            }
        }
        "search" => {
            if rest.is_empty() {
                return vec![MEMORY_USAGE.to_string()];
            }
            let query = MemoryQuery::new()
                .with_text(rest)
                .with_scopes(scopes)
                .with_limit(10);
            match wiring.store.search(&query).await {
                Ok(hits) if hits.is_empty() => vec!["No matching memories.".to_string()],
                Ok(hits) => hits.iter().map(|h| render_hit(h, wiring)).collect(),
                Err(e) => vec![format!("Error searching memories: {e}")],
            }
        }
        "forget" => {
            // Release builds fall through to the id parse, where `all` is
            // just an invalid id.
            #[cfg(debug_assertions)]
            {
                let mut parts = rest.split_whitespace();
                if parts.next() == Some("all") {
                    // The mandatory scope is the confirmation step: wiping is
                    // irreversible, and the user scope is shared across every
                    // project - never wipe on a bare `forget all`.
                    let scope = match parts.next() {
                        Some("user") => wiring.user_scope.clone(),
                        Some("project") => wiring.project_scope.clone(),
                        _ => {
                            return vec![
                                "Specify which scope to wipe: /memory forget all <user|project> \
                                 (the user scope is shared across all projects)."
                                    .to_string(),
                            ]
                        }
                    };
                    return forget_all(&scope, wiring).await;
                }
            }
            let Ok(id) = rest.parse::<sweet_core::MemoryId>() else {
                return vec![format!("Invalid memory id `{rest}`. {MEMORY_USAGE}")];
            };
            // Only ids in this run's scopes are deletable - same visibility
            // rule the model's tools follow.
            let visible = match wiring.store.get(&id).await {
                Ok(Some(record)) => scopes.contains(&record.scope),
                Ok(None) => false,
                Err(e) => return vec![format!("Error looking up memory: {e}")],
            };
            if !visible {
                return vec![format!("No memory with id {id}.")];
            }
            match wiring.store.delete(&id).await {
                Ok(_) => vec![format!("Deleted memory ({id})")],
                Err(e) => vec![format!("Error deleting memory: {e}")],
            }
        }
        "help" => {
            #[allow(unused_mut)]
            let mut lines: Vec<String> = std::iter::once(MEMORY_USAGE.to_string())
                .chain(MEMORY_HELP.iter().map(|s| s.to_string()))
                .collect();
            #[cfg(debug_assertions)]
            lines.extend(MEMORY_HELP_DEBUG.iter().map(|s| s.to_string()));
            lines
        }
        other => vec![format!("Unknown subcommand `{other}`. {MEMORY_USAGE}")],
    }
}

/// Delete every memory in `scope`, batch by batch. App-side loop rather
/// than a bulk-delete on the `Memory` trait: volumes here are small, and
/// per-id deletes reuse the same code path (and FTS/vector cleanup) as
/// `forget <id>`.
#[cfg(debug_assertions)]
async fn forget_all(scope: &sweet_core::MemoryScope, wiring: &MemoryWiring) -> Vec<String> {
    let mut deleted = 0usize;
    loop {
        let query = MemoryQuery::new()
            .with_scopes([scope.clone()])
            .with_limit(100);
        let hits = match wiring.store.search(&query).await {
            Ok(hits) => hits,
            Err(e) => return vec![format!("Error listing memories: {e} ({deleted} deleted)")],
        };
        if hits.is_empty() {
            break;
        }
        let before = deleted;
        for hit in &hits {
            match wiring.store.delete(&hit.record.id).await {
                Ok(true) => deleted += 1,
                Ok(false) => {}
                Err(e) => {
                    return vec![format!(
                        "Error deleting memory ({}): {e} ({deleted} deleted)",
                        hit.record.id
                    )]
                }
            }
        }
        // A batch that deletes nothing would loop forever - bail instead.
        if deleted == before {
            return vec![format!(
                "Stopped: store kept returning undeletable memories ({deleted} deleted)."
            )];
        }
    }
    let scope_name = if scope == &wiring.user_scope {
        "user"
    } else {
        "project"
    };
    match deleted {
        0 => vec![format!("No memories in the {scope_name} scope.")],
        n => vec![format!("Deleted {n} memories from the {scope_name} scope.")],
    }
}

fn render_hit(hit: &sweet_core::MemoryHit, wiring: &MemoryWiring) -> String {
    let scope = if hit.record.scope == wiring.user_scope {
        "user"
    } else {
        "project"
    };
    let tags = if hit.record.tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", hit.record.tags.join(", "))
    };
    format!(
        "  ({}) [{scope}]{tags} {}",
        hit.record.id, hit.record.content
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::{EphemeralMemory, MemoryScope};

    fn wiring() -> MemoryWiring {
        let store: Arc<dyn sweet_core::Memory> = Arc::new(EphemeralMemory::new());
        MemoryWiring {
            user_scope: MemoryScope::User("default".into()),
            project_scope: MemoryScope::Project("/repo".into()),
            session_id: "s1".into(),
            recall_limit: 5,
            distiller: Arc::new(MemoryDistiller::new(
                Arc::clone(&store),
                MemoryScope::Project("/repo".into()),
                DistillConfig::default(),
            )),
            auto_distill: true,
            store,
        }
    }

    fn record(content: &str) -> sweet_core::MemoryRecord {
        sweet_core::MemoryRecord {
            id: sweet_core::MemoryId::new(),
            scope: MemoryScope::Project("/repo".into()),
            content: content.to_string(),
            tags: Vec::new(),
            source_session: None,
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn distill_summary_is_silent_when_nothing_written() {
        assert_eq!(distill_summary(&DistillReport::default()), None);
        let report = DistillReport {
            saved: vec![record("a fact")],
            updated: 2,
        };
        assert_eq!(
            distill_summary(&report).unwrap(),
            "✦ long-term memory: 1 saved, 2 updated"
        );
    }

    #[test]
    fn distill_result_lines_show_saved_facts() {
        let lines = distill_result_lines(&DistillReport::default());
        assert_eq!(lines, vec!["Nothing qualified for long-term memory."]);

        let report = DistillReport {
            saved: vec![record("user prefers rebase")],
            updated: 1,
        };
        let lines = distill_result_lines(&report);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("user prefers rebase"));
        assert!(lines[1].contains("1 existing updated"));
    }

    #[tokio::test]
    async fn add_list_search_forget_roundtrip() {
        let w = wiring();

        let lines = run_subcommand("add prefers rebase over merge", &w).await;
        assert!(lines[0].starts_with("Saved memory ("));

        let lines = run_subcommand("list", &w).await;
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("prefers rebase over merge"));
        assert!(lines[0].contains("[project]"));

        let lines = run_subcommand("search rebase", &w).await;
        assert!(lines[0].contains("prefers rebase"));

        let id = lines[0]
            .trim_start()
            .trim_start_matches('(')
            .split(')')
            .next()
            .unwrap()
            .to_string();
        let lines = run_subcommand(&format!("forget {id}"), &w).await;
        assert!(lines[0].starts_with("Deleted memory"));

        let lines = run_subcommand("list", &w).await;
        assert_eq!(lines, vec!["No memories saved yet.".to_string()]);
    }

    #[tokio::test]
    async fn list_filters_by_scope() {
        let w = wiring();
        w.store
            .save(w.user_scope.clone(), "user fact", &[], None)
            .await
            .unwrap();
        w.store
            .save(w.project_scope.clone(), "project fact", &[], None)
            .await
            .unwrap();

        let lines = run_subcommand("list user", &w).await;
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("user fact"));

        let lines = run_subcommand("list project", &w).await;
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("project fact"));
    }

    #[tokio::test]
    async fn forget_refuses_out_of_scope_ids() {
        let w = wiring();
        let foreign = w
            .store
            .save(MemoryScope::Project("/elsewhere".into()), "x", &[], None)
            .await
            .unwrap();

        let lines = run_subcommand(&format!("forget {}", foreign.id), &w).await;
        assert!(lines[0].starts_with("No memory with id"));
        assert!(w.store.get(&foreign.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn unknown_subcommand_shows_usage() {
        let lines = run_subcommand("bogus", &wiring()).await;
        assert!(lines[0].contains("Usage:"));
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn forget_all_requires_an_explicit_scope() {
        let w = wiring();
        w.store
            .save(w.project_scope.clone(), "keep me", &[], None)
            .await
            .unwrap();

        let lines = run_subcommand("forget all", &w).await;
        assert!(lines[0].contains("Specify which scope"), "got: {lines:?}");
        assert_eq!(w.store.search(&MemoryQuery::new()).await.unwrap().len(), 1);
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn forget_all_wipes_only_the_named_scope() {
        let w = wiring();
        for text in ["project fact one", "project fact two"] {
            w.store
                .save(w.project_scope.clone(), text, &[], None)
                .await
                .unwrap();
        }
        w.store
            .save(w.user_scope.clone(), "user fact", &[], None)
            .await
            .unwrap();

        let lines = run_subcommand("forget all project", &w).await;
        assert_eq!(lines, vec!["Deleted 2 memories from the project scope."]);

        let remaining = w.store.search(&MemoryQuery::new()).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].record.content, "user fact");

        let lines = run_subcommand("forget all project", &w).await;
        assert_eq!(lines, vec!["No memories in the project scope."]);
    }

    #[tokio::test]
    async fn help_lists_every_subcommand() {
        let lines = run_subcommand("help", &wiring()).await;
        assert!(lines[0].starts_with("Usage:"));
        let text = lines.join("\n");
        for sub in ["list", "add", "search", "forget", "distill", "help"] {
            assert!(text.contains(sub), "missing {sub}");
        }
    }
}
