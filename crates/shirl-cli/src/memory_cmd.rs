// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Startup wiring and the `/memory` slash command for long-term memory.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use shirl_agents::MemoryWiring;
use shirl_core::{AuthStore, ShirlConfig};
use shirl_llm::catalog::Catalog;
use sweet_agent::{DistillConfig, MemoryDistiller};
use sweet_core::MemoryQuery;

use crate::model::resolve_provider_params;
use crate::RuntimeCtx;

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

    let distiller = config.memory.auto_distill.then(|| {
        Arc::new(MemoryDistiller::new(
            Arc::clone(&store),
            project_scope.clone(),
            DistillConfig::default(),
        ))
    });

    (
        Some(MemoryWiring {
            store,
            user_scope: shirl_core::memory::user_scope(),
            project_scope,
            session_id: session_id.to_string(),
            recall_limit: config.memory.recall_limit,
            distiller,
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
    let params = resolve_provider_params(provider_id, config, auth, catalog, model_id)?;
    if params.api_key.is_empty() {
        anyhow::bail!("no API key for provider `{provider_id}` in auth.toml");
    }
    shirl_llm::build_embedder(params.protocol, model_id, &params.base_url, &params.api_key)
}

/// Handle `/memory [list [user|project] | add <text> | search <query> | forget <id>]`.
pub(crate) async fn handle_memory_command(args: &str, ctx: &RuntimeCtx<'_>) -> Result<()> {
    let lines = match ctx.memory {
        Some(wiring) => run_subcommand(args, wiring).await,
        None => vec!["Long-term memory is disabled ([memory] in config.toml).".to_string()],
    };
    let mut io_guard = ctx.shared_io.lock().await;
    io_guard.insert_lines(&lines)?;
    Ok(())
}

const MEMORY_USAGE: &str =
    "Usage: /memory [list [user|project] | add <text> | search <query> | forget <id>]";

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
            let Ok(id) = rest.parse::<sweet_core::MemoryId>() else {
                return vec![format!("Invalid memory id `{rest}`. {MEMORY_USAGE}")];
            };
            // Only ids in this run's scopes are deletable — same visibility
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
        other => vec![format!("Unknown subcommand `{other}`. {MEMORY_USAGE}")],
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
        MemoryWiring {
            store: Arc::new(EphemeralMemory::new()),
            user_scope: MemoryScope::User("default".into()),
            project_scope: MemoryScope::Project("/repo".into()),
            session_id: "s1".into(),
            recall_limit: 5,
            distiller: None,
        }
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
}
