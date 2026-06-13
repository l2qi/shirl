// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Long-term memory storage and scope resolution.
//!
//! One `~/.shirl/memory.db` holds every memory across all projects. Two
//! scopes matter to shirl:
//!
//! - **User** (`MemoryScope::User("default")`) — personal preferences that
//!   follow the user everywhere.
//! - **Project** (`MemoryScope::Project(<canonical git root>)`) — facts about
//!   one codebase, keyed by the same git-root identity that AGENTS.md
//!   discovery uses, so the scope is stable no matter which subdirectory
//!   shirl is launched from.
//!
//! Saves default to the project scope; recall and search see both.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use sweet_core::{Embedder, Memory, MemoryScope};
use sweet_memory::SqliteMemory;

use crate::discovery;

/// Path to the shared memory database: `~/.shirl/memory.db`.
pub fn memory_db_path() -> Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    Ok(home.join(".shirl").join("memory.db"))
}

/// Open (or create) the shared memory store, optionally with an embedder for
/// semantic recall. The store is WAL-mode sqlite, safe to share between
/// concurrently running shirl instances.
pub fn open_store(embedder: Option<Arc<dyn Embedder>>) -> Result<Arc<dyn Memory>> {
    let path = memory_db_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let store = SqliteMemory::open(&path)
        .map_err(|e| anyhow::anyhow!("failed to open memory store at {}: {e}", path.display()))?;
    let store = match embedder {
        Some(embedder) => store.with_embedder(embedder),
        None => store,
    };
    Ok(Arc::new(store))
}

/// The user-level scope. Shirl is single-user per home directory, so the key
/// is a constant.
pub fn user_scope() -> MemoryScope {
    MemoryScope::User("default".to_string())
}

/// The project scope for `start`: the canonical path of the containing git
/// root (or of `start` itself outside a repo) — the same identity AGENTS.md
/// discovery uses.
pub fn project_scope(start: &Path) -> Result<MemoryScope> {
    let root = discovery::find_git_root(start).unwrap_or(start);
    let canonical = root
        .canonicalize()
        .with_context(|| format!("cannot canonicalize project root {}", root.display()))?;
    Ok(MemoryScope::Project(
        canonical.to_string_lossy().into_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_scope_uses_git_root() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        std::fs::create_dir_all(repo_root.join(".git")).unwrap();
        let sub = repo_root.join("crates").join("x");
        std::fs::create_dir_all(&sub).unwrap();

        let from_root = project_scope(repo_root).unwrap();
        let from_sub = project_scope(&sub).unwrap();
        assert_eq!(from_root, from_sub);
        match from_root {
            MemoryScope::Project(key) => {
                assert_eq!(key, repo_root.canonicalize().unwrap().to_string_lossy())
            }
            other => panic!("expected project scope, got {other:?}"),
        }
    }

    #[test]
    fn project_scope_outside_repo_uses_start_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let scope = project_scope(tmp.path()).unwrap();
        match scope {
            MemoryScope::Project(key) => {
                assert_eq!(key, tmp.path().canonicalize().unwrap().to_string_lossy())
            }
            other => panic!("expected project scope, got {other:?}"),
        }
    }
}
