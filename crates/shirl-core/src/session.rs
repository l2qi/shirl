// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;
use std::path::{Path, PathBuf};

use sweet_core::{MemoryItem, Message, Result, Session, SessionError, SessionId};
use sweet_session::SqliteSession;

/// A SQLite-backed session stored under `<base>/sessions/<id>/session.db`.
///
/// `new` and `resume` use the default base (`~/.shirl`) for the binary's
/// production paths. Tests can use [`open_in`](Self::open_in) with a
/// `TempDir` to drive the same logic without touching the user's home.
pub struct PersistedSession {
    inner: SqliteSession,
}

impl PersistedSession {
    /// Open a fresh session under `~/.shirl/sessions/`.
    pub fn new() -> Result<Self> {
        Self::open_in(&default_base()?, SessionId::new())
    }

    /// Resume an existing session under `~/.shirl/sessions/<id>/`.
    pub fn resume(id: SessionId) -> Result<Self> {
        Self::open_in(&default_base()?, id)
    }

    /// Open (or create) a session at `<base>/sessions/<id>/session.db`.
    /// Creates intermediate directories. If the database file already exists
    /// its rows are loaded into the cache.
    pub fn open_in(base: &Path, id: SessionId) -> Result<Self> {
        let path = base.join(format!("sessions/{}/session.db", id));
        let parent = path.parent().ok_or_else(|| {
            SessionError::storage(std::io::Error::other(format!(
                "session path {} has no parent directory",
                path.display()
            )))
        })?;
        std::fs::create_dir_all(parent).map_err(SessionError::storage)?;
        let inner = SqliteSession::open_with_id(&path, id).map_err(SessionError::storage)?;
        Ok(Self { inner })
    }
}

impl Session for PersistedSession {
    fn id(&self) -> &SessionId {
        self.inner.id()
    }

    fn push(&mut self, item: MemoryItem) -> Result<()> {
        self.inner.push(item)
    }

    fn items(&self) -> &[MemoryItem] {
        self.inner.items()
    }

    fn messages(&self) -> Vec<Message> {
        self.inner.messages()
    }

    fn clear(&mut self) -> Result<()> {
        self.inner.clear()
    }

    fn token_count(&self) -> usize {
        self.inner.token_count()
    }

    fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    fn context_size(&self) -> usize {
        self.inner.context_size()
    }

    fn replace_range(&mut self, range: Range<usize>, replacement: Vec<MemoryItem>) -> Result<()> {
        self.inner.replace_range(range, replacement)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn default_base() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| {
        SessionError::storage(std::io::Error::other("could not determine home directory"))
    })?;
    Ok(home.join(".shirl"))
}

/// Root holding every session's on-disk artifacts: `~/.shirl/sessions/`.
///
/// Used as a sandbox read root so the agent can read back plan/review files
/// across a `/new` (which rotates the session id) without re-exposing the rest
/// of `~/.shirl` — notably `auth.toml`, which lives outside this directory.
pub fn sessions_root() -> Result<PathBuf> {
    Ok(default_base()?.join("sessions"))
}

/// Directory holding a single session's on-disk artifacts (`session.db` plus
/// the workflow files written by [`crate::tracker`]). Mirrors the layout used
/// by [`PersistedSession::open_in`]: `~/.shirl/sessions/<id>/`.
pub fn session_dir(id: &SessionId) -> Result<PathBuf> {
    Ok(sessions_root()?.join(id.to_string()))
}
