// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! API key storage for connected providers and services.
//!
//! Keys are stored in `~/.shirl/auth.toml` with sections for LLM providers,
//! web search providers, and MCP server credentials:
//!
//! ```toml
//! [llm]
//! openai = "sk-..."
//! anthropic = "sk-ant-..."
//!
//! [web_search]
//! tavily = "tvly-..."
//!
//! [mcp]
//! github_token = "ghp_..."
//! ```
//!
//! The file is created with `0o600` permissions (owner read/write only).
//! A provider is considered "connected" when it has an entry in the
//! corresponding section.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthStore {
    #[serde(default)]
    pub llm: HashMap<String, String>,
    #[serde(default)]
    pub web_search: HashMap<String, String>,
    #[serde(default)]
    pub mcp: HashMap<String, String>,
}

impl AuthStore {
    pub fn default_path() -> Result<std::path::PathBuf> {
        let home =
            dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
        Ok(home.join(".shirl").join("auth.toml"))
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let store: Self =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;

        #[cfg(unix)]
        {
            use std::io::Write as _;
            use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

            // Open (creating with 0o600) and truncate *before* writing any
            // secret bytes. Creating with the mode avoids the brief
            // world-readable window a write-then-chmod would leave on a new
            // file; the explicit set_permissions then also tightens a
            // pre-existing file whose mode predates this code.
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)
                .with_context(|| format!("writing {}", path.display()))?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
            file.write_all(text.as_bytes())
                .with_context(|| format!("writing {}", path.display()))?;
        }

        #[cfg(not(unix))]
        std::fs::write(path, &text).with_context(|| format!("writing {}", path.display()))?;

        Ok(())
    }

    pub fn get(&self, provider_id: &str) -> Option<&str> {
        self.llm.get(provider_id).map(|s| s.as_str())
    }

    pub fn set(&mut self, provider_id: impl Into<String>, key: impl Into<String>) {
        self.llm.insert(provider_id.into(), key.into());
    }

    pub fn remove(&mut self, provider_id: &str) {
        self.llm.remove(provider_id);
    }

    pub fn connected_ids(&self) -> impl Iterator<Item = &str> {
        self.llm.keys().map(|s| s.as_str())
    }

    pub fn contains(&self, provider_id: &str) -> bool {
        self.llm.contains_key(provider_id)
    }

    pub fn get_web_search_key(&self, provider_id: &str) -> Option<&str> {
        self.web_search.get(provider_id).map(|s| s.as_str())
    }

    pub fn get_mcp_key(&self, key: &str) -> Option<&str> {
        self.mcp.get(key).map(|s| s.as_str())
    }

    pub fn set_mcp_key(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.mcp.insert(key.into(), value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_returns_default_when_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.toml");
        let store = AuthStore::load(&path).unwrap();
        assert!(store.connected_ids().next().is_none());
    }

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let mut store = AuthStore::default();
        store.set("openai", "sk-test-123");
        store.set("anthropic", "sk-ant-456");
        store
            .web_search
            .insert("tavily".to_string(), "tvly-789".to_string());
        store.save(&path).unwrap();

        let loaded = AuthStore::load(&path).unwrap();
        assert_eq!(loaded.get("openai"), Some("sk-test-123"));
        assert_eq!(loaded.get("anthropic"), Some("sk-ant-456"));
        assert_eq!(loaded.get("unknown"), None);
        assert_eq!(loaded.get_web_search_key("tavily"), Some("tvly-789"));
        assert_eq!(loaded.get_web_search_key("unknown"), None);
    }

    #[test]
    #[cfg(unix)]
    fn save_sets_restricted_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let mut store = AuthStore::default();
        store.set("test", "key");
        store.save(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn connected_ids_returns_keys() {
        let mut store = AuthStore::default();
        store.set("alpha", "k1");
        store.set("beta", "k2");
        let mut ids: Vec<&str> = store.connected_ids().collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn remove_disconnects_provider() {
        let mut store = AuthStore::default();
        store.set("openai", "sk-test");
        assert!(store.contains("openai"));
        store.remove("openai");
        assert!(!store.contains("openai"));
    }

    #[test]
    fn mcp_key_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let mut store = AuthStore::default();
        store.set_mcp_key("github_token", "ghp_abc123");
        store.save(&path).unwrap();

        let loaded = AuthStore::load(&path).unwrap();
        assert_eq!(loaded.get_mcp_key("github_token"), Some("ghp_abc123"));
        assert_eq!(loaded.get_mcp_key("nonexistent"), None);
    }

    #[test]
    fn backward_compat_without_mcp_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        std::fs::write(&path, "[llm]\nopenai = \"sk-test\"\n").unwrap();

        let store = AuthStore::load(&path).unwrap();
        assert_eq!(store.get("openai"), Some("sk-test"));
        assert!(store.mcp.is_empty());
    }
}
