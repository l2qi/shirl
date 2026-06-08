// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Persistent configuration for the Shirl CLI.
//!
//! Config is stored in `~/.shirl/config.toml` and defines the default
//! provider/model plus optional per-agent overrides, custom provider
//! definitions, and model extensions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Top-level configuration file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShirlConfig {
    /// Default provider/model used for all agents unless overridden.
    pub default: AgentModelConfig,
    /// Per-agent overrides (optional).
    #[serde(default)]
    pub agents: HashMap<String, AgentModelConfig>,
    /// Custom provider definitions (optional).
    #[serde(default)]
    pub providers: HashMap<String, CustomProviderEntry>,
    /// Model extensions per provider (optional).
    #[serde(default)]
    pub models: HashMap<String, HashMap<String, ModelExtension>>,
}

/// Provider + model pair for a single agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentModelConfig {
    /// Provider id (e.g. `"anthropic"`, `"openai"`, or a custom provider id).
    pub provider: String,
    /// Model identifier (e.g. `"claude-sonnet-4.5"`).
    pub model: String,
    /// Web search provider id (e.g. `"tavily"`). Optional.
    /// The API key must exist in `[web_search]` section of auth.toml.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search: Option<String>,
}

/// A user-defined provider declared in `config.toml` under `[providers.*]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderEntry {
    /// Base URL for the API endpoint.
    pub base_url: String,
    /// Wire protocol: `"openai"`, `"anthropic"`, or `"gemini"`.
    pub protocol: String,
    /// Human-readable name for the picker (optional, defaults to id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

/// A user-defined model extension declared under `[models.<provider_id>.*]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExtension {
    /// Human-readable name for the picker (optional, defaults to model id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Context window in tokens (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
}

impl ShirlConfig {
    /// Path to the default config file: `~/.shirl/config.toml`.
    pub fn default_path() -> Result<PathBuf> {
        let home =
            dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
        Ok(home.join(".shirl").join("config.toml"))
    }

    /// Load config from `path`.  Returns `Ok(None)` when the file does not
    /// exist.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(path)?;
        let config: ShirlConfig = toml::from_str(&text)?;
        Ok(Some(config))
    }

    /// Save config to `path`, creating parent directories if necessary.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Returns true when the config has the required fields (`default.provider`
    /// and `default.model`) set and non-empty.
    pub fn is_complete(&self) -> bool {
        !self.default.provider.is_empty() && !self.default.model.is_empty()
    }

    /// Resolve the effective provider for an agent kind.
    ///
    /// Falls back to `default` when no agent-specific override exists.
    pub fn provider_for(&self, agent: &str) -> &str {
        self.agents
            .get(agent)
            .map(|a| a.provider.as_str())
            .unwrap_or(self.default.provider.as_str())
    }

    /// Resolve the effective model for an agent kind.
    pub fn model_for(&self, agent: &str) -> &str {
        self.agents
            .get(agent)
            .map(|a| a.model.as_str())
            .unwrap_or(self.default.model.as_str())
    }

    /// Resolve the effective web search provider for an agent kind.
    /// Returns `None` if not configured for this agent or the default.
    pub fn web_search_for(&self, agent: &str) -> Option<&str> {
        self.agents
            .get(agent)
            .and_then(|a| a.web_search.as_deref())
            .or(self.default.web_search.as_deref())
    }

    /// Update the config for a specific agent kind.
    pub fn set_agent_model(
        &mut self,
        agent: &str,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) {
        self.agents.insert(
            agent.to_string(),
            AgentModelConfig {
                provider: provider.into(),
                model: model.into(),
                web_search: None,
            },
        );
    }

    /// Update the default config.
    pub fn set_default(&mut self, provider: impl Into<String>, model: impl Into<String>) {
        self.default = AgentModelConfig {
            provider: provider.into(),
            model: model.into(),
            web_search: None,
        };
    }
}
