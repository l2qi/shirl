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
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShirlConfig {
    /// Default provider/model used for all agents unless overridden.
    pub default: AgentModelConfig,
    /// Long-term memory settings (optional; defaults to enabled).
    #[serde(default)]
    pub memory: MemoryConfig,
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

/// Long-term memory settings, `[memory]` in config.toml.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Master switch for long-term memory (tools, recall, distillation).
    pub enabled: bool,
    /// Embedding model for semantic recall as `"provider/model-id"`
    /// (e.g. `"openai/text-embedding-3-small"`). `None` means keyword-only
    /// recall — no embedding API calls are ever made.
    ///
    /// Vectors are tied to the embedder that produced them: changing this
    /// demotes existing memories to keyword-only recall (they are not
    /// re-embedded) until each is next updated.
    pub embedder: Option<String>,
    /// Maximum memories injected into the system prompt per turn.
    pub recall_limit: usize,
    /// Automatically distill durable facts from the transcript (an extra
    /// model call every ~dozen session items and at session boundaries).
    pub auto_distill: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            embedder: None,
            recall_limit: 5,
            auto_distill: true,
        }
    }
}

/// Provider + model pair for a single agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        let web_search = self.agents.get(agent).and_then(|a| a.web_search.clone());
        self.agents.insert(
            agent.to_string(),
            AgentModelConfig {
                provider: provider.into(),
                model: model.into(),
                web_search,
            },
        );
    }

    /// Update the default config.
    pub fn set_default(&mut self, provider: impl Into<String>, model: impl Into<String>) {
        let web_search = self.default.web_search.take();
        self.default = AgentModelConfig {
            provider: provider.into(),
            model: model.into(),
            web_search,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_missing_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.toml");
        assert_eq!(ShirlConfig::load(&path).unwrap(), None);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");

        let mut config = ShirlConfig::default();
        config.set_default("anthropic", "claude-sonnet-4-20250514");
        config.set_agent_model("plan", "openai", "gpt-4o");
        config.save(&path).unwrap();

        let loaded = ShirlConfig::load(&path).unwrap().unwrap();
        assert_eq!(loaded.default.provider, "anthropic");
        assert_eq!(loaded.default.model, "claude-sonnet-4-20250514");
        assert_eq!(loaded.provider_for("plan"), "openai");
        assert_eq!(loaded.model_for("plan"), "gpt-4o");
    }

    #[test]
    fn is_complete_requires_provider_and_model() {
        let mut config = ShirlConfig::default();
        assert!(!config.is_complete());

        config.default.provider = "anthropic".to_string();
        assert!(!config.is_complete());

        config.default.model = "claude-sonnet-4-20250514".to_string();
        assert!(config.is_complete());
    }

    #[test]
    fn provider_for_falls_back_to_default() {
        let mut config = ShirlConfig::default();
        config.set_default("openai", "gpt-4o");
        assert_eq!(config.provider_for("plan"), "openai");
        assert_eq!(config.provider_for("review"), "openai");

        config.set_agent_model("plan", "anthropic", "claude-sonnet-4-20250514");
        assert_eq!(config.provider_for("plan"), "anthropic");
        assert_eq!(config.provider_for("review"), "openai");
    }

    #[test]
    fn model_for_falls_back_to_default() {
        let mut config = ShirlConfig::default();
        config.set_default("anthropic", "default-model");
        config.set_agent_model("plan", "openai", "plan-model");
        assert_eq!(config.model_for("main"), "default-model");
        assert_eq!(config.model_for("plan"), "plan-model");
    }

    #[test]
    fn web_search_for_agent_overrides_default() {
        let mut config = ShirlConfig::default();
        config.set_default("anthropic", "model");
        config.default.web_search = Some("tavily".to_string());
        assert_eq!(config.web_search_for("main"), Some("tavily"));

        let plan = AgentModelConfig {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            web_search: Some("brave".to_string()),
        };
        config.agents.insert("plan".to_string(), plan);
        assert_eq!(config.web_search_for("plan"), Some("brave"));
        // "review" has no override — falls back to default.
        assert_eq!(config.web_search_for("review"), Some("tavily"));
    }

    #[test]
    fn web_search_for_none_when_not_configured() {
        let config = ShirlConfig::default();
        assert_eq!(config.web_search_for("main"), None);
    }

    #[test]
    fn load_parses_custom_providers_and_extensions() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");

        let toml = r#"
[default]
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[providers.my-local]
base_url = "http://localhost:11434"
protocol = "openai"
display_name = "Local Ollama"

[models.my-local.llama3]
display_name = "Llama 3 8B"
context_window = 8192
"#;
        std::fs::write(&path, toml).unwrap();
        let config = ShirlConfig::load(&path).unwrap().unwrap();

        assert_eq!(config.providers.len(), 1);
        let provider = &config.providers["my-local"];
        assert_eq!(provider.base_url, "http://localhost:11434");
        assert_eq!(provider.protocol, "openai");
        assert_eq!(provider.display_name.as_deref(), Some("Local Ollama"));

        let ext = &config.models["my-local"]["llama3"];
        assert_eq!(ext.display_name.as_deref(), Some("Llama 3 8B"));
        assert_eq!(ext.context_window, Some(8192));
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested/deep/config.toml");
        let config = ShirlConfig::default();
        config.save(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn set_default_preserves_web_search() {
        let mut config = ShirlConfig::default();
        config.set_default("anthropic", "model-a");
        config.default.web_search = Some("tavily".to_string());
        // Switching model should preserve the web_search setting.
        config.set_default("openai", "model-b");
        assert_eq!(config.default.provider, "openai");
        assert_eq!(config.default.model, "model-b");
        assert_eq!(config.default.web_search.as_deref(), Some("tavily"));
    }

    #[test]
    fn set_agent_model_preserves_web_search() {
        let mut config = ShirlConfig::default();
        config.set_agent_model("plan", "anthropic", "model-a");
        config.agents.get_mut("plan").unwrap().web_search = Some("brave".to_string());
        // Switching model should preserve the web_search setting.
        config.set_agent_model("plan", "openai", "model-b");
        let plan = &config.agents["plan"];
        assert_eq!(plan.provider, "openai");
        assert_eq!(plan.model, "model-b");
        assert_eq!(plan.web_search.as_deref(), Some("brave"));
    }
}
