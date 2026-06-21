// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Persistent configuration for the Shirl CLI.
//!
//! Config is stored in `~/.shirl/config.toml` and defines the default
//! provider/model plus optional per-agent overrides, custom provider
//! definitions, and model extensions.

use std::collections::{BTreeMap, HashMap};
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
    /// recall - no embedding API calls are ever made.
    ///
    /// Vectors are tied to the embedder that produced them: changing this
    /// demotes existing memories to keyword-only recall (they are not
    /// re-embedded) until each is next updated.
    pub embedder: Option<String>,
    /// Maximum memories injected into the system prompt per turn.
    pub recall_limit: usize,
    /// Automatically distill durable facts from the transcript: a background
    /// model call every ~dozen session items and on `/new` (never blocks the
    /// UI). The explicit `/memory distill` command works regardless.
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

/// User override for how a model's reasoning is controlled, layered on top of
/// the catalog's reasoning flag and dialect. Plain data - no `shirl-llm`
/// dependency - so `shirl-cli` merges it with the catalog into a
/// `shirl_llm::ReasoningSettings` at model-build time.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningPref {
    /// Force reasoning on (`Some(true)`) or off (`Some(false)`); `None` defers
    /// to the catalog's reasoning flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Reasoning effort level (e.g. `"low"`/`"medium"`/`"high"`/`"none"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Thinking-token budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// User-set sampling / generation parameters, layered on top of the model's
/// own defaults. Plain data - no `shirl-llm`/`sweet-llm` dependency - so
/// `shirl-cli` maps it into a `sweet_llm::SamplingConfig` at model-build time.
/// Every field is optional: an absent field is not sent, so the model applies
/// its own default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingPref {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Arbitrary provider-specific fields merged verbatim into the request body
    /// (an escape hatch), e.g. `logit_bias`, `safetySettings`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, serde_json::Value>,
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
    /// Per-agent reasoning override. When absent, the catalog's reasoning flag
    /// and dialect drive the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningPref>,
    /// Per-agent sampling override. When absent, no sampling params are sent and
    /// the model uses its own defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingPref>,
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
    /// Path to the default config file: `<config_home>/config.toml`
    /// (`~/.shirl/config.toml` by default).
    pub fn default_path() -> Result<PathBuf> {
        Ok(crate::paths::config_home()?.join("config.toml"))
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

    /// Resolve the effective reasoning override for an agent kind, falling back
    /// to the default. Returns `None` when neither sets one.
    pub fn reasoning_for(&self, agent: &str) -> Option<&ReasoningPref> {
        self.agents
            .get(agent)
            .and_then(|a| a.reasoning.as_ref())
            .or(self.default.reasoning.as_ref())
    }

    /// Set (or clear with `None`) the reasoning override for an agent kind.
    ///
    /// `"main"` writes the default entry; other agents write their own override,
    /// inheriting the default provider/model when no agent entry exists yet.
    pub fn set_reasoning(&mut self, agent: &str, pref: Option<ReasoningPref>) {
        if agent == "main" {
            self.default.reasoning = pref;
            return;
        }
        let provider = self.provider_for(agent).to_string();
        let model = self.model_for(agent).to_string();
        let entry = self
            .agents
            .entry(agent.to_string())
            .or_insert_with(|| AgentModelConfig {
                provider,
                model,
                web_search: None,
                reasoning: None,
                sampling: None,
            });
        entry.reasoning = pref;
    }

    /// Resolve the effective sampling override for an agent kind, falling back
    /// to the default. Returns `None` when neither sets one.
    pub fn sampling_for(&self, agent: &str) -> Option<&SamplingPref> {
        self.agents
            .get(agent)
            .and_then(|a| a.sampling.as_ref())
            .or(self.default.sampling.as_ref())
    }

    /// Update the config for a specific agent kind.
    pub fn set_agent_model(
        &mut self,
        agent: &str,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) {
        let web_search = self.agents.get(agent).and_then(|a| a.web_search.clone());
        let reasoning = self.agents.get(agent).and_then(|a| a.reasoning.clone());
        let sampling = self.agents.get(agent).and_then(|a| a.sampling.clone());
        self.agents.insert(
            agent.to_string(),
            AgentModelConfig {
                provider: provider.into(),
                model: model.into(),
                web_search,
                reasoning,
                sampling,
            },
        );
    }

    /// Update the default config.
    pub fn set_default(&mut self, provider: impl Into<String>, model: impl Into<String>) {
        let web_search = self.default.web_search.take();
        let reasoning = self.default.reasoning.take();
        let sampling = self.default.sampling.take();
        self.default = AgentModelConfig {
            provider: provider.into(),
            model: model.into(),
            web_search,
            reasoning,
            sampling,
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
    fn reasoning_pref_round_trips_through_toml() {
        let config = ShirlConfig {
            default: AgentModelConfig {
                provider: "cerebras".to_string(),
                model: "gpt-oss-120b".to_string(),
                web_search: None,
                reasoning: Some(ReasoningPref {
                    enabled: Some(true),
                    effort: Some("high".to_string()),
                    budget_tokens: None,
                }),
                sampling: None,
            },
            ..Default::default()
        };

        let toml = toml::to_string(&config).unwrap();
        let parsed: ShirlConfig = toml::from_str(&toml).unwrap();
        let reasoning = parsed.default.reasoning.expect("reasoning preserved");
        assert_eq!(reasoning.enabled, Some(true));
        assert_eq!(reasoning.effort.as_deref(), Some("high"));
        assert_eq!(reasoning.budget_tokens, None);
    }

    #[test]
    fn sampling_pref_round_trips_through_toml() {
        let mut options = BTreeMap::new();
        options.insert(
            "logit_bias".to_string(),
            serde_json::json!({ "50256": -100 }),
        );
        let mut config = ShirlConfig::default();
        config.set_default("openai", "gpt-4o");
        // Sampling is config-only (no runtime setter); set the field directly.
        config.default.sampling = Some(SamplingPref {
            temperature: Some(0.5),
            top_p: Some(0.25),
            stop: vec!["END".to_string()],
            max_tokens: Some(2048),
            options,
            ..Default::default()
        });

        let toml = toml::to_string(&config).unwrap();
        let parsed: ShirlConfig = toml::from_str(&toml).unwrap();
        let sampling = parsed.sampling_for("main").expect("sampling preserved");
        assert_eq!(sampling.temperature, Some(0.5));
        assert_eq!(sampling.top_p, Some(0.25));
        assert_eq!(sampling.stop, vec!["END".to_string()]);
        assert_eq!(sampling.max_tokens, Some(2048));
        assert_eq!(
            sampling.options.get("logit_bias"),
            Some(&serde_json::json!({ "50256": -100 }))
        );
    }

    #[test]
    fn sampling_pref_absent_by_default() {
        let toml = toml::to_string(&ShirlConfig::default()).unwrap();
        assert!(
            !toml.contains("sampling"),
            "default config should not serialize a sampling section: {toml}"
        );
    }

    #[test]
    fn reasoning_pref_absent_by_default() {
        let toml = toml::to_string(&ShirlConfig::default()).unwrap();
        assert!(
            !toml.contains("reasoning"),
            "default config should not serialize a reasoning section: {toml}"
        );
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
            reasoning: None,
            sampling: None,
        };
        config.agents.insert("plan".to_string(), plan);
        assert_eq!(config.web_search_for("plan"), Some("brave"));
        // "review" has no override - falls back to default.
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
