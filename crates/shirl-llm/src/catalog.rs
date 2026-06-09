// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Model catalog backed by [models.dev](https://models.dev).
//!
//! Fetches `https://models.dev/api.json`, caches it locally, and parses it
//! into a structured catalog of providers and models. Only providers whose
//! wire protocol maps to one of the three supported protocols (OpenAI,
//! Anthropic, Gemini) are retained. Models are filtered to those that support
//! tool calling.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const CACHE_FILENAME: &str = "models-dev.json";

/// Wire protocol supported by shirl.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    OpenAI,
    Anthropic,
    Gemini,
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Protocol::OpenAI => write!(f, "openai"),
            Protocol::Anthropic => write!(f, "anthropic"),
            Protocol::Gemini => write!(f, "gemini"),
        }
    }
}

/// A single model from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub context_window: Option<usize>,
    pub reasoning: bool,
    /// Whether the model accepts image inputs.
    #[serde(default)]
    pub vision: bool,
}

/// A single provider from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogProvider {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub protocol: Protocol,
    pub env: Vec<String>,
    pub models: Vec<CatalogModel>,
}

/// The full catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub providers: Vec<CatalogProvider>,
    pub fetched_at: SystemTime,
}

impl Default for Catalog {
    fn default() -> Self {
        Self {
            providers: Vec::new(),
            fetched_at: SystemTime::UNIX_EPOCH,
        }
    }
}

/// Raw serde types matching the models.dev JSON shape.
mod raw {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Debug, Deserialize)]
    pub struct ModelsDev(HashMap<String, Provider>);

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Provider {
        pub id: String,
        pub name: String,
        pub npm: String,
        #[serde(default)]
        pub api: Option<String>,
        #[serde(default)]
        pub env: Vec<String>,
        #[serde(default)]
        pub models: HashMap<String, Model>,
    }

    #[derive(Debug, Deserialize)]
    pub struct Model {
        pub id: String,
        #[serde(default)]
        pub name: String,
        #[serde(default)]
        pub limit: Option<Limit>,
        #[serde(default)]
        pub tool_call: Option<bool>,
        #[serde(default)]
        pub reasoning: bool,
        #[serde(default)]
        pub modalities: Option<Modalities>,
    }

    #[derive(Debug, Deserialize)]
    pub struct Modalities {
        #[serde(default)]
        pub input: Vec<String>,
    }

    impl Model {
        pub fn accepts_image_input(&self) -> bool {
            self.modalities
                .as_ref()
                .is_some_and(|m| m.input.iter().any(|s| s == "image"))
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct Limit {
        pub context: Option<u64>,
    }

    impl ModelsDev {
        pub fn into_providers(self) -> Vec<super::CatalogProvider> {
            let mut providers: Vec<super::CatalogProvider> = self
                .0
                .into_values()
                .filter_map(|p| {
                    let protocol = super::protocol_from_npm(&p.npm)?;
                    let base_url = p
                        .api
                        .clone()
                        .or_else(|| super::known_base_url(&p.id).map(|s| s.to_string()))?;

                    let mut models: Vec<super::CatalogModel> = p
                        .models
                        .values()
                        .filter(|m| m.tool_call.unwrap_or(true))
                        .map(|m| super::CatalogModel {
                            id: m.id.clone(),
                            name: if m.name.is_empty() {
                                m.id.clone()
                            } else {
                                m.name.clone()
                            },
                            context_window: m
                                .limit
                                .as_ref()
                                .and_then(|l| l.context)
                                .map(|n| n as usize),
                            reasoning: m.reasoning,
                            vision: m.accepts_image_input(),
                        })
                        .collect();

                    if models.is_empty() {
                        return None;
                    }

                    // `p.models` is a HashMap, so iteration order is otherwise
                    // non-deterministic; sort by id for a stable catalog.
                    models.sort_by(|a, b| a.id.cmp(&b.id));

                    Some(super::CatalogProvider {
                        id: p.id,
                        name: p.name,
                        base_url,
                        protocol,
                        env: p.env,
                        models,
                    })
                })
                .collect();

            providers.sort_by(|a, b| a.id.cmp(&b.id));
            providers
        }
    }
}

fn protocol_from_npm(npm: &str) -> Option<Protocol> {
    match npm {
        n if n.contains("anthropic") => Some(Protocol::Anthropic),
        "@ai-sdk/google" | "@ai-sdk/google-vertex" => Some(Protocol::Gemini),
        _ => {
            let openai_like = [
                "@ai-sdk/openai",
                "@ai-sdk/openai-compatible",
                "@ai-sdk/cerebras",
                "@ai-sdk/groq",
                "@ai-sdk/deepinfra",
                "@ai-sdk/mistral",
                "@ai-sdk/xai",
                "@ai-sdk/perplexity",
                "@ai-sdk/togetherai",
                "@ai-sdk/cohere",
                "@ai-sdk/azure",
                "@openrouter/ai-sdk-provider",
                "@ai-sdk/gateway",
                "@ai-sdk/vercel",
                "@aiihubmix/ai-sdk-provider",
                "ai-gateway-provider",
                "venice-ai-sdk-provider",
                "gitlab-ai-provider",
            ];
            if openai_like.contains(&npm) {
                Some(Protocol::OpenAI)
            } else {
                None
            }
        }
    }
}

/// Fallback base URL for providers whose models.dev entry lacks an `api` field.
///
/// Many major providers (OpenAI, Cerebras, Groq, etc.) ship their own SDK/npm
/// package rather than a generic OpenAI-compatible endpoint, so models.dev
/// omits `api`. We hardcode the well-known URL so these providers still appear
/// in the catalog.
///
/// Providers returning `None` require non-standard auth flows (e.g. Azure
/// Active Directory) or customer-specific subdomain URLs (e.g. Google Vertex
/// `{region}-aiplatform.googleapis.com`) that can't be configured from a
/// single API key. They are excluded from the catalog.
fn known_base_url(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        "anthropic" => Some("https://api.anthropic.com/v1"),
        "openai" => Some("https://api.openai.com/v1"),
        "cerebras" => Some("https://api.cerebras.ai/v1"),
        "groq" => Some("https://api.groq.com/openai/v1"),
        "deepinfra" => Some("https://api.deepinfra.com/v1/openai"),
        "mistral" => Some("https://api.mistral.ai/v1"),
        "xai" => Some("https://api.x.ai/v1"),
        "perplexity" => Some("https://api.perplexity.ai"),
        "togetherai" => Some("https://api.together.xyz/v1"),
        "cohere" => Some("https://api.cohere.com/v2"),
        "azure" | "azure-cognitive-services" => None,
        "vercel" | "v0" => None,
        "aihubmix" => None,
        "cloudflare-ai-gateway" => None,
        "venice" => Some("https://api.venice.ai/api/v1"),
        "gitlab" => None,
        "google" => Some("https://generativelanguage.googleapis.com"),
        "google-vertex" | "google-vertex-anthropic" => None,
        _ => None,
    }
}

impl Catalog {
    pub fn cache_path() -> Result<PathBuf> {
        let home =
            dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
        Ok(home.join(".shirl").join("cache").join(CACHE_FILENAME))
    }

    pub async fn load(http: &reqwest::Client) -> Result<Self> {
        let cache_path = Self::cache_path()?;

        if let Ok(cached) = Self::load_from_cache(&cache_path) {
            if let Ok(elapsed) = cached.fetched_at.elapsed() {
                if elapsed < CACHE_TTL {
                    return Ok(cached);
                }
            }
        }

        match Self::fetch(http).await {
            Ok(catalog) => {
                if let Err(e) = catalog.save_to_cache(&cache_path) {
                    tracing::warn!("failed to cache catalog: {e}");
                }
                Ok(catalog)
            }
            Err(fetch_err) => {
                if let Ok(cached) = Self::load_from_cache(&cache_path) {
                    tracing::warn!("failed to refresh catalog, using stale cache: {fetch_err}");
                    Ok(cached)
                } else {
                    Err(fetch_err.context("failed to fetch model catalog and no cache available"))
                }
            }
        }
    }

    fn load_from_cache(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| "reading cached catalog")?;
        let catalog: Catalog =
            serde_json::from_str(&text).with_context(|| "parsing cached catalog")?;
        Ok(catalog)
    }

    fn save_to_cache(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    async fn fetch(http: &reqwest::Client) -> Result<Self> {
        let resp = http
            .get(MODELS_DEV_URL)
            .send()
            .await
            .context("fetching models.dev catalog")?
            .error_for_status()
            .context("models.dev returned error")?;

        let raw_text = resp.text().await.context("reading models.dev response")?;
        let raw: raw::ModelsDev =
            serde_json::from_str(&raw_text).context("parsing models.dev catalog")?;

        let providers = raw.into_providers();
        tracing::info!("loaded {} providers from models.dev", providers.len());

        Ok(Catalog {
            providers,
            fetched_at: SystemTime::now(),
        })
    }

    pub fn get_provider(&self, id: &str) -> Option<&CatalogProvider> {
        self.providers.iter().find(|p| p.id == id)
    }

    pub fn provider_ids(&self) -> impl Iterator<Item = &str> {
        self.providers.iter().map(|p| p.id.as_str())
    }

    pub fn providers_with_auth(
        &self,
        is_connected: impl Fn(&str) -> bool,
    ) -> Vec<&CatalogProvider> {
        self.providers
            .iter()
            .filter(|p| is_connected(&p.id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_from_npm_known_values() {
        assert_eq!(protocol_from_npm("@ai-sdk/openai"), Some(Protocol::OpenAI));
        assert_eq!(
            protocol_from_npm("@ai-sdk/openai-compatible"),
            Some(Protocol::OpenAI)
        );
        assert_eq!(
            protocol_from_npm("@ai-sdk/anthropic"),
            Some(Protocol::Anthropic)
        );
        assert_eq!(protocol_from_npm("@ai-sdk/google"), Some(Protocol::Gemini));
        assert_eq!(
            protocol_from_npm("@openrouter/ai-sdk-provider"),
            Some(Protocol::OpenAI)
        );
        assert_eq!(
            protocol_from_npm("@ai-sdk/cerebras"),
            Some(Protocol::OpenAI)
        );
        assert_eq!(protocol_from_npm("@ai-sdk/groq"), Some(Protocol::OpenAI));
    }

    #[test]
    fn protocol_from_npm_skips_unknown() {
        assert_eq!(protocol_from_npm("@ai-sdk/amazon-bedrock"), None);
        assert_eq!(protocol_from_npm("@jerome/benoit/sap-ai-provider-v2"), None);
    }

    #[test]
    fn parse_minimal_models_dev_json() {
        let json = r#"{
            "test-provider": {
                "id": "test-provider",
                "name": "Test Provider",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.test.example/v1",
                "env": ["TEST_API_KEY"],
                "models": {
                    "model-a": {
                        "id": "model-a",
                        "name": "Model A",
                        "tool_call": true,
                        "limit": { "context": 128000 }
                    },
                    "model-b-no-tools": {
                        "id": "model-b-no-tools",
                        "name": "Model B (no tools)",
                        "tool_call": false
                    },
                    "model-c": {
                        "id": "model-c",
                        "name": "Model C",
                        "tool_call": true
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert_eq!(providers.len(), 1);

        let p = &providers[0];
        assert_eq!(p.id, "test-provider");
        assert_eq!(p.name, "Test Provider");
        assert_eq!(p.base_url, "https://api.test.example/v1");
        assert_eq!(p.protocol, Protocol::OpenAI);
        assert_eq!(p.env, vec!["TEST_API_KEY"]);

        assert_eq!(p.models.len(), 2);
        let ids: Vec<&str> = p.models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"model-a"));
        assert!(ids.contains(&"model-c"));
        let model_a = p.models.iter().find(|m| m.id == "model-a").unwrap();
        assert_eq!(model_a.context_window, Some(128000));
        assert!(!model_a.reasoning);
        let model_c = p.models.iter().find(|m| m.id == "model-c").unwrap();
        assert!(!model_c.reasoning);
    }

    #[test]
    fn models_are_sorted_by_id() {
        let json = r#"{
            "p": {
                "id": "p",
                "name": "P",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.test.example/v1",
                "env": [],
                "models": {
                    "gamma": { "id": "gamma", "name": "Gamma", "tool_call": true },
                    "alpha": { "id": "alpha", "name": "Alpha", "tool_call": true },
                    "beta":  { "id": "beta",  "name": "Beta",  "tool_call": true }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        let ids: Vec<&str> = providers[0].models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn provider_without_tool_call_models_is_excluded() {
        let json = r#"{
            "embed-only": {
                "id": "embed-only",
                "name": "Embed Only",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.test.example/v1",
                "env": [],
                "models": {
                    "emb": {
                        "id": "emb",
                        "name": "Embedding Model",
                        "tool_call": false
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert!(providers.is_empty());
    }

    #[test]
    fn unsupported_npm_is_excluded() {
        let json = r#"{
            "bedrock": {
                "id": "bedrock",
                "name": "Bedrock",
                "npm": "@ai-sdk/amazon-bedrock",
                "api": "https://bedrock.example.com/v1",
                "env": [],
                "models": {
                    "m": {
                        "id": "m",
                        "name": "M",
                        "tool_call": true
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert!(providers.is_empty());
    }

    #[test]
    fn provider_without_api_field_is_excluded() {
        let json = r#"{
            "no-url": {
                "id": "no-url",
                "name": "No URL",
                "npm": "@ai-sdk/openai-compatible",
                "env": [],
                "models": {
                    "m": {
                        "id": "m",
                        "name": "M",
                        "tool_call": true
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert!(providers.is_empty());
    }

    #[test]
    fn catalog_get_provider_finds_by_id() {
        let catalog = Catalog {
            providers: vec![CatalogProvider {
                id: "openai".to_string(),
                name: "OpenAI".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                protocol: Protocol::OpenAI,
                env: vec!["OPENAI_API_KEY".to_string()],
                models: vec![],
            }],
            fetched_at: SystemTime::now(),
        };
        assert!(catalog.get_provider("openai").is_some());
        assert!(catalog.get_provider("anthropic").is_none());
    }

    #[test]
    fn model_without_tool_call_field_is_included() {
        let json = r#"{
            "unknown-tools": {
                "id": "unknown-tools",
                "name": "Unknown Tools",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.test.example/v1",
                "env": [],
                "models": {
                    "m": {
                        "id": "m",
                        "name": "M"
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].models.len(), 1);
        assert_eq!(providers[0].models[0].id, "m");
        assert!(!providers[0].models[0].reasoning);
    }

    #[test]
    fn reasoning_field_is_parsed() {
        let json = r#"{
            "reasoning-provider": {
                "id": "reasoning-provider",
                "name": "Reasoning Provider",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.test.example/v1",
                "env": [],
                "models": {
                    "thinker": {
                        "id": "thinker",
                        "name": "Thinker",
                        "tool_call": true,
                        "reasoning": true
                    },
                    "basic": {
                        "id": "basic",
                        "name": "Basic",
                        "tool_call": true,
                        "reasoning": false
                    },
                    "defaulted": {
                        "id": "defaulted",
                        "name": "Defaulted",
                        "tool_call": true
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert_eq!(providers.len(), 1);

        let thinker = providers[0]
            .models
            .iter()
            .find(|m| m.id == "thinker")
            .unwrap();
        assert!(thinker.reasoning);

        let basic = providers[0]
            .models
            .iter()
            .find(|m| m.id == "basic")
            .unwrap();
        assert!(!basic.reasoning);

        let defaulted = providers[0]
            .models
            .iter()
            .find(|m| m.id == "defaulted")
            .unwrap();
        assert!(!defaulted.reasoning);
    }

    #[test]
    fn known_base_url_fallback() {
        let json = r#"{
            "cerebras": {
                "id": "cerebras",
                "name": "Cerebras",
                "npm": "@ai-sdk/cerebras",
                "env": ["CEREBRAS_API_KEY"],
                "models": {
                    "llama3.1-8b": {
                        "id": "llama3.1-8b",
                        "name": "Llama 3.1 8B",
                        "tool_call": true
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].base_url, "https://api.cerebras.ai/v1");
    }

    #[test]
    fn anthropic_parsed_without_api_field() {
        let json = r#"{
            "anthropic": {
                "id": "anthropic",
                "name": "Anthropic",
                "npm": "@ai-sdk/anthropic",
                "env": ["ANTHROPIC_API_KEY"],
                "models": {
                    "claude-sonnet-4-20250514": {
                        "id": "claude-sonnet-4-20250514",
                        "name": "Claude Sonnet 4"
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].id, "anthropic");
        assert_eq!(providers[0].protocol, Protocol::Anthropic);
        assert_eq!(providers[0].base_url, "https://api.anthropic.com/v1");
        assert_eq!(providers[0].env, vec!["ANTHROPIC_API_KEY"]);
        assert_eq!(providers[0].models.len(), 1);
    }

    #[test]
    fn cache_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models-dev.json");

        let catalog = Catalog {
            providers: vec![CatalogProvider {
                id: "openai".to_string(),
                name: "OpenAI".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                protocol: Protocol::OpenAI,
                env: vec!["OPENAI_API_KEY".to_string()],
                models: vec![CatalogModel {
                    id: "gpt-4o".to_string(),
                    name: "GPT-4o".to_string(),
                    context_window: Some(128000),
                    reasoning: false,
                    vision: false,
                }],
            }],
            fetched_at: SystemTime::now(),
        };

        catalog.save_to_cache(&path).unwrap();
        let loaded = Catalog::load_from_cache(&path).unwrap();
        assert_eq!(loaded.providers.len(), 1);
        assert_eq!(loaded.providers[0].id, "openai");
        assert_eq!(loaded.providers[0].models.len(), 1);
        assert_eq!(loaded.providers[0].models[0].id, "gpt-4o");
    }

    #[test]
    fn vision_parsed_from_modalities() {
        let json = r#"{
            "vision-provider": {
                "id": "vision-provider",
                "name": "Vision Provider",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.test.example/v1",
                "env": [],
                "models": {
                    "see": {
                        "id": "see",
                        "name": "See",
                        "tool_call": true,
                        "modalities": { "input": ["text", "image"] }
                    },
                    "blind": {
                        "id": "blind",
                        "name": "Blind",
                        "tool_call": true,
                        "modalities": { "input": ["text"] }
                    },
                    "no-mod": {
                        "id": "no-mod",
                        "name": "No Mod",
                        "tool_call": true
                    }
                }
            }
        }"#;

        let raw: raw::ModelsDev = serde_json::from_str(json).unwrap();
        let providers = raw.into_providers();
        assert_eq!(providers.len(), 1);

        let see = providers[0].models.iter().find(|m| m.id == "see").unwrap();
        assert!(see.vision);

        let blind = providers[0]
            .models
            .iter()
            .find(|m| m.id == "blind")
            .unwrap();
        assert!(!blind.vision);

        let blind_model = providers[0]
            .models
            .iter()
            .find(|m| m.id == "no-mod")
            .unwrap();
        assert!(!blind_model.vision);
    }
}
