// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use shirl_core::{AuthStore, ReasoningPref, SamplingPref, ShirlConfig};
use shirl_llm::catalog::{Catalog, Protocol, ReasoningOption};
use shirl_llm::factory::{build_model, ReasoningSettings};
use shirl_llm::SamplingConfig;
use sweet_core::Model;
use sweet_tools::WebSearchBackend;

use shirl_agents::agents::AgentKind;

pub(crate) struct StoredModel {
    pub model: Arc<dyn Model>,
    pub name: String,
    pub context_window: Option<usize>,
}

pub(crate) struct ModelStore {
    models: HashMap<AgentKind, StoredModel>,
}

impl ModelStore {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn get(&self, kind: AgentKind) -> Option<Arc<dyn Model>> {
        self.models.get(&kind).map(|s| s.model.clone())
    }

    pub fn name(&self, kind: AgentKind) -> Option<String> {
        self.models.get(&kind).map(|s| s.name.clone())
    }

    pub fn context_window(&self, kind: AgentKind) -> Option<usize> {
        self.models
            .get(&kind)
            .and_then(|s| s.context_window)
            .filter(|n| *n > 0)
    }

    pub fn insert(
        &mut self,
        kind: AgentKind,
        model: Arc<dyn Model>,
        name: String,
        context_window: Option<usize>,
    ) {
        self.models.insert(
            kind,
            StoredModel {
                model,
                name,
                context_window,
            },
        );
    }
}

pub(crate) struct ResolvedParams {
    pub protocol: Protocol,
    pub base_url: String,
    pub api_key: String,
    pub context_window: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub reasoning: ReasoningSettings,
    pub sampling: SamplingConfig,
}

pub(crate) fn resolve_provider_params(
    provider_id: &str,
    config: &ShirlConfig,
    auth: &AuthStore,
    catalog: &Catalog,
    model_id: &str,
    reasoning_pref: Option<&ReasoningPref>,
    sampling_pref: Option<&SamplingPref>,
) -> Result<ResolvedParams> {
    if let Some(custom) = config.providers.get(provider_id) {
        let protocol = parse_protocol(&custom.protocol)?;
        let api_key = auth.get(provider_id).unwrap_or("").to_string();
        let context_window = lookup_context_window_from_extensions(config, provider_id, model_id);
        return Ok(ResolvedParams {
            protocol,
            base_url: custom.base_url.clone(),
            api_key,
            context_window,
            max_output_tokens: None,
            reasoning: custom_reasoning_settings(reasoning_pref),
            sampling: sampling_from_pref(sampling_pref),
        });
    }

    let provider = catalog
        .get_provider(provider_id)
        .with_context(|| format!("unknown provider: {}", provider_id))?;
    let api_key = auth.get(provider_id).unwrap_or("").to_string();
    let context_window = lookup_context_window(catalog, config, provider_id, model_id);
    let max_output_tokens = lookup_max_output_tokens(catalog, provider_id, model_id);
    let reasoning = resolve_reasoning(catalog, provider_id, model_id, reasoning_pref);
    Ok(ResolvedParams {
        protocol: provider.protocol,
        base_url: provider.base_url.clone(),
        api_key,
        context_window,
        max_output_tokens,
        reasoning,
        sampling: sampling_from_pref(sampling_pref),
    })
}

/// Map a config [`SamplingPref`] into the provider-facing [`SamplingConfig`].
/// Absent or unset fields stay `None`/empty so the model uses its own defaults.
fn sampling_from_pref(pref: Option<&SamplingPref>) -> SamplingConfig {
    match pref {
        None => SamplingConfig::default(),
        Some(p) => SamplingConfig {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            seed: p.seed,
            stop: p.stop.clone(),
            frequency_penalty: p.frequency_penalty,
            presence_penalty: p.presence_penalty,
            max_tokens: p.max_tokens,
            extra: p.options.clone(),
        },
    }
}

fn parse_protocol(s: &str) -> Result<Protocol> {
    match s {
        "openai" => Ok(Protocol::OpenAI),
        "cerebras" => Ok(Protocol::Cerebras),
        "anthropic" => Ok(Protocol::Anthropic),
        "gemini" => Ok(Protocol::Gemini),
        _ => anyhow::bail!("unknown protocol: {}", s),
    }
}

fn lookup_context_window(
    catalog: &Catalog,
    config: &ShirlConfig,
    provider_id: &str,
    model_id: &str,
) -> Option<usize> {
    if let Some(cw) = lookup_context_window_from_extensions(config, provider_id, model_id) {
        return Some(cw);
    }
    catalog
        .get_provider(provider_id)?
        .models
        .iter()
        .find(|m| m.id == model_id)
        .and_then(|m| m.context_window)
}

fn lookup_context_window_from_extensions(
    config: &ShirlConfig,
    provider_id: &str,
    model_id: &str,
) -> Option<usize> {
    config
        .models
        .get(provider_id)?
        .get(model_id)?
        .context_window
}

/// The model's maximum output tokens (models.dev `limit.output`), used to set
/// the per-model output cap for Anthropic/Gemini.
fn lookup_max_output_tokens(catalog: &Catalog, provider_id: &str, model_id: &str) -> Option<usize> {
    catalog
        .get_provider(provider_id)?
        .models
        .iter()
        .find(|m| m.id == model_id)
        .and_then(|m| m.max_output_tokens)
}

/// The wire protocol for a provider, whether a custom `[providers.*]` entry or a
/// catalog-defined one. Used to give protocol-accurate `/reasoning` feedback.
pub(crate) fn lookup_protocol(
    catalog: &Catalog,
    config: &ShirlConfig,
    provider_id: &str,
) -> Option<Protocol> {
    if let Some(custom) = config.providers.get(provider_id) {
        return parse_protocol(&custom.protocol).ok();
    }
    catalog.get_provider(provider_id).map(|p| p.protocol)
}

/// Reasoning dialects a catalog model advertises, for validating `/reasoning`
/// input and showing capability hints. Empty when the model/provider is unknown.
pub(crate) fn lookup_reasoning_options(
    catalog: &Catalog,
    provider_id: &str,
    model_id: &str,
) -> Vec<ReasoningOption> {
    catalog
        .get_provider(provider_id)
        .and_then(|p| p.models.iter().find(|m| m.id == model_id))
        .map(|m| m.reasoning_options.clone())
        .unwrap_or_default()
}

/// A compact, human-readable summary of a model's reasoning dialect(s), e.g.
/// `on/off, effort[low/medium/high]`. Used for the picker hint and `/reasoning`
/// help. Empty options render as `on/off` (a plain enable/disable override).
pub(crate) fn reasoning_capability_summary(options: &[ReasoningOption]) -> String {
    if options.is_empty() {
        return "on/off".to_string();
    }
    options
        .iter()
        .map(|o| match o {
            ReasoningOption::Toggle => "on/off".to_string(),
            ReasoningOption::Effort { values } => format!("effort[{}]", values.join("/")),
            ReasoningOption::BudgetTokens { min, max } => {
                let range = match (min, max) {
                    (Some(a), Some(b)) => format!("{a}..{b}"),
                    (Some(a), None) => format!(">={a}"),
                    (None, Some(b)) => format!("<={b}"),
                    (None, None) => "any".to_string(),
                };
                format!("budget[{range}]")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Merge the catalog's reasoning flag + dialect with any user override into the
/// [`ReasoningSettings`] the factory consumes.
fn resolve_reasoning(
    catalog: &Catalog,
    provider_id: &str,
    model_id: &str,
    pref: Option<&ReasoningPref>,
) -> ReasoningSettings {
    let model = catalog
        .get_provider(provider_id)
        .and_then(|p| p.models.iter().find(|m| m.id == model_id));
    let catalog_reasoning = model.map(|m| m.reasoning).unwrap_or(false);
    let options = model
        .map(|m| m.reasoning_options.clone())
        .unwrap_or_default();
    ReasoningSettings {
        enabled: pref.and_then(|p| p.enabled).unwrap_or(catalog_reasoning),
        options,
        effort: pref.and_then(|p| p.effort.clone()),
        budget_tokens: pref.and_then(|p| p.budget_tokens),
    }
}

/// Reasoning settings for a custom (`[providers.*]`) provider, which has no
/// catalog dialect. The dialect is synthesized from whichever override the user
/// set so the factory can still honor it.
fn custom_reasoning_settings(pref: Option<&ReasoningPref>) -> ReasoningSettings {
    let mut options = Vec::new();
    if let Some(p) = pref {
        if p.effort.is_some() {
            options.push(ReasoningOption::Effort { values: vec![] });
        }
        if p.budget_tokens.is_some() {
            options.push(ReasoningOption::BudgetTokens {
                min: None,
                max: None,
            });
        }
        if options.is_empty() && p.enabled == Some(true) {
            options.push(ReasoningOption::Toggle);
        }
    }
    ReasoningSettings {
        enabled: pref.and_then(|p| p.enabled).unwrap_or(false),
        options,
        effort: pref.and_then(|p| p.effort.clone()),
        budget_tokens: pref.and_then(|p| p.budget_tokens),
    }
}

pub(crate) fn build_web_search_backend(
    provider_id: &str,
    api_key: &str,
) -> Option<Arc<dyn WebSearchBackend>> {
    match provider_id {
        "tavily" => Some(Arc::new(sweet_tools::TavilyBackend::new(api_key))),
        "brave" => Some(Arc::new(sweet_tools::BraveBackend::new(api_key))),
        _ => None,
    }
}

pub(crate) async fn resolve_web_search(
    kind: AgentKind,
    config: &tokio::sync::Mutex<ShirlConfig>,
    auth: &tokio::sync::Mutex<AuthStore>,
) -> Option<Arc<dyn WebSearchBackend>> {
    let agent_name = match kind {
        AgentKind::Main => "main",
        AgentKind::Plan => "plan",
        AgentKind::Review => "review",
    };
    let config_guard = config.lock().await;
    let auth_guard = auth.lock().await;
    let provider_id = config_guard.web_search_for(agent_name)?;
    let key = auth_guard.get_web_search_key(provider_id)?;
    build_web_search_backend(provider_id, key)
}

pub(crate) async fn load_agent_model(
    store: &mut ModelStore,
    kind: AgentKind,
    provider: &str,
    model: &str,
    config: &ShirlConfig,
    auth: &AuthStore,
    catalog: &Catalog,
) -> Result<Option<usize>> {
    let pref = config.reasoning_for(kind.display_name());
    let sampling_pref = config.sampling_for(kind.display_name());
    let params =
        resolve_provider_params(provider, config, auth, catalog, model, pref, sampling_pref)?;
    let built = build_model(
        params.protocol,
        model,
        &params.base_url,
        &params.api_key,
        params.context_window,
        params.max_output_tokens,
        &params.reasoning,
        &params.sampling,
    )?;
    store.insert(
        kind,
        built,
        format!("{}/{}", provider, model),
        params.context_window,
    );
    Ok(params.context_window)
}
