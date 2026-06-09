// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use shirl_core::{AuthStore, ShirlConfig};
use shirl_llm::catalog::{Catalog, Protocol};
use shirl_llm::factory::build_model;
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
    pub reasoning: bool,
}

pub(crate) fn resolve_provider_params(
    provider_id: &str,
    config: &ShirlConfig,
    auth: &AuthStore,
    catalog: &Catalog,
    model_id: &str,
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
            reasoning: false,
        });
    }

    let provider = catalog
        .get_provider(provider_id)
        .with_context(|| format!("unknown provider: {}", provider_id))?;
    let api_key = auth.get(provider_id).unwrap_or("").to_string();
    let context_window = lookup_context_window(catalog, config, provider_id, model_id);
    let reasoning = lookup_reasoning(catalog, provider_id, model_id);
    Ok(ResolvedParams {
        protocol: provider.protocol,
        base_url: provider.base_url.clone(),
        api_key,
        context_window,
        reasoning,
    })
}

fn parse_protocol(s: &str) -> Result<Protocol> {
    match s {
        "openai" => Ok(Protocol::OpenAI),
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

fn lookup_reasoning(catalog: &Catalog, provider_id: &str, model_id: &str) -> bool {
    catalog
        .get_provider(provider_id)
        .and_then(|p| p.models.iter().find(|m| m.id == model_id))
        .map(|m| m.reasoning)
        .unwrap_or(false)
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
    let params = resolve_provider_params(provider, config, auth, catalog, model)?;
    let built = build_model(
        params.protocol,
        model,
        &params.base_url,
        &params.api_key,
        params.context_window,
        params.reasoning,
    )?;
    store.insert(
        kind,
        built,
        format!("{}/{}", provider, model),
        params.context_window,
    );
    Ok(params.context_window)
}
