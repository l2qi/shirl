// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Factory for building provider trait objects.
//!
//! The factory constructs an `Arc<dyn Model>` from a provider id, model id,
//! API key, base URL, and protocol. It dispatches to the correct concrete
//! provider type in `sweet-llm`.
//!
//! When a context window is supplied (discovered by the catalog) the built
//! model is wrapped so it reports that value.

use std::sync::Arc;

use sweet_core::{async_trait, Message, Model, Result, StreamSink, ToolSpec};
use sweet_llm::{ReasoningConfig, SamplingConfig};

use crate::catalog::{Protocol, ReasoningOption};

/// Default thinking-token budget used when enabling a budget-dialect model that
/// advertises no `min` (mirrors the floor most providers accept).
const DEFAULT_THINKING_BUDGET: u32 = 2048;

/// Resolved reasoning configuration for a model: the catalog's `reasoning` flag
/// and dialect (`reasoning_options`) plus any user overrides. The factory turns
/// this into the dialect-correct [`ReasoningConfig`] for the target provider.
#[derive(Debug, Clone, Default)]
pub struct ReasoningSettings {
    /// Whether reasoning should be enabled (catalog flag, possibly user-forced).
    pub enabled: bool,
    /// Reasoning dialects the model advertises (from models.dev).
    pub options: Vec<ReasoningOption>,
    /// User-chosen effort level, if any (validated against `options`).
    pub effort: Option<String>,
    /// User-chosen thinking budget, if any (validated against `options`).
    pub budget_tokens: Option<u32>,
}

/// Whether reasoning can be explicitly turned *off* for a `protocol` model with
/// these dialect `options`. Anthropic/Gemini always can (an explicit disable
/// toggle); an OpenAI/Cerebras model only if it advertises a toggle or an effort
/// level of `none`. Otherwise the model reasons by default with no off-switch.
///
/// This is the same predicate `plan_reasoning` uses for its disable path, so
/// callers (e.g. shirl's `/reasoning off`) can detect a no-op disable without
/// re-deriving - and drifting from - the rule.
pub fn can_disable_reasoning(protocol: Protocol, options: &[ReasoningOption]) -> bool {
    let has_toggle = options.iter().any(|o| matches!(o, ReasoningOption::Toggle));
    let effort_allows_none = options.iter().any(
        |o| matches!(o, ReasoningOption::Effort { values } if values.iter().any(|v| v == "none")),
    );
    let explicit_enable = matches!(protocol, Protocol::Anthropic | Protocol::Gemini);
    has_toggle || explicit_enable || effort_allows_none
}

/// Translate resolved [`ReasoningSettings`] into the dialect-correct
/// [`ReasoningConfig`] for `protocol`, or `None` to send no reasoning parameter.
///
/// The dispatch is dialect-aware so it never sends an unsupported control (e.g.
/// the `thinking` object to an effort-only Cerebras model - the original
/// `HTTP 400` bug). It is also provider-aware about the *default* reasoning
/// state: OpenAI/Cerebras effort models reason by default (send nothing),
/// whereas Anthropic/Gemini must be explicitly enabled.
fn plan_reasoning(protocol: Protocol, settings: &ReasoningSettings) -> Option<ReasoningConfig> {
    let has_toggle = settings
        .options
        .iter()
        .any(|o| matches!(o, ReasoningOption::Toggle));
    let has_effort = settings
        .options
        .iter()
        .any(|o| matches!(o, ReasoningOption::Effort { .. }));
    let budget_min = settings.options.iter().find_map(|o| match o {
        ReasoningOption::BudgetTokens { min, .. } => Some(min.unwrap_or(DEFAULT_THINKING_BUDGET)),
        _ => None,
    });

    // Explicit user overrides win when the model supports that dialect.
    if let Some(effort) = &settings.effort {
        if has_effort {
            return Some(ReasoningConfig::Effort(effort.clone()));
        }
        tracing::warn!("model does not support an effort reasoning control; ignoring override");
    }
    if let Some(budget) = settings.budget_tokens {
        if budget_min.is_some() {
            return Some(ReasoningConfig::Budget(budget));
        }
        tracing::warn!("model does not support a thinking budget; ignoring override");
    }

    // Anthropic/Gemini reasoning is off by default and must be explicitly
    // enabled or disabled; OpenAI/Cerebras effort models reason by default.
    let explicit_enable = matches!(protocol, Protocol::Anthropic | Protocol::Gemini);

    if !settings.enabled {
        if !can_disable_reasoning(protocol, &settings.options) {
            // Reasons by default with no off-switch: send nothing.
            return None;
        }
        // Toggle / explicit-enable providers disable via the toggle; an
        // effort-only model with a `none` level disables by selecting it.
        return Some(if has_toggle || explicit_enable {
            ReasoningConfig::Toggle(false)
        } else {
            ReasoningConfig::Effort("none".to_string())
        });
    }

    if has_toggle {
        return Some(ReasoningConfig::Toggle(true));
    }
    // A catalog budget dialect is honored directly here and must NOT be changed
    // to fall through to the `explicit_enable -> Toggle(true)` path below in the
    // hope of giving Anthropic models adaptive thinking. `Protocol::Anthropic`
    // covers EVERY models.dev provider whose npm contains "anthropic" - not just
    // direct Anthropic, but also `google-vertex-anthropic`, `freemodel`, and the
    // MiniMax / Kimi-via-Anthropic-SDK proxies. `budget_tokens` is accepted by
    // all of them; `{type: adaptive}` is valid only on direct-Anthropic Claude
    // 4.6+. `anthropic_reasoning`'s adaptive detection parses only the
    // direct-Anthropic id form, so a Vertex id like `claude-sonnet-4@20250514`
    // (a budget-era 4.0) parses to `None` and would be MISCLASSIFIED as adaptive,
    // earning an HTTP 400 - the very failure this reasoning dispatch exists to
    // avoid. Keeping budget-dialect models on `Budget` here confines adaptive
    // detection to effort-only / toggle-less models (the clean modern
    // direct-Anthropic ids). The cost - direct-Anthropic 4.6, which also
    // advertises a budget dialect, getting a min budget instead of adaptive - is
    // the deliberate safe tradeoff. Don't "fix" it without first making adaptive
    // provider-aware (i.e. gating it to the `anthropic` provider id, not the
    // shared protocol). Verified against models.dev 2026-06-22.
    if let Some(min) = budget_min {
        // Gemini enables dynamic thinking via the toggle; others take a budget.
        return Some(match protocol {
            Protocol::Gemini => ReasoningConfig::Toggle(true),
            _ => ReasoningConfig::Budget(min),
        });
    }
    if explicit_enable {
        Some(ReasoningConfig::Toggle(true))
    } else {
        // Effort-only or no knob: these models reason by default.
        None
    }
}

/// Translate the planned reasoning into the Anthropic-specific variant.
///
/// The cross-provider plan uses [`Toggle(true)`](ReasoningConfig::Toggle) for
/// "reasoning on"; sweet-llm maps that to adaptive thinking, which the
/// budget-dialect Claude models (3.7, 4.0-4.5) reject. For those we send an
/// explicit [`Budget`](ReasoningConfig::Budget) instead. This adaptive-vs-budget
/// knowledge lives here, in the consumer that tracks models - sweet-llm stays
/// catalog-agnostic and never sniffs the model name.
fn anthropic_reasoning(plan: Option<ReasoningConfig>, model_id: &str) -> Option<ReasoningConfig> {
    match plan {
        Some(ReasoningConfig::Toggle(true)) if !anthropic_uses_adaptive_thinking(model_id) => {
            Some(ReasoningConfig::Budget(DEFAULT_THINKING_BUDGET))
        }
        other => other,
    }
}

/// Whether an Anthropic model speaks the adaptive thinking dialect
/// (`thinking: {type: adaptive}`) rather than the explicit `budget_tokens`
/// dialect. Adaptive is Claude 4.6+ plus the fable/mythos families; 3.7 and
/// 4.0-4.5 use a budget. models.dev does not encode this, so we recognize it by
/// model id. Unknown ids default to adaptive, the going-forward Anthropic shape.
///
/// This parser assumes the *direct-Anthropic* id form (`claude-<fam>-<maj>-<min>`).
/// It is deliberately only reached for effort-only / toggle-less models, because
/// `plan_reasoning` routes any budget-dialect model straight to `Budget` first
/// (see the note there). That guard matters: other `Protocol::Anthropic`
/// providers use id forms this can't parse - e.g. Vertex's
/// `claude-sonnet-4@20250514` yields `None` and would be wrongly treated as
/// adaptive - so adaptive must never be the fallback for those. Don't relax the
/// budget guard to widen what reaches here without making adaptive
/// provider-aware.
fn anthropic_uses_adaptive_thinking(model_id: &str) -> bool {
    if model_id.contains("fable") || model_id.contains("mythos") {
        return true;
    }
    match anthropic_version(model_id) {
        Some(version) => version >= (4, 6),
        None => true,
    }
}

/// Extract `(major, minor)` from a Claude model id: `claude-opus-4-8` -> `(4, 8)`,
/// `claude-sonnet-4-20250514` -> `(4, 0)` (an 8-digit date in the minor slot is
/// treated as minor 0). `None` if no numeric version segment is found.
fn anthropic_version(model_id: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = model_id.split('-').collect();
    let i = parts.iter().position(|p| p.parse::<u32>().is_ok())?;
    let major: u32 = parts[i].parse().ok()?;
    let minor = parts
        .get(i + 1)
        .filter(|s| s.len() == 1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    Some((major, minor))
}

struct WithContextWindow {
    inner: Arc<dyn Model>,
    context_window: usize,
}

#[async_trait]
impl Model for WithContextWindow {
    async fn complete(&self, messages: &[Message], tools: &[ToolSpec]) -> Result<Message> {
        self.inner.complete(messages, tools).await
    }

    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: &mut dyn StreamSink,
    ) -> Result<Message> {
        self.inner.complete_stream(messages, tools, sink).await
    }

    fn context_window(&self) -> Option<usize> {
        Some(self.context_window)
    }
}

fn apply_context_window(model: Arc<dyn Model>, context_window: Option<usize>) -> Arc<dyn Model> {
    match context_window.filter(|n| *n > 0) {
        Some(ctx) => Arc::new(WithContextWindow {
            inner: model,
            context_window: ctx,
        }),
        None => model,
    }
}

/// Build an `Arc<dyn Model>` from a resolved protocol, base URL, and API key.
///
/// `reasoning` carries the catalog's reasoning flag, the model's reasoning
/// dialect(s), and any user overrides; `plan_reasoning` translates it into the
/// dialect-correct [`ReasoningConfig`] for the target provider (or sends no
/// reasoning parameter at all). This is what keeps an effort-only provider like
/// Cerebras from ever receiving the `thinking` object it rejects.
///
/// `max_output_tokens` (models.dev `limit.output`) sets the per-model output cap
/// for the protocols that take one: Anthropic (where it is required and bounds
/// thinking + visible output) and Gemini (`generationConfig.maxOutputTokens`).
/// OpenAI/Cerebras ignore it - those send no `max_tokens` and let the endpoint
/// use the model's natural default, avoiding the reasoning-token truncation a
/// fixed cap would risk.
///
/// `sampling` carries cross-provider generation parameters (temperature, top_p,
/// stop, etc.) plus an `extra` passthrough; each provider applies the subset it
/// supports.
#[allow(clippy::too_many_arguments)]
pub fn build_model(
    protocol: Protocol,
    model_id: &str,
    base_url: &str,
    api_key: &str,
    context_window: Option<usize>,
    max_output_tokens: Option<usize>,
    reasoning: &ReasoningSettings,
    sampling: &SamplingConfig,
) -> Result<Arc<dyn Model>> {
    let plan = plan_reasoning(protocol, reasoning);
    let model: Arc<dyn Model> = match protocol {
        Protocol::OpenAI => {
            let mut p = sweet_llm::OpenAIProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            if let Some(cfg) = plan {
                p = p.with_reasoning(cfg);
            }
            p = p.with_sampling(sampling.clone());
            Arc::new(p)
        }
        Protocol::Cerebras => {
            let mut p = sweet_llm::CerebrasProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            if let Some(cfg) = plan {
                p = p.with_reasoning(cfg);
            }
            p = p.with_sampling(sampling.clone());
            Arc::new(p)
        }
        Protocol::Anthropic => {
            let mut p = sweet_llm::AnthropicProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            if let Some(tokens) = max_output_tokens.filter(|n| *n > 0) {
                p = p.with_max_tokens(tokens);
            }
            // `Toggle(true)` -> adaptive thinking in sweet-llm; budget-dialect
            // Claude models take an explicit budget instead (see
            // `anthropic_reasoning`).
            if let Some(cfg) = anthropic_reasoning(plan, model_id) {
                p = p.with_reasoning(cfg);
            }
            p = p.with_sampling(sampling.clone());
            Arc::new(p)
        }
        Protocol::Gemini => {
            let mut p = sweet_llm::GeminiProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            if let Some(tokens) = max_output_tokens.filter(|n| *n > 0) {
                p = p.with_max_tokens(tokens);
            }
            if let Some(cfg) = plan {
                p = p.with_reasoning(cfg);
            }
            p = p.with_sampling(sampling.clone());
            Arc::new(p)
        }
    };
    Ok(apply_context_window(model, context_window))
}

/// Build an `Arc<dyn Embedder>` for semantic memory recall from a resolved
/// protocol, base URL, and API key.
///
/// Only the OpenAI and Gemini protocols offer embedding endpoints; Anthropic
/// does not, so configs pointing an embedder at an Anthropic-protocol
/// provider are rejected with a clear error.
pub fn build_embedder(
    protocol: Protocol,
    model_id: &str,
    base_url: &str,
    api_key: &str,
) -> anyhow::Result<Arc<dyn sweet_core::Embedder>> {
    match protocol {
        Protocol::OpenAI => {
            let mut e = sweet_llm::OpenAIEmbedder::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                e = e.with_base_url(base_url);
            }
            Ok(Arc::new(e))
        }
        Protocol::Gemini => {
            let mut e = sweet_llm::GeminiEmbedder::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                e = e.with_base_url(base_url);
            }
            Ok(Arc::new(e))
        }
        Protocol::Anthropic => anyhow::bail!(
            "provider protocol `anthropic` has no embeddings API - \
             configure an openai- or gemini-protocol embedder"
        ),
        Protocol::Cerebras => anyhow::bail!(
            "provider protocol `cerebras` has no embeddings API - \
             configure an openai- or gemini-protocol embedder"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModel;

    #[async_trait]
    impl Model for DummyModel {
        async fn complete(&self, _: &[Message], _: &[ToolSpec]) -> Result<Message> {
            Ok(Message::system(String::new()))
        }

        async fn complete_stream(
            &self,
            _: &[Message],
            _: &[ToolSpec],
            _: &mut dyn StreamSink,
        ) -> Result<Message> {
            Ok(Message::system(String::new()))
        }
    }

    #[test]
    fn apply_context_window_wraps_when_positive() {
        let model = apply_context_window(Arc::new(DummyModel), Some(200_000));
        assert_eq!(model.context_window(), Some(200_000));
    }

    #[test]
    fn apply_context_window_passes_through_none_and_zero() {
        assert_eq!(
            apply_context_window(Arc::new(DummyModel), None).context_window(),
            None
        );
        assert_eq!(
            apply_context_window(Arc::new(DummyModel), Some(0)).context_window(),
            None
        );
    }

    fn effort_settings(values: &[&str]) -> ReasoningSettings {
        ReasoningSettings {
            enabled: true,
            options: vec![ReasoningOption::Effort {
                values: values.iter().map(|s| s.to_string()).collect(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn build_model_handles_cerebras_reasoning() {
        // A Cerebras reasoning model must build via the dedicated provider.
        let model = build_model(
            Protocol::Cerebras,
            "zai-glm-4.7",
            "https://api.cerebras.ai/v1",
            "test-key",
            Some(131_072),
            Some(40_960),
            &effort_settings(&["none"]),
            &SamplingConfig::default(),
        )
        .expect("cerebras reasoning model builds");
        assert_eq!(model.context_window(), Some(131_072));
    }

    #[test]
    fn build_model_constructs_each_protocol() {
        for protocol in [
            Protocol::OpenAI,
            Protocol::Cerebras,
            Protocol::Anthropic,
            Protocol::Gemini,
        ] {
            let model = build_model(
                protocol,
                "test-model",
                "https://example.test/v1",
                "test-key",
                Some(123_000),
                Some(8_192),
                &ReasoningSettings::default(),
                &SamplingConfig::default(),
            )
            .expect("provider builds");
            assert_eq!(model.context_window(), Some(123_000));
        }
    }

    fn toggle_settings(enabled: bool) -> ReasoningSettings {
        ReasoningSettings {
            enabled,
            options: vec![ReasoningOption::Toggle],
            ..Default::default()
        }
    }

    #[test]
    fn plan_toggle_dialect_enables_and_disables() {
        assert_eq!(
            plan_reasoning(Protocol::OpenAI, &toggle_settings(true)),
            Some(ReasoningConfig::Toggle(true))
        );
        assert_eq!(
            plan_reasoning(Protocol::OpenAI, &toggle_settings(false)),
            Some(ReasoningConfig::Toggle(false))
        );
    }

    #[test]
    fn can_disable_reasoning_by_protocol_and_dialect() {
        let effort = |vals: &[&str]| ReasoningOption::Effort {
            values: vals.iter().map(|s| s.to_string()).collect(),
        };
        // Anthropic/Gemini always accept an explicit off, whatever the dialect.
        assert!(can_disable_reasoning(Protocol::Anthropic, &[]));
        assert!(can_disable_reasoning(
            Protocol::Gemini,
            &[effort(&["low", "high"])]
        ));
        // OpenAI/Cerebras can disable only with a toggle or an effort `none`.
        assert!(can_disable_reasoning(
            Protocol::OpenAI,
            &[ReasoningOption::Toggle]
        ));
        assert!(can_disable_reasoning(
            Protocol::Cerebras,
            &[effort(&["none", "low", "high"])]
        ));
        // Effort-by-default with no `none` and no toggle: cannot be turned off.
        assert!(!can_disable_reasoning(
            Protocol::OpenAI,
            &[effort(&["low", "high"])]
        ));
        // A reasoning model exposing no knob at all: cannot be turned off.
        assert!(!can_disable_reasoning(Protocol::Cerebras, &[]));
    }

    #[test]
    fn plan_effort_dialect_reasons_by_default_on_openai() {
        // gpt-oss-style: reasons by default, so enabling sends nothing.
        assert_eq!(
            plan_reasoning(
                Protocol::OpenAI,
                &effort_settings(&["low", "medium", "high"])
            ),
            None
        );
    }

    #[test]
    fn plan_empty_options_sends_nothing_on_openai() {
        // kimi-k2-thinking: reasoning:true but no knob -> never send `thinking`.
        let settings = ReasoningSettings {
            enabled: true,
            ..Default::default()
        };
        assert_eq!(plan_reasoning(Protocol::OpenAI, &settings), None);
    }

    #[test]
    fn plan_effort_none_disables_when_supported() {
        let settings = ReasoningSettings {
            enabled: false,
            ..effort_settings(&["none"])
        };
        assert_eq!(
            plan_reasoning(Protocol::Cerebras, &settings),
            Some(ReasoningConfig::Effort("none".to_string()))
        );
    }

    #[test]
    fn plan_user_effort_override_wins() {
        let settings = ReasoningSettings {
            effort: Some("high".to_string()),
            ..effort_settings(&["low", "medium", "high"])
        };
        assert_eq!(
            plan_reasoning(Protocol::Cerebras, &settings),
            Some(ReasoningConfig::Effort("high".to_string()))
        );
    }

    #[test]
    fn plan_budget_dialect_is_provider_aware() {
        let settings = ReasoningSettings {
            enabled: true,
            options: vec![ReasoningOption::BudgetTokens {
                min: Some(1024),
                max: None,
            }],
            ..Default::default()
        };
        // Anthropic takes a concrete budget; Gemini enables dynamic thinking.
        assert_eq!(
            plan_reasoning(Protocol::Anthropic, &settings),
            Some(ReasoningConfig::Budget(1024))
        );
        assert_eq!(
            plan_reasoning(Protocol::Gemini, &settings),
            Some(ReasoningConfig::Toggle(true))
        );
    }

    #[test]
    fn plan_anthropic_effort_dialect_enables_explicitly() {
        // Anthropic reasoning is off by default, so an effort-only model must be
        // explicitly enabled rather than left alone (unlike OpenAI/Cerebras).
        assert_eq!(
            plan_reasoning(
                Protocol::Anthropic,
                &effort_settings(&["low", "medium", "high"])
            ),
            Some(ReasoningConfig::Toggle(true))
        );
    }

    #[test]
    fn anthropic_adaptive_models_recognized() {
        for m in [
            "claude-opus-4-6",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-fable-5",
            "claude-opus-5-0",
        ] {
            assert!(
                anthropic_uses_adaptive_thinking(m),
                "{m} should be adaptive"
            );
        }
    }

    #[test]
    fn anthropic_budget_models_recognized() {
        for m in [
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5",
            "claude-opus-4-0",
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5",
        ] {
            assert!(!anthropic_uses_adaptive_thinking(m), "{m} should be budget");
        }
    }

    #[test]
    fn anthropic_reasoning_translates_toggle_for_budget_models() {
        // "reasoning on" -> an explicit budget on a budget-dialect model, since
        // sweet-llm would otherwise send adaptive thinking (which it rejects).
        assert_eq!(
            anthropic_reasoning(Some(ReasoningConfig::Toggle(true)), "claude-sonnet-4-5"),
            Some(ReasoningConfig::Budget(DEFAULT_THINKING_BUDGET))
        );
        // Adaptive models keep `Toggle(true)` (-> `{type: adaptive}`).
        assert_eq!(
            anthropic_reasoning(Some(ReasoningConfig::Toggle(true)), "claude-opus-4-8"),
            Some(ReasoningConfig::Toggle(true))
        );
        // Disable and explicit budget pass through unchanged.
        assert_eq!(
            anthropic_reasoning(Some(ReasoningConfig::Toggle(false)), "claude-sonnet-4-5"),
            Some(ReasoningConfig::Toggle(false))
        );
    }

    #[test]
    fn build_embedder_supports_openai_and_gemini() {
        for protocol in [Protocol::OpenAI, Protocol::Gemini] {
            build_embedder(
                protocol,
                "embed-model",
                "https://example.test/v1",
                "test-key",
            )
            .expect("embedder builds");
        }
    }

    #[test]
    fn build_embedder_rejects_anthropic() {
        // `Arc<dyn Embedder>` isn't `Debug`, so `expect_err` won't compile.
        let err = match build_embedder(Protocol::Anthropic, "x", "", "test-key") {
            Ok(_) => panic!("anthropic has no embeddings API"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("anthropic"),
            "unexpected error: {err}"
        );
    }
}
