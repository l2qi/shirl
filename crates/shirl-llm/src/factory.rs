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

use crate::catalog::Protocol;

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
/// When `reasoning` is true the factory auto-enables thinking on the provider
/// so that `reasoning_content` is correctly handled in multi-turn tool-use.
pub fn build_model(
    protocol: Protocol,
    model_id: &str,
    base_url: &str,
    api_key: &str,
    context_window: Option<usize>,
    reasoning: bool,
) -> Result<Arc<dyn Model>> {
    let model: Arc<dyn Model> = match protocol {
        Protocol::OpenAI => {
            let mut p = sweet_llm::OpenAIProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            if reasoning {
                p = p.with_thinking(sweet_llm::openai::ThinkingMode::ENABLED);
            }
            Arc::new(p)
        }
        Protocol::Anthropic => {
            let mut p = sweet_llm::AnthropicProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            if reasoning {
                p = p.with_thinking(sweet_llm::anthropic::ThinkingConfig::adaptive());
            }
            Arc::new(p)
        }
        Protocol::Gemini => {
            let mut p = sweet_llm::GeminiProvider::new(api_key).with_model(model_id);
            if !base_url.is_empty() {
                p = p.with_base_url(base_url);
            }
            Arc::new(p)
        }
    };
    Ok(apply_context_window(model, context_window))
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

    #[test]
    fn build_model_constructs_each_protocol() {
        for protocol in [Protocol::OpenAI, Protocol::Anthropic, Protocol::Gemini] {
            let model = build_model(
                protocol,
                "test-model",
                "https://example.test/v1",
                "test-key",
                Some(123_000),
                false,
            )
            .expect("provider builds");
            assert_eq!(model.context_window(), Some(123_000));
        }
    }
}
