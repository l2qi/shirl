// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Web research subagent.
//!
//! Fetches HTTP URLs and, when a search backend is configured, also runs web
//! searches. Used by the main, plan, and review agents to look up external
//! documentation, APIs, and best practices.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use sweet_agent::{SubagentContext, SubagentHandler, SubagentSpec};
use sweet_core::{ToolError, ToolRisk};
use sweet_tools::{HttpFetch, WebSearch, WebSearchBackend};

use super::run_leaf;

#[derive(Deserialize, JsonSchema)]
struct WebResearchInput {
    /// The research question, or a question paired with URL(s) to consult.
    /// If a web-search backend is configured, the subagent can search the open
    /// web; otherwise it can only fetch URLs provided in the prompt.
    question: String,
}

const FETCH_ONLY_PROMPT: &str =
    "You are a web-research subagent. You can fetch HTTP URLs to read web pages, \
    documentation, or APIs and return a concise answer.\n\
    \n\
    Guidelines:\n\
    - You can ONLY fetch URLs provided in the prompt — you cannot search the open web\n\
    - If the prompt contains a URL, fetch it directly and extract the relevant information\n\
    - If the prompt only contains a question with no URL, say so and ask the caller to provide one\n\
    - Cite the source URL(s) you fetched in your answer\n\
    - Be concise: summarize key findings rather than copying entire pages";

const SEARCH_AND_FETCH_PROMPT: &str =
    "You are a web-research subagent. You can search the open web and fetch HTTP URLs to \
    read web pages, documentation, or APIs.\n\
    \n\
    Strategy:\n\
    - If the prompt includes a URL, fetch it directly\n\
    - Otherwise call web_search to find relevant pages, then fetch the most promising ones\n\
    - Cite the source URL(s) in your answer\n\
    - Be concise: summarize key findings rather than copying entire pages";

/// Build the web-research subagent spec.
///
/// When `search_backend` is `Some`, the subagent gets both `HttpFetch` and
/// `WebSearch` tools; when `None`, it falls back to fetch-only and its
/// description reflects that constraint so the parent model picks the right
/// affordance.
pub fn web_research_spec(search_backend: Option<Arc<dyn WebSearchBackend>>) -> SubagentSpec {
    let description = if search_backend.is_some() {
        "Search the web and/or fetch URLs to research documentation, APIs, or best \
         practices. Returns a concise summary with source URLs."
    } else {
        "Fetch and analyze one or more web pages. Provide the URL(s) to consult in the \
         question field (no web-search backend is configured, so this subagent cannot \
         search the open web). Returns a concise summary with source URLs."
    };
    SubagentSpec::new(
        "web_research",
        description,
        serde_json::to_value(schemars::schema_for!(WebResearchInput))
            .expect("schema for WebResearchInput"),
        WebResearchHandler {
            search: search_backend,
        },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct WebResearchHandler {
    search: Option<Arc<dyn WebSearchBackend>>,
}

#[sweet_core::async_trait]
impl SubagentHandler for WebResearchHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: WebResearchInput = serde_json::from_value(args)?;
        let prompt = if self.search.is_some() {
            SEARCH_AND_FETCH_PROMPT
        } else {
            FETCH_ONLY_PROMPT
        };
        let search = self.search.clone();
        run_leaf("web_research", prompt, input.question, ctx, |a| {
            let a = a.with_tool(HttpFetch::default());
            match search {
                Some(backend) => a.with_tool(WebSearch::new(backend)),
                None => a,
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::tool::ToolSpec;

    #[test]
    fn fetch_only_spec_round_trips() {
        let tool: ToolSpec = web_research_spec(None).into();
        assert_eq!(tool.name, "web_research");
        assert!(tool.description.contains("cannot search the open web"));
        assert!(tool.parameters_schema.to_string().contains("question"));
    }

    #[test]
    fn with_search_spec_round_trips() {
        struct NoopBackend;
        #[sweet_core::async_trait]
        impl WebSearchBackend for NoopBackend {
            async fn search(
                &self,
                _query: &str,
                _max_results: usize,
            ) -> Result<Vec<sweet_tools::SearchResult>, sweet_tools::WebSearchError> {
                Ok(vec![])
            }
        }
        let tool: ToolSpec = web_research_spec(Some(Arc::new(NoopBackend))).into();
        assert_eq!(tool.name, "web_research");
        assert!(tool.description.contains("Search the web"));
    }
}
