// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Code explanation subagent.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use sweet_agent::{SubagentContext, SubagentHandler, SubagentSpec};
use sweet_core::sandbox::Sandbox;
use sweet_core::{ToolError, ToolRisk};

use super::run_leaf;
use shirl_tools::{
    directory_tree_tool, glob_tool, grep_tool, head_file_tool, read_file_tool, tail_file_tool,
};

#[derive(Deserialize, JsonSchema)]
struct ExplainInput {
    /// The file path to explain, optionally with :line or :line-line range.
    target: String,
}

const EXPLAIN_PROMPT: &str = "You are a code explanation subagent. Given a file path (optionally \
    with :line or :line-line range), explain what the code does in clear, plain language.\n\
    \n\
    Guidelines:\n\
    - Start by reading the specified file or region\n\
    - If no line range is given, read the file and focus on the main entry points\n\
    - Explain the purpose, control flow, and key data structures\n\
    - Reference specific functions, types, and their roles\n\
    - If the code references external modules or types, look them up to provide context\n\
    - Assume the reader is a competent developer unfamiliar with this specific code";

pub fn explain_spec(sandbox: Arc<dyn Sandbox>) -> SubagentSpec {
    SubagentSpec::new(
        "explain",
        "Explain what a specific code file or region does in plain language. \
         Provide the file path, optionally with :line or :line-line range. \
         Returns a clear explanation covering purpose, control flow, and key structures.",
        serde_json::to_value(schemars::schema_for!(ExplainInput)).expect("schema for ExplainInput"),
        ExplainHandler { sandbox },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct ExplainHandler {
    sandbox: Arc<dyn Sandbox>,
}

#[sweet_core::async_trait]
impl SubagentHandler for ExplainHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: ExplainInput = serde_json::from_value(args)?;
        let prompt = format!("Explain the following code:\n\n{}", input.target);
        let fs = self.sandbox.fs();
        run_leaf("explain", EXPLAIN_PROMPT, prompt, ctx, |a| {
            a.with_tool(read_file_tool(fs.clone()))
                .with_tool(head_file_tool(fs.clone()))
                .with_tool(tail_file_tool(fs.clone()))
                .with_tool(grep_tool(fs.clone()))
                .with_tool(glob_tool(fs.clone()))
                .with_tool(directory_tree_tool(fs.clone()))
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::sandbox::DirectSandbox;
    use sweet_core::tool::ToolSpec;

    #[test]
    fn spec_round_trips_through_toolspec() {
        let tool: ToolSpec = explain_spec(Arc::new(DirectSandbox::new())).into();
        assert_eq!(tool.name, "explain");
        assert!(tool.parameters_schema.to_string().contains("target"));
    }
}
