// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Debugging subagent.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use sweet_agent::{SubagentContext, SubagentHandler, SubagentSpec};
use sweet_core::sandbox::Sandbox;
use sweet_core::{ToolError, ToolRisk};

use super::run_leaf;
use shirl_tools::{
    directory_tree_tool, get_file_info_tool, glob_tool, grep_tool, head_file_tool,
    list_directory_tool, read_file_tool, tail_file_tool,
};

#[derive(Deserialize, JsonSchema)]
struct DiagnoseInput {
    /// The error message, test failure output, or description of the problem.
    error: String,
    /// Optional additional context (e.g. what the user was doing, recent changes).
    #[serde(default)]
    context: Option<String>,
}

const DIAGNOSE_PROMPT: &str =
    "You are a debugging subagent. Given an error message and optional context, \
    systematically investigate the codebase to diagnose the root cause.\n\
    \n\
    Strategy:\n\
    1. Search for the error message in the codebase using Grep\n\
    2. Read the relevant source files to understand the error path\n\
    3. Check related test files for expected behavior\n\
    4. Look for similar patterns elsewhere that might explain the issue\n\
    \n\
    Guidelines:\n\
    - Start by locating where the error originates\n\
    - Trace the call chain by searching for function definitions\n\
    - Report your diagnosis clearly: root cause, affected files, and suggested fix direction\n\
    - If the cause is ambiguous, list the most likely candidates ranked by probability\n\
    - Do not modify any files — you are read-only";

pub fn diagnose_spec(sandbox: Arc<dyn Sandbox>) -> SubagentSpec {
    SubagentSpec::new(
        "diagnose",
        "Debug an error or failing test by systematically investigating the codebase. \
         Provide the error message and optional context. Returns a diagnosis with root cause, \
         affected files, and suggested fix direction.",
        serde_json::to_value(schemars::schema_for!(DiagnoseInput))
            .expect("schema for DiagnoseInput"),
        DiagnoseHandler { sandbox },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct DiagnoseHandler {
    sandbox: Arc<dyn Sandbox>,
}

#[sweet_core::async_trait]
impl SubagentHandler for DiagnoseHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: DiagnoseInput = serde_json::from_value(args)?;
        let mut prompt = format!("Diagnose this error:\n\n{}", input.error);
        if let Some(extra) = &input.context {
            prompt.push_str(&format!("\n\nAdditional context: {}", extra));
        }
        let fs = self.sandbox.fs();
        run_leaf("diagnose", DIAGNOSE_PROMPT, prompt, ctx, |a| {
            a.with_tool(glob_tool(fs.clone()))
                .with_tool(grep_tool(fs.clone()))
                .with_tool(read_file_tool(fs.clone()))
                .with_tool(head_file_tool(fs.clone()))
                .with_tool(tail_file_tool(fs.clone()))
                .with_tool(directory_tree_tool(fs.clone()))
                .with_tool(list_directory_tool(fs.clone()))
                .with_tool(get_file_info_tool(fs.clone()))
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
        let tool: ToolSpec = diagnose_spec(Arc::new(DirectSandbox::new())).into();
        assert_eq!(tool.name, "diagnose");
        let schema = tool.parameters_schema.to_string();
        assert!(schema.contains("error"));
        assert!(schema.contains("context"));
    }
}
