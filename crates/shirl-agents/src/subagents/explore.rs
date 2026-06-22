// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Codebase exploration subagent.

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

const EXPLORE_PROMPT: &str = "You are an exploration subagent. Given a question about a codebase, \
    efficiently investigate and return a concise textual summary.\n\
    \n\
    You can answer two broad kinds of question:\n\
    - Structural questions (\"what does this module do?\", \"where is X handled?\")\n\
    - Symbol questions (\"where is fn foo defined?\", \"who calls bar?\", \"all uses of Baz\")\n\
    \n\
    Strategy:\n\
    - For symbol lookups, start with Grep on the exact name; for usages, search the workspace\n\
    - For structural questions, start with DirectoryTree or Glob, then Grep into hot spots\n\
    - Use ReadFile to inspect specific regions (prefer HeadFile/TailFile for large files)\n\
    - Use GetFileInfo to check file metadata when relevant\n\
    \n\
    Guidelines:\n\
    - Be efficient: don't load entire files when a targeted search suffices\n\
    - Cite file paths and line numbers when referencing specific code\n\
    - For symbol queries, distinguish definitions from usages and group results by file\n\
    - If you cannot find a definitive answer, say so and report what you did find\n\
    - Keep the summary focused on the question asked - avoid tangential information";

#[derive(Deserialize, JsonSchema)]
struct ExploreInput {
    /// Question or goal to investigate in the codebase.
    goal: String,
}

pub fn explore_spec(sandbox: Arc<dyn Sandbox>) -> SubagentSpec {
    SubagentSpec::new(
        "explore",
        "Explore the codebase to answer a question: find files, search for symbols, read snippets, \
         and return a concise textual summary. Use this when you need to understand existing code \
         before writing or modifying.",
        serde_json::to_value(schemars::schema_for!(ExploreInput))
            .expect("schema for ExploreInput"),
        ExploreHandler { sandbox },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct ExploreHandler {
    sandbox: Arc<dyn Sandbox>,
}

#[sweet_core::async_trait]
impl SubagentHandler for ExploreHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: ExploreInput = serde_json::from_value(args)?;
        let fs = self.sandbox.fs();
        run_leaf("explore", EXPLORE_PROMPT, input.goal, ctx, |a| {
            a.with_tool(read_file_tool(fs.clone()))
                .with_tool(glob_tool(fs.clone()))
                .with_tool(grep_tool(fs.clone()))
                .with_tool(head_file_tool(fs.clone()))
                .with_tool(tail_file_tool(fs.clone()))
                .with_tool(directory_tree_tool(fs.clone()))
                .with_tool(list_directory_tool(fs.clone()))
                .with_tool(get_file_info_tool(fs))
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
        let tool: ToolSpec = explore_spec(Arc::new(DirectSandbox::new())).into();
        assert_eq!(tool.name, "explore");
        assert!(tool.description.contains("Explore the codebase"));
        assert!(tool.parameters_schema.to_string().contains("goal"));
    }
}
