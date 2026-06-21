// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Test generation subagent.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use sweet_agent::{SubagentContext, SubagentHandler, SubagentSpec};
use sweet_core::sandbox::Sandbox;
use sweet_core::{ToolError, ToolRisk};

use super::run_leaf;
use shirl_tools::{glob_tool, grep_tool, head_file_tool, read_file_tool, tail_file_tool};

#[derive(Deserialize, JsonSchema)]
struct TestgenInput {
    /// The target to generate tests for: a function name, module path, or file path.
    target: String,
}

const TESTGEN_PROMPT: &str =
    "You are a test generation subagent. Given a function name, module path, \
    or file, analyze the code and generate comprehensive test cases.\n\
    \n\
    Strategy:\n\
    1. Read the target code to understand its interface and behavior\n\
    2. Search for existing tests to understand the project's testing patterns and conventions\n\
    3. Generate test cases covering: happy paths, edge cases, error paths, boundary conditions\n\
    \n\
    Guidelines:\n\
    - Follow the project's existing test conventions (test framework, naming, organization)\n\
    - Include descriptive test names that explain the expected behavior\n\
    - Return ONLY the test code as plain text - do not write any files\n\
    - If the target has existing tests, suggest additions rather than duplicates";

pub fn testgen_spec(sandbox: Arc<dyn Sandbox>) -> SubagentSpec {
    SubagentSpec::new(
        "testgen",
        "Generate test cases for a specific function, module, or file. Analyzes the code \
         and existing test patterns, then returns comprehensive test code as plain text \
         (does not write files).",
        serde_json::to_value(schemars::schema_for!(TestgenInput)).expect("schema for TestgenInput"),
        TestgenHandler { sandbox },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct TestgenHandler {
    sandbox: Arc<dyn Sandbox>,
}

#[sweet_core::async_trait]
impl SubagentHandler for TestgenHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: TestgenInput = serde_json::from_value(args)?;
        let prompt = format!("Generate tests for: {}", input.target);
        let fs = self.sandbox.fs();
        run_leaf("testgen", TESTGEN_PROMPT, prompt, ctx, |a| {
            a.with_tool(read_file_tool(fs.clone()))
                .with_tool(grep_tool(fs.clone()))
                .with_tool(glob_tool(fs.clone()))
                .with_tool(head_file_tool(fs.clone()))
                .with_tool(tail_file_tool(fs.clone()))
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
        let tool: ToolSpec = testgen_spec(Arc::new(DirectSandbox::new())).into();
        assert_eq!(tool.name, "testgen");
        assert!(tool.parameters_schema.to_string().contains("target"));
    }
}
