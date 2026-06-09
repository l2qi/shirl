// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Execute a bash command and capture output.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::CommandRunner;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct BashArgs {
    /// Bash command to execute.
    pub command: String,
    /// Optional working directory. Defaults to the current directory.
    #[serde(default)]
    pub cwd: Option<String>,
}

pub fn bash_tool(runner: Arc<dyn CommandRunner>) -> ToolSpec {
    ToolSpec::new(
        "bash",
        "Run a bash command and return stdout, stderr, and exit code",
        serde_json::to_value(schemars::schema_for!(BashArgs)).expect("schema"),
        BashHandler { runner },
    )
    .with_risk(ToolRisk::Dangerous)
}

struct BashHandler {
    runner: Arc<dyn CommandRunner>,
}

#[async_trait]
impl ToolHandler for BashHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: BashArgs = serde_json::from_value(args)?;
        let out = self
            .runner
            .run(&args.command, args.cwd.as_deref().map(Path::new), None)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        let stdout = out.stdout.trim_end();
        let stderr = out.stderr.trim_end();

        let result = if out.exit_code == 0 {
            // Success: stdout first (useful content up top for UI preview
            // and model consumption), metadata last.
            match (stdout.is_empty(), stderr.is_empty()) {
                (true, true) => "Exit code: 0".to_string(),
                (false, true) => stdout.to_string(),
                (true, false) => format!("Exit code: 0\nstderr:\n{stderr}"),
                (false, false) => format!("{stdout}\n\nExit code: 0\nstderr:\n{stderr}"),
            }
        } else {
            // Failure: exit code first, then stderr (the diagnosis), then
            // stdout last.  Omit empty sections.
            let mut parts = vec![format!("Exit code: {}", out.exit_code)];
            if !stderr.is_empty() {
                parts.push(format!("stderr:\n{stderr}"));
            }
            if !stdout.is_empty() {
                parts.push(format!("stdout:\n{stdout}"));
            }
            parts.join("\n\n")
        };
        Ok(result)
    }
}
