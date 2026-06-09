// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

fn default_n() -> usize {
    10
}

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct TailFileArgs {
    /// Path to the file.
    pub path: String,
    /// Number of lines to read from the end.
    #[serde(default = "default_n")]
    pub n: usize,
}

pub fn tail_file_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "tail_file",
        "Read the last N lines of a text file",
        serde_json::to_value(schemars::schema_for!(TailFileArgs)).expect("schema"),
        TailFileHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct TailFileHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for TailFileHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: TailFileArgs = serde_json::from_value(args)?;
        let content = self
            .fs
            .read_to_string(Path::new(&args.path))
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        let lines: Vec<&str> = content.lines().rev().take(args.n).collect::<Vec<_>>();
        Ok(lines.into_iter().rev().collect::<Vec<_>>().join("\n"))
    }
}
