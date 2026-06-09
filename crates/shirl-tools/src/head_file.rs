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
pub struct HeadFileArgs {
    /// Path to the file.
    pub path: String,
    /// Number of lines to read from the start.
    #[serde(default = "default_n")]
    pub n: usize,
}

pub fn head_file_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "head_file",
        "Read the first N lines of a text file",
        serde_json::to_value(schemars::schema_for!(HeadFileArgs)).expect("schema"),
        HeadFileHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct HeadFileHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for HeadFileHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: HeadFileArgs = serde_json::from_value(args)?;
        let content = self
            .fs
            .read_to_string(Path::new(&args.path))
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        let lines: Vec<&str> = content.lines().take(args.n).collect();
        Ok(lines.join("\n"))
    }
}
