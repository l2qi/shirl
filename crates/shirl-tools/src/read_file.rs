// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

fn default_offset() -> usize {
    0
}

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct ReadFileArgs {
    /// Path to the file.
    pub path: String,
    /// Line offset to start from (0-based). Defaults to 0 (start of file).
    #[serde(default = "default_offset")]
    pub offset: usize,
    /// Maximum number of lines to return. If omitted, returns the entire file.
    #[serde(default)]
    pub limit: Option<usize>,
}

pub fn read_file_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "read_file",
        "Read the contents of a UTF-8 file, optionally a range of lines",
        serde_json::to_value(schemars::schema_for!(ReadFileArgs)).expect("schema"),
        ReadFileHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct ReadFileHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for ReadFileHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: ReadFileArgs = serde_json::from_value(args)?;
        let content = self
            .fs
            .read_to_string(Path::new(&args.path))
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        if args.offset == 0 && args.limit.is_none() {
            return Ok(content);
        }

        let lines: Vec<&str> = content
            .lines()
            .skip(args.offset)
            .take(args.limit.unwrap_or(usize::MAX))
            .collect();
        Ok(lines.join("\n"))
    }
}
