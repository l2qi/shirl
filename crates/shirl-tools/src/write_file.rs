// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

pub fn write_file_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "write_file",
        "Write content to a UTF-8 file, creating or truncating it",
        serde_json::to_value(schemars::schema_for!(WriteFileArgs)).expect("schema"),
        WriteFileHandler { fs },
    )
    .with_risk(ToolRisk::FileWrite)
}

struct WriteFileHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for WriteFileHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: WriteFileArgs = serde_json::from_value(args)?;
        self.fs
            .write(Path::new(&args.path), args.content.as_bytes())
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        Ok(format!(
            "Wrote {} bytes to {}",
            args.content.len(),
            args.path
        ))
    }
}
