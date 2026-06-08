// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct CreateDirectoryArgs {
    /// Path of the directory to create.
    pub path: String,
}

pub fn create_directory_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "create_directory",
        "Create a directory and all parent directories (mkdir -p)",
        serde_json::to_value(schemars::schema_for!(CreateDirectoryArgs)).expect("schema"),
        CreateDirectoryHandler { fs },
    )
    .with_risk(ToolRisk::FileWrite)
}

struct CreateDirectoryHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for CreateDirectoryHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: CreateDirectoryArgs = serde_json::from_value(args)?;
        self.fs
            .create_dir_all(Path::new(&args.path))
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        Ok(format!("Created directory {}", args.path))
    }
}
