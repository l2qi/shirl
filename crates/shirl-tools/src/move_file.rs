// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MoveFileArgs {
    /// Source path.
    pub source: String,
    /// Destination path. Must not already exist.
    pub destination: String,
}

pub fn move_file_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "move_file",
        "Move or rename a file or directory",
        serde_json::to_value(schemars::schema_for!(MoveFileArgs)).expect("schema"),
        MoveFileHandler { fs },
    )
    .with_risk(ToolRisk::FileWrite)
}

struct MoveFileHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for MoveFileHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: MoveFileArgs = serde_json::from_value(args)?;
        let src = Path::new(&args.source);
        let dst = Path::new(&args.destination);

        if !self.fs.exists(src).await {
            return Err(ToolError::Execution(
                format!("source does not exist: {}", args.source).into(),
            ));
        }
        if self.fs.exists(dst).await {
            return Err(ToolError::Execution(
                format!("destination already exists: {}", args.destination).into(),
            ));
        }

        if let Some(parent) = dst.parent() {
            self.fs
                .create_dir_all(parent)
                .await
                .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        }

        self.fs
            .rename(src, dst)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        Ok(format!("Moved {} to {}", args.source, args.destination))
    }
}
