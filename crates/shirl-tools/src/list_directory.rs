// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct ListDirectoryArgs {
    /// Path to the directory to list.
    pub path: String,
}

pub fn list_directory_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "list_directory",
        "List files and directories in a directory",
        serde_json::to_value(schemars::schema_for!(ListDirectoryArgs)).expect("schema"),
        ListDirectoryHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct ListDirectoryHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for ListDirectoryHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: ListDirectoryArgs = serde_json::from_value(args)?;
        let entries = self
            .fs
            .list_dir(Path::new(&args.path))
            .await
            .map_err(|e| ToolError::Execution(format!("{}: {e}", args.path).into()))?;

        let mut entries: Vec<(String, bool)> = entries
            .into_iter()
            .map(|e| (e.name, e.metadata.is_dir))
            .collect();

        entries.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase()))
        });

        let lines: Vec<String> = entries
            .iter()
            .map(|(name, is_dir)| {
                if *is_dir {
                    format!("[DIR]  {name}/")
                } else {
                    format!("[FILE] {name}")
                }
            })
            .collect();

        if lines.is_empty() {
            Ok("(empty directory)".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    }
}
