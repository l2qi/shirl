// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct DirectorySizeArgs {
    /// Path to the directory.
    pub path: String,
}

pub fn directory_size_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "directory_size",
        "Calculate the total recursive size of a directory (respects .gitignore)",
        serde_json::to_value(schemars::schema_for!(DirectorySizeArgs)).expect("schema"),
        DirectorySizeHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct DirectorySizeHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for DirectorySizeHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: DirectorySizeArgs = serde_json::from_value(args)?;
        let root = Path::new(&args.path);

        let meta = self
            .fs
            .metadata(root)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        if !meta.is_dir {
            return Err(ToolError::Execution(
                format!("{} is not a directory", args.path).into(),
            ));
        }

        // Use walk to get all files, then sum sizes
        let files = self
            .fs
            .walk("**", root)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        let mut total: u64 = 0;
        for file_path in files {
            if let Ok(m) = self.fs.metadata(&file_path).await {
                if !m.is_dir {
                    total += m.size;
                }
            }
        }

        Ok(if total < 1024 {
            format_bytes(total)
        } else {
            format!("{}\n{total} bytes", format_bytes(total))
        })
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}
