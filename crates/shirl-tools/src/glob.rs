// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

fn default_limit() -> usize {
    50
}

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GlobArgs {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

pub fn glob_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "glob",
        "Find files matching a glob pattern. Respects .gitignore.",
        serde_json::to_value(schemars::schema_for!(GlobArgs)).expect("schema"),
        GlobHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct GlobHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for GlobHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: GlobArgs = serde_json::from_value(args)?;
        let base = Path::new(args.path.as_deref().unwrap_or("."));
        let results = self
            .fs
            .walk(&args.pattern, base)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        // Collect up to limit+1 to detect truncation.
        // limit == 0 means unlimited.
        let (visible, truncated) = if args.limit == 0 {
            (results, false)
        } else {
            let mut iter = results.into_iter();
            let visible: Vec<_> = iter.by_ref().take(args.limit).collect();
            let truncated = iter.next().is_some();
            (visible, truncated)
        };

        if visible.is_empty() {
            return Ok("No files found.".to_string());
        }

        let mut output: Vec<String> = visible
            .into_iter()
            .map(|p| p.display().to_string())
            .collect();

        if truncated {
            output.push(format!("... ({} more results not shown)", args.limit));
        }

        Ok(output.join("\n"))
    }
}
