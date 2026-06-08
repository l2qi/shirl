// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Search file contents by literal string or regex.

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
pub struct GrepArgs {
    /// Pattern to search for.
    pub pattern: String,
    /// Path to a file or directory to search.
    pub path: String,
    /// Use regex instead of literal search.
    #[serde(default)]
    pub regex: bool,
    /// Maximum number of matching lines to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

pub fn grep_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "grep",
        "Search file contents for a pattern. Respects .gitignore when searching directories.",
        serde_json::to_value(schemars::schema_for!(GrepArgs)).expect("schema"),
        GrepHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct GrepHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for GrepHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: GrepArgs = serde_json::from_value(args)?;
        let base = Path::new(&args.path);

        // If path is a file, search it directly
        let meta = self
            .fs
            .metadata(base)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        let mut results = Vec::new();

        if !meta.is_dir {
            let content = self
                .fs
                .read_to_string(base)
                .await
                .map_err(|e| ToolError::Execution(e.to_string().into()))?;
            let re = build_regex(&args.pattern, args.regex)?;
            for (line_no, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(format!("{}:{}: {}", args.path, line_no + 1, line));
                    if results.len() >= args.limit {
                        results.push(format!("... (limit {} reached)", args.limit));
                        break;
                    }
                }
            }
        } else {
            let matches = self
                .fs
                .search(&args.pattern, base, args.regex, args.limit)
                .await
                .map_err(|e| ToolError::Execution(e.to_string().into()))?;
            let at_limit = matches.len() >= args.limit;
            for m in matches {
                results.push(format!(
                    "{}:{}: {}",
                    m.path.display(),
                    m.line_number,
                    m.line
                ));
            }
            if at_limit {
                results.push(format!("... (limit {} reached)", args.limit));
            }
        }

        if results.is_empty() {
            Ok("No matches found.".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
}

fn build_regex(pattern: &str, is_regex: bool) -> Result<regex::Regex, ToolError> {
    if is_regex {
        regex::Regex::new(pattern)
            .map_err(|e| ToolError::Execution(format!("invalid regex: {e}").into()))
    } else {
        regex::Regex::new(&regex::escape(pattern))
            .map_err(|e| ToolError::Execution(format!("invalid regex: {e}").into()))
    }
}
