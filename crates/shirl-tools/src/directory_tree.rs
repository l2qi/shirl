// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct DirectoryTreeArgs {
    /// Root directory to scan.
    pub path: String,
    /// Maximum recursion depth. Unbounded if omitted.
    #[serde(default)]
    pub max_depth: Option<usize>,
}

pub fn directory_tree_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "directory_tree",
        "Return a recursive tree of files and directories",
        serde_json::to_value(schemars::schema_for!(DirectoryTreeArgs)).expect("schema"),
        DirectoryTreeHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct DirectoryTreeHandler {
    fs: Arc<dyn Filesystem>,
}

#[derive(Serialize)]
struct TreeNode {
    name: String,
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<TreeNode>,
}

#[async_trait]
impl ToolHandler for DirectoryTreeHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: DirectoryTreeArgs = serde_json::from_value(args)?;
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

        let root_name = root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());

        // walk_entries uses ignore::WalkBuilder under DirectFs,
        // so .gitignore is respected.
        let entries = self
            .fs
            .walk_entries(root)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        let root_depth = root.components().count();
        let mut flat: Vec<(usize, String, bool)> = Vec::new();

        for entry in &entries {
            let depth = entry.path.components().count().saturating_sub(root_depth);
            if let Some(md) = args.max_depth {
                if depth > md {
                    continue;
                }
            }
            let name = entry
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            flat.push((depth, name, entry.metadata.is_dir));
        }

        let (children, _) = build_children(&flat, 0, 1);
        let tree = TreeNode {
            name: root_name,
            kind: "directory",
            children,
        };

        Ok(serde_json::to_string_pretty(&tree)
            .map_err(|e| ToolError::Execution(format!("json error: {e}").into()))?)
    }
}

/// Build tree nodes from a flat list of (depth, name, is_dir) entries.
fn build_children(
    flat: &[(usize, String, bool)],
    mut pos: usize,
    parent_depth: usize,
) -> (Vec<TreeNode>, usize) {
    let mut children = Vec::new();
    while pos < flat.len() {
        let (depth, name, is_dir) = &flat[pos];
        if *depth < parent_depth {
            break;
        }
        let node = if *is_dir {
            let (grandchildren, next) = build_children(flat, pos + 1, parent_depth + 1);
            pos = next;
            TreeNode {
                name: name.clone(),
                kind: "directory",
                children: grandchildren,
            }
        } else {
            pos += 1;
            TreeNode {
                name: name.clone(),
                kind: "file",
                children: Vec::new(),
            }
        };
        children.push(node);
    }
    (children, pos)
}
