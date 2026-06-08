// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GetFileInfoArgs {
    /// Path to inspect.
    pub path: String,
}

pub fn get_file_info_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "get_file_info",
        "Get file or directory metadata: size, created, modified, permissions",
        serde_json::to_value(schemars::schema_for!(GetFileInfoArgs)).expect("schema"),
        GetFileInfoHandler { fs },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct GetFileInfoHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for GetFileInfoHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: GetFileInfoArgs = serde_json::from_value(args)?;
        let path = Path::new(&args.path);
        let meta = self
            .fs
            .metadata(path)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        let size = meta.size;
        let is_file = !meta.is_dir;
        let is_dir = meta.is_dir;

        let modified = meta
            .modified
            .map(|t| {
                let datetime: chrono::DateTime<chrono::Local> = t.into();
                datetime.format("%a %b %d %Y %H:%M:%S %:z").to_string()
            })
            .unwrap_or_default();

        let created = meta
            .created
            .map(|t| {
                let datetime: chrono::DateTime<chrono::Local> = t.into();
                datetime.format("%a %b %d %Y %H:%M:%S %:z").to_string()
            })
            .unwrap_or_default();

        #[cfg(unix)]
        let permissions = meta
            .unix_permissions
            .map(|mode| format!("{:o}", mode & 0o777))
            .unwrap_or_else(|| "unknown".to_string());
        #[cfg(not(unix))]
        let permissions = "unknown".to_string();

        Ok(format!(
            "size: {size}\ncreated: {created}\nmodified: {modified}\nisDirectory: {is_dir}\nisFile: {is_file}\npermissions: {permissions}"
        ))
    }
}
