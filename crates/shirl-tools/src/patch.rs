// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Apply a unified diff patch to a file.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct PatchArgs {
    /// Path to the file to patch.
    pub path: String,
    /// Unified diff patch text.
    pub patch: String,
}

pub fn patch_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "patch",
        "Apply a unified diff patch to a UTF-8 file",
        serde_json::to_value(schemars::schema_for!(PatchArgs)).expect("schema"),
        PatchHandler { fs },
    )
    .with_risk(ToolRisk::FileWrite)
}

struct PatchHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for PatchHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: PatchArgs = serde_json::from_value(args)?;
        let path = Path::new(&args.path);
        let content = self
            .fs
            .read_to_string(path)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        let lines: Vec<&str> = content.lines().collect();

        let mut new_lines = Vec::new();
        let mut patch_lines = args.patch.lines().peekable();
        let mut first_hunk = true;

        while let Some(line) = patch_lines.peek() {
            if line.starts_with("@@") {
                break;
            }
            patch_lines.next();
        }

        let mut source_idx = 0usize;

        while let Some(line) = patch_lines.next() {
            if line.starts_with("@@") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 3 {
                    return Err(ToolError::Execution(
                        format!("malformed hunk header: {line}").into(),
                    ));
                }
                let old_range = parts[1];
                let start_str = old_range
                    .trim_start_matches('-')
                    .split(',')
                    .next()
                    .ok_or_else(|| {
                        ToolError::Execution(format!("malformed hunk header: {line}").into())
                    })?;
                let start: usize = start_str
                    .parse()
                    .map_err(|e| ToolError::Execution(format!("invalid hunk start: {e}").into()))?;

                if first_hunk {
                    for i in 0..(start.saturating_sub(1)) {
                        if i < lines.len() {
                            new_lines.push(lines[i].to_string());
                        }
                    }
                    source_idx = start.saturating_sub(1);
                    first_hunk = false;
                } else {
                    while source_idx < start.saturating_sub(1) && source_idx < lines.len() {
                        new_lines.push(lines[source_idx].to_string());
                        source_idx += 1;
                    }
                }

                while let Some(&hunk_line) = patch_lines.peek() {
                    if hunk_line.starts_with("@@")
                        || hunk_line.starts_with("---")
                        || hunk_line.starts_with("+++")
                    {
                        break;
                    }
                    patch_lines.next();

                    if let Some(ctx) = hunk_line.strip_prefix(' ') {
                        if source_idx < lines.len() && lines[source_idx] == ctx {
                            new_lines.push(ctx.to_string());
                            source_idx += 1;
                        } else {
                            return Err(ToolError::Execution(
                                format!(
                                    "patch context mismatch at line {}: expected {:?}, got {:?}",
                                    source_idx + 1,
                                    lines.get(source_idx),
                                    ctx
                                )
                                .into(),
                            ));
                        }
                    } else if hunk_line.is_empty() {
                        // A blank line with no prefix. Canonical unified diffs
                        // prefix empty context lines with a single space, but
                        // patches (often model-generated) frequently drop it.
                        // Treat a bare blank line as empty context.
                        if source_idx < lines.len() && lines[source_idx].is_empty() {
                            new_lines.push(String::new());
                            source_idx += 1;
                        } else {
                            return Err(ToolError::Execution(
                                format!(
                                    "patch context mismatch at line {}: expected {:?}, got empty line",
                                    source_idx + 1,
                                    lines.get(source_idx),
                                )
                                .into(),
                            ));
                        }
                    } else if let Some(removed) = hunk_line.strip_prefix('-') {
                        if source_idx < lines.len() && lines[source_idx] == removed {
                            source_idx += 1;
                        } else {
                            return Err(ToolError::Execution(
                                format!(
                                    "patch removal mismatch at line {}: expected {:?}, got {:?}",
                                    source_idx + 1,
                                    lines.get(source_idx),
                                    removed
                                )
                                .into(),
                            ));
                        }
                    } else if let Some(added) = hunk_line.strip_prefix('+') {
                        new_lines.push(added.to_string());
                    } else if hunk_line == "\\ No newline at end of file" {
                        // ignore
                    } else {
                        return Err(ToolError::Execution(
                            format!("unexpected patch line: {hunk_line}").into(),
                        ));
                    }
                }
            }
        }

        while source_idx < lines.len() {
            new_lines.push(lines[source_idx].to_string());
            source_idx += 1;
        }

        // Preserve the file's dominant line ending rather than silently
        // rewriting a CRLF file to LF. `lines()` strips `\r` from every line,
        // so both the source and patch context compare cleanly; we only need
        // to restore the ending when joining the result back together.
        let eol = if content.contains("\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let new_content = new_lines.join(eol);
        let new_content = if content.ends_with('\n') && !new_content.ends_with('\n') {
            new_content + eol
        } else {
            new_content
        };

        self.fs
            .write(path, new_content.as_bytes())
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        Ok(format!(
            "Patched {} ({} lines after patch)",
            args.path,
            new_lines.len()
        ))
    }
}
