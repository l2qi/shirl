// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sweet_core::permission::ToolRisk;
use sweet_core::sandbox::Filesystem;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct EditFileArgs {
    /// Path to the file.
    pub path: String,
    /// The exact string to search for. Ignored when `edits` is provided.
    #[serde(default)]
    pub old_string: String,
    /// The replacement string. Ignored when `edits` is provided.
    #[serde(default)]
    pub new_string: String,
    /// Multiple edit operations to apply sequentially. Each has `old_text` and `new_text`,
    /// and each `old_text` must match exactly once. If provided, `old_string`/`new_string`
    /// (and `replace_all`) are ignored.
    #[serde(default)]
    pub edits: Vec<EditOperation>,
    /// If true, preview changes as a diff without writing to disk.
    #[serde(default)]
    pub dry_run: bool,
    /// If true, replace all occurrences of `old_string` instead of requiring exactly one match.
    /// Only applies to single-edit mode (`old_string`/`new_string`).
    #[serde(default)]
    pub replace_all: bool,
}

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
pub struct EditOperation {
    /// Exact text to search for.
    pub old_text: String,
    /// Text to replace it with.
    pub new_text: String,
}

pub fn edit_file_tool(fs: Arc<dyn Filesystem>) -> ToolSpec {
    ToolSpec::new(
        "edit_file",
        "Replace occurrences of text in a UTF-8 file. Supports multiple edits in one call, dry-run preview, and replace-all mode.",
        serde_json::to_value(schemars::schema_for!(EditFileArgs)).expect("schema"),
        EditFileHandler { fs },
    )
    .with_risk(ToolRisk::FileWrite)
}

struct EditFileHandler {
    fs: Arc<dyn Filesystem>,
}

#[async_trait]
impl ToolHandler for EditFileHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: EditFileArgs = serde_json::from_value(args)?;
        let path = Path::new(&args.path);
        let content = self
            .fs
            .read_to_string(path)
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        let had_explicit_edits = !args.edits.is_empty();
        let replace_all = args.replace_all && !had_explicit_edits;

        let edits = if had_explicit_edits {
            args.edits
        } else {
            vec![EditOperation {
                old_text: args.old_string,
                new_text: args.new_string,
            }]
        };

        let mut result = content.clone();

        for (i, edit) in edits.iter().enumerate() {
            if edit.old_text.is_empty() {
                return Err(ToolError::Execution(
                    format!(
                        "edit {}: old_text must not be empty — provide the exact text to replace",
                        i + 1
                    )
                    .into(),
                ));
            }
            let count = result.matches(&edit.old_text).count();
            if count == 0 {
                return Err(ToolError::Execution(
                    format!("edit {}: old_text not found in {}", i + 1, args.path).into(),
                ));
            }
            if count > 1 && !replace_all {
                return Err(ToolError::Execution(
                    format!(
                        "edit {}: old_text found {count} times in {} — expected exactly one match. Use replace_all to replace all occurrences.",
                        i + 1,
                        args.path
                    )
                    .into(),
                ));
            }
            if replace_all {
                result = result.replace(&edit.old_text, &edit.new_text);
            } else {
                result = result.replacen(&edit.old_text, &edit.new_text, 1);
            }
        }

        if args.dry_run {
            let diff = unified_diff(&content, &result, &args.path);
            return Ok(format!("Dry run — no changes written.\n{diff}"));
        }

        self.fs
            .write(path, result.as_bytes())
            .await
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;

        let total_old: usize = edits.iter().map(|e| e.old_text.len()).sum();
        let total_new: usize = edits.iter().map(|e| e.new_text.len()).sum();
        Ok(format!(
            "Edited {} ({} edit(s), replaced {} bytes with {} bytes)",
            args.path,
            edits.len(),
            total_old,
            total_new
        ))
    }
}

/// Maximum number of diff output lines before truncation.
const DIFF_LINE_CAP: usize = 200;

/// Maximum distance (in lines) to search forward for a resynchronization
/// point. Caps the O(old × new) search per hunk to O(MAX_SYNC_WINDOW²),
/// preventing stalls when diffing large files with extensive changes.
const MAX_SYNC_WINDOW: usize = 100;

/// Number of unchanged context lines shown before and after each hunk.
const CONTEXT_LINES: usize = 3;

/// Compute a unified diff between `old` and `new`, labelled with `path`.
///
/// Includes up to `CONTEXT_LINES` lines of context around each change
/// hunk (standard `git diff` style); changes whose context windows would
/// overlap are merged into a single hunk. Output is capped at
/// `DIFF_LINE_CAP` lines; if exceeded, a truncation notice is appended.
pub fn unified_diff(old: &str, new: &str, path: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let mut out = Vec::new();
    out.push(format!("--- {path}"));
    out.push(format!("+++ {path}"));

    let mut raw_hunks: Vec<(usize, usize, usize, usize)> = Vec::new();

    let mut old_idx = 0usize;
    let mut new_idx = 0usize;

    while old_idx < old_lines.len() || new_idx < new_lines.len() {
        while old_idx < old_lines.len()
            && new_idx < new_lines.len()
            && old_lines[old_idx] == new_lines[new_idx]
        {
            old_idx += 1;
            new_idx += 1;
        }

        if old_idx >= old_lines.len() && new_idx >= new_lines.len() {
            break;
        }

        let hunk_old_start = old_idx;
        let hunk_new_start = new_idx;

        let mut hunk_old_end = old_lines.len();
        let mut hunk_new_end = new_lines.len();

        'outer: for oe in old_idx..=(old_idx + MAX_SYNC_WINDOW).min(old_lines.len()) {
            for ne in new_idx..=(new_idx + MAX_SYNC_WINDOW).min(new_lines.len()) {
                let remaining_old = old_lines.len() - oe;
                let remaining_new = new_lines.len() - ne;
                if remaining_old == 0 && remaining_new == 0 {
                    hunk_old_end = oe;
                    hunk_new_end = ne;
                    break 'outer;
                }
                if remaining_old > 0 && remaining_new > 0 && old_lines[oe] == new_lines[ne] {
                    hunk_old_end = oe;
                    hunk_new_end = ne;
                    break 'outer;
                }
            }
        }

        raw_hunks.push((hunk_old_start, hunk_old_end, hunk_new_start, hunk_new_end));
        old_idx = hunk_old_end;
        new_idx = hunk_new_end;
    }

    let mut group_start = 0usize;
    while group_start < raw_hunks.len() {
        let mut group_end = group_start;
        while group_end + 1 < raw_hunks.len()
            && raw_hunks[group_end + 1].0 - raw_hunks[group_end].1 <= 2 * CONTEXT_LINES
        {
            group_end += 1;
        }

        emit_hunk_group(
            &mut out,
            &old_lines,
            &new_lines,
            &raw_hunks[group_start..=group_end],
        );
        group_start = group_end + 1;
    }

    if out.len() > DIFF_LINE_CAP {
        let truncated = out.len() - DIFF_LINE_CAP;
        out.truncate(DIFF_LINE_CAP);
        out.push(format!("({truncated} more lines)"));
    }

    out.join("\n")
}

fn emit_hunk_group(
    out: &mut Vec<String>,
    old_lines: &[&str],
    new_lines: &[&str],
    group: &[(usize, usize, usize, usize)],
) {
    let (first_old_start, _, first_new_start, _) = group[0];
    let (_, last_old_end, _, last_new_end) = group[group.len() - 1];

    let ctx_old_start = first_old_start.saturating_sub(CONTEXT_LINES);
    let ctx_old_end = (last_old_end + CONTEXT_LINES).min(old_lines.len());
    let ctx_new_start = first_new_start - (first_old_start - ctx_old_start);
    let ctx_new_end = (last_new_end + CONTEXT_LINES).min(new_lines.len());

    out.push(format!(
        "@@ -{},{} +{},{} @@",
        ctx_old_start + 1,
        ctx_old_end - ctx_old_start,
        ctx_new_start + 1,
        ctx_new_end - ctx_new_start,
    ));

    for line in &old_lines[ctx_old_start..first_old_start] {
        out.push(format!(" {line}"));
    }

    for (i, &(old_start, old_end, new_start, new_end)) in group.iter().enumerate() {
        for line in &old_lines[old_start..old_end] {
            out.push(format!("-{line}"));
        }
        for line in &new_lines[new_start..new_end] {
            out.push(format!("+{line}"));
        }
        let context_end = group.get(i + 1).map_or(ctx_old_end, |next| next.0);
        for line in &old_lines[old_end..context_end] {
            out.push(format!(" {line}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_no_changes() {
        let diff = unified_diff("hello\nworld\n", "hello\nworld\n", "test.txt");
        assert_eq!(diff, "--- test.txt\n+++ test.txt");
    }

    #[test]
    fn diff_single_line_change_with_context() {
        let old = "line0\nline1\nline2\nline3\nline4\nline5\nline6";
        let new = "line0\nline1\nline2\nCHANGED\nline4\nline5\nline6";
        let diff = unified_diff(old, new, "test.rs");
        let lines: Vec<&str> = diff.lines().collect();
        assert_eq!(lines[0], "--- test.rs");
        assert_eq!(lines[1], "+++ test.rs");
        assert!(lines[2].starts_with("@@ -"));
        assert_eq!(lines[3], " line0");
        assert_eq!(lines[4], " line1");
        assert_eq!(lines[5], " line2");
        assert_eq!(lines[6], "-line3");
        assert_eq!(lines[7], "+CHANGED");
        assert_eq!(lines[8], " line4");
        assert_eq!(lines[9], " line5");
        assert_eq!(lines[10], " line6");
    }

    #[test]
    fn diff_addition_at_start() {
        let old = "line1\nline2";
        let new = "NEW\nline1\nline2";
        let diff = unified_diff(old, new, "f.txt");
        let lines: Vec<&str> = diff.lines().collect();
        assert!(lines[2].starts_with("@@ -"));
        assert_eq!(lines[3], "+NEW");
        assert_eq!(lines[4], " line1");
        assert_eq!(lines[5], " line2");
    }

    #[test]
    fn diff_truncates_at_cap() {
        let old_lines: Vec<String> = (0..300).map(|i| format!("old{i}")).collect();
        let new_lines: Vec<String> = (0..300).map(|i| format!("new{i}")).collect();
        let old = old_lines.join("\n");
        let new = new_lines.join("\n");
        let diff = unified_diff(&old, &new, "big.txt");
        let line_count = diff.lines().count();
        assert_eq!(line_count, DIFF_LINE_CAP + 1);
        assert!(diff.ends_with("more lines)"));
    }

    #[test]
    fn diff_distant_changes_stay_separate_hunks() {
        let old: Vec<String> = (0..20).map(|i| format!("line{i}")).collect();
        let mut new = old.clone();
        new[2] = "CHANGED2".to_string();
        new[17] = "CHANGED17".to_string();
        let diff = unified_diff(&old.join("\n"), &new.join("\n"), "two.txt");
        assert_eq!(diff.lines().filter(|l| l.starts_with("@@")).count(), 2);
    }

    #[test]
    fn diff_close_changes_merge_into_one_hunk() {
        let diff = unified_diff("X\nM\nY", "x\nM\ny", "f.txt");
        let lines: Vec<&str> = diff.lines().collect();
        assert_eq!(
            lines,
            vec![
                "--- f.txt",
                "+++ f.txt",
                "@@ -1,3 +1,3 @@",
                "-X",
                "+x",
                " M",
                "-Y",
                "+y",
            ]
        );
    }

    #[test]
    fn diff_merged_hunk_renders_gap_as_context() {
        let old = "a\nB\nc\nd\ne\nf\nG\nh";
        let new = "a\nX\nc\nd\ne\nf\nY\nh";
        let diff = unified_diff(old, new, "m.txt");
        assert_eq!(diff.lines().filter(|l| l.starts_with("@@")).count(), 1);
        assert_eq!(diff.lines().filter(|l| *l == " c").count(), 1);
        assert_eq!(diff.lines().filter(|l| *l == " d").count(), 1);
        assert_eq!(diff.lines().filter(|l| *l == " e").count(), 1);
    }
}
