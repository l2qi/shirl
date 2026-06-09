// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sweet_core::sandbox::{DirectSandbox, Sandbox};
use tempfile::{NamedTempFile, TempDir};

use shirl_tools::{
    bash_tool, create_directory_tool, directory_size_tool, directory_tree_tool, edit_file_tool,
    get_file_info_tool, glob_tool, grep_tool, head_file_tool, list_directory_tool, move_file_tool,
    patch_tool, read_file_tool, tail_file_tool, write_file_tool,
};

/// Helper: build all tools from a DirectSandbox and call the given tool.
struct Tools {
    fs: Arc<dyn sweet_core::sandbox::Filesystem>,
    runner: Arc<dyn sweet_core::sandbox::CommandRunner>,
}

impl Tools {
    fn new() -> Self {
        let sandbox = DirectSandbox::new();
        Self {
            fs: sandbox.fs(),
            runner: sandbox.runner(),
        }
    }
}

#[tokio::test]
async fn write_file_creates_and_overwrites() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.txt").to_str().unwrap().to_string();

    let tool = write_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path, "content": "first"}))
        .await
        .unwrap();
    assert!(result.contains("5 bytes"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");

    let result2 = tool
        .call(serde_json::json!({"path": path, "content": "second"}))
        .await
        .unwrap();
    assert!(result2.contains("6 bytes"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
}

#[tokio::test]
async fn bash_captures_stdout_and_exit() {
    let tools = Tools::new();
    let tool = bash_tool(tools.runner.clone());
    let result = tool
        .call(serde_json::json!({"command": "echo hello"}))
        .await
        .unwrap();
    // Success with stdout: stdout first, no "Exit code" header.
    assert!(result.starts_with("hello"));
    assert!(!result.contains("stderr"));
}

#[tokio::test]
async fn bash_failure_shows_exit_code_then_stderr() {
    let tools = Tools::new();
    let tool = bash_tool(tools.runner.clone());
    let result = tool
        .call(serde_json::json!({"command": "echo oops >&2 && exit 1"}))
        .await
        .unwrap();
    // Failure: exit code first, then stderr, then stdout.
    assert!(result.starts_with("Exit code: 1"));
    assert!(result.contains("oops"));
    assert!(result.contains("stderr"));
}

#[tokio::test]
async fn edit_file_replaces_once() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("edit.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "foo bar baz").unwrap();

    let tool = edit_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path, "old_string": "bar", "new_string": "qux"}))
        .await
        .unwrap();
    assert!(result.contains("Edited"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "foo qux baz");
}

#[tokio::test]
async fn edit_file_rejects_multiple_matches() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("multi.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "aaa bbb aaa").unwrap();

    let tool = edit_file_tool(tools.fs.clone());
    let err = tool
        .call(serde_json::json!({"path": path, "old_string": "aaa", "new_string": "ccc"}))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("2 times"));
    assert!(std::fs::read_to_string(&path).unwrap() == "aaa bbb aaa");
}

#[tokio::test]
async fn edit_file_replace_all() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("all.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "aaa bbb aaa").unwrap();

    let tool = edit_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "path": path,
            "old_string": "aaa",
            "new_string": "ccc",
            "replace_all": true
        }))
        .await
        .unwrap();
    assert!(result.contains("Edited"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "ccc bbb ccc");
}

#[tokio::test]
async fn edit_file_dry_run_does_not_write() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("dry.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "foo bar").unwrap();

    let tool = edit_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "path": path,
            "old_string": "bar",
            "new_string": "baz",
            "dry_run": true
        }))
        .await
        .unwrap();
    assert!(result.contains("Dry run"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "foo bar");
}

#[tokio::test]
async fn edit_file_multi_edit() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("multi.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "alpha beta gamma").unwrap();

    let tool = edit_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "path": path,
            "edits": [
                {"old_text": "alpha", "new_text": "ONE"},
                {"old_text": "gamma", "new_text": "THREE"}
            ]
        }))
        .await
        .unwrap();
    assert!(result.contains("2 edit(s)"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "ONE beta THREE");
}

#[tokio::test]
async fn grep_finds_literal_in_file() {
    let tools = Tools::new();
    let mut tmp = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut tmp, b"line one\nline two\nline three").unwrap();
    let path = tmp.path().to_str().unwrap().to_string();

    let tool = grep_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"pattern": "two", "path": path}))
        .await
        .unwrap();
    assert!(result.contains("line two"));
    assert!(!result.contains("line one"));
}

#[tokio::test]
async fn glob_respects_gitignore() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let base = dir.path();

    std::fs::write(base.join("keep.rs"), "").unwrap();
    std::fs::write(base.join("skip.rs"), "").unwrap();
    std::fs::write(base.join(".gitignore"), "skip.rs\n").unwrap();
    std::fs::create_dir(base.join(".git")).unwrap();

    let tool = glob_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"pattern": "*.rs", "path": base.to_str().unwrap()}))
        .await
        .unwrap();
    assert!(result.contains("keep.rs"));
    assert!(!result.contains("skip.rs"));
}

#[tokio::test]
async fn glob_matches_relative_path_pattern() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let base = dir.path();

    std::fs::create_dir_all(base.join("scripts")).unwrap();
    std::fs::write(base.join("scripts").join("install.sh"), "#!/bin/bash\n").unwrap();
    std::fs::write(base.join("scripts").join("build.sh"), "#!/bin/bash\n").unwrap();
    std::fs::create_dir(base.join(".git")).unwrap();

    let tool = glob_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "pattern": "scripts/install.sh",
            "path": base.to_str().unwrap()
        }))
        .await
        .unwrap();
    assert!(
        result.contains("install.sh"),
        "should match scripts/install.sh"
    );
    assert!(
        !result.contains("build.sh"),
        "should not match scripts/build.sh"
    );
}

#[tokio::test]
async fn patch_applies_diff() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("target.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "line one\nline two\nline three\n").unwrap();

    let patch_text = r#"--- target.txt
+++ target.txt
@@ -1,3 +1,3 @@
 line one
-line two
+line two modified
 line three
"#;

    let tool = patch_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path, "patch": patch_text}))
        .await
        .unwrap();
    assert!(result.contains("Patched"));
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("line two modified"));
}

#[tokio::test]
async fn patch_tolerates_unprefixed_blank_context_line() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("target.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "line one\n\nline three\n").unwrap();

    // The blank context line is emitted with no leading space, as
    // whitespace-trimming editors and many models produce.
    let patch_text = "--- target.txt\n+++ target.txt\n@@ -1,3 +1,3 @@\n line one\n\n-line three\n+line three modified\n";

    let tool = patch_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path, "patch": patch_text}))
        .await
        .unwrap();
    assert!(result.contains("Patched"));
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "line one\n\nline three modified\n");
}

#[tokio::test]
async fn patch_preserves_crlf_line_endings() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("target.txt").to_str().unwrap().to_string();
    std::fs::write(&path, "line one\r\nline two\r\nline three\r\n").unwrap();

    let patch_text = "--- target.txt\n+++ target.txt\n@@ -1,3 +1,3 @@\n line one\n-line two\n+line two modified\n line three\n";

    let tool = patch_tool(tools.fs.clone());
    tool.call(serde_json::json!({"path": path, "patch": patch_text}))
        .await
        .unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "line one\r\nline two modified\r\nline three\r\n");
}

#[tokio::test]
async fn move_file_renames_within_dir() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let src = dir.path().join("old.txt");
    let dst = dir.path().join("new.txt");
    std::fs::write(&src, "data").unwrap();

    let tool = move_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "source": src.to_str().unwrap(),
            "destination": dst.to_str().unwrap()
        }))
        .await
        .unwrap();
    assert!(result.contains("Moved"));
    assert!(!src.exists());
    assert_eq!(std::fs::read_to_string(&dst).unwrap(), "data");
}

#[tokio::test]
async fn move_file_rejects_existing_destination() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let src = dir.path().join("a.txt");
    let dst = dir.path().join("b.txt");
    std::fs::write(&src, "a").unwrap();
    std::fs::write(&dst, "b").unwrap();

    let tool = move_file_tool(tools.fs.clone());
    let err = tool
        .call(serde_json::json!({
            "source": src.to_str().unwrap(),
            "destination": dst.to_str().unwrap()
        }))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("already exists"));
}

#[tokio::test]
async fn create_directory_makes_parents() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let target = dir.path().join("a/b/c");
    let path_str = target.to_str().unwrap().to_string();

    let tool = create_directory_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path_str}))
        .await
        .unwrap();
    assert!(result.contains("Created directory"));
    assert!(target.is_dir());
}

#[tokio::test]
async fn directory_tree_returns_json() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("foo.rs"), "").unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub").join("bar.rs"), "").unwrap();

    let tool = directory_tree_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": dir.path().to_str().unwrap()}))
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["type"], "directory");
    let children = parsed["children"].as_array().unwrap();
    assert!(children.iter().any(|c| c["name"] == "foo.rs"));
    assert!(children
        .iter()
        .any(|c| c["name"] == "sub" && c["type"] == "directory"));
}

#[tokio::test]
async fn directory_tree_respects_max_depth() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
    std::fs::write(dir.path().join("a/b").join("deep.rs"), "").unwrap();

    let tool = directory_tree_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "path": dir.path().to_str().unwrap(),
            "max_depth": 1
        }))
        .await
        .unwrap();
    assert!(!result.contains("deep.rs"));
}

#[tokio::test]
async fn directory_tree_respects_gitignore() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let base = dir.path();

    // Set up a git repo so .gitignore is honored
    std::fs::create_dir_all(base.join(".git")).unwrap();
    std::fs::write(base.join(".gitignore"), "skip.rs\n").unwrap();
    std::fs::write(base.join("keep.rs"), "").unwrap();
    std::fs::write(base.join("skip.rs"), "").unwrap();

    let tool = directory_tree_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({
            "path": base.to_str().unwrap()
        }))
        .await
        .unwrap();

    assert!(result.contains("keep.rs"), "keep.rs should appear in tree");
    assert!(
        !result.contains("skip.rs"),
        "skip.rs (gitignored) should NOT appear in tree"
    );
}

#[tokio::test]
async fn get_file_info_returns_metadata() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("info.txt");
    std::fs::write(&file_path, "hello").unwrap();

    let tool = get_file_info_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": file_path.to_str().unwrap()}))
        .await
        .unwrap();
    assert!(result.contains("size: 5"));
    assert!(result.contains("isFile: true"));
    assert!(result.contains("isDirectory: false"));
    assert!(result.contains("permissions:"));
}

#[tokio::test]
async fn directory_size_calculates_recursive() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.txt"), "aaa").unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub").join("b.txt"), "bb").unwrap();

    let tool = directory_size_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": dir.path().to_str().unwrap()}))
        .await
        .unwrap();
    assert!(result.contains("5 bytes"));
}

#[tokio::test]
async fn read_file_returns_whole_file() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("whole.txt");
    std::fs::write(&path, "line0\nline1\nline2\n").unwrap();
    let path_str = path.to_str().unwrap().to_string();

    let tool = read_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path_str}))
        .await
        .unwrap();
    assert_eq!(result, "line0\nline1\nline2\n");
}

#[tokio::test]
async fn read_file_with_offset_and_limit() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("lines.txt");
    std::fs::write(&path, "line0\nline1\nline2\nline3\nline4").unwrap();
    let path_str = path.to_str().unwrap().to_string();

    let tool = read_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path_str, "offset": 1, "limit": 2}))
        .await
        .unwrap();
    assert_eq!(result, "line1\nline2");
}

#[tokio::test]
async fn read_file_offset_only() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("lines.txt");
    std::fs::write(&path, "line0\nline1\nline2").unwrap();
    let path_str = path.to_str().unwrap().to_string();

    let tool = read_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path_str, "offset": 2}))
        .await
        .unwrap();
    assert_eq!(result, "line2");
}

#[tokio::test]
async fn head_file_reads_first_n_lines() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("head.txt");
    std::fs::write(&path, "a\nb\nc\nd\ne").unwrap();
    let path_str = path.to_str().unwrap().to_string();

    let tool = head_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path_str, "n": 3}))
        .await
        .unwrap();
    assert_eq!(result, "a\nb\nc");
}

#[tokio::test]
async fn tail_file_reads_last_n_lines() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tail.txt");
    std::fs::write(&path, "a\nb\nc\nd\ne").unwrap();
    let path_str = path.to_str().unwrap().to_string();

    let tool = tail_file_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": path_str, "n": 2}))
        .await
        .unwrap();
    assert_eq!(result, "d\ne");
}

#[tokio::test]
async fn list_directory_shows_entries() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("file.txt"), "").unwrap();
    std::fs::create_dir(dir.path().join("subdir")).unwrap();

    let tool = list_directory_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": dir.path().to_str().unwrap()}))
        .await
        .unwrap();
    assert!(result.contains("[DIR]  subdir/"));
    assert!(result.contains("[FILE] file.txt"));
}

#[tokio::test]
async fn list_directory_shows_dirs_first() {
    let tools = Tools::new();
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("aaa.txt"), "").unwrap();
    std::fs::create_dir(dir.path().join("zzz")).unwrap();

    let tool = list_directory_tool(tools.fs.clone());
    let result = tool
        .call(serde_json::json!({"path": dir.path().to_str().unwrap()}))
        .await
        .unwrap();
    let dir_pos = result.find("[DIR]").unwrap();
    let file_pos = result.find("[FILE]").unwrap();
    assert!(dir_pos < file_pos);
}
