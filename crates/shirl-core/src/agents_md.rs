// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Extension that loads project-level and user-level `AGENTS.md` instruction
//! files and contributes them as [`PromptSpec`] capabilities.

use std::path::{Path, PathBuf};

use sweet_agent::{Capability, CapabilityProvider, PromptSpec};

/// File name to search for (compared case-insensitively).
const FILE_NAME: &str = "AGENTS.md";

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Load AGENTS.md instruction files from two locations:
///
/// 1. **User-level**: `~/.shirl/AGENTS.md` (optional)
/// 2. **Project-level**: walk from `start_dir` upward to the git repo root,
///    looking for `AGENTS.md` (case-insensitive) at each directory (optional)
///
/// Both are optional. If neither exists the extension contributes no
/// capabilities. When both exist the user-level prompt is ordered *before* the
/// project-level prompt so that project instructions have the strongest
/// influence (LLM recency bias).
pub fn load() -> AgentsMd {
    let start_dir = std::env::current_dir().ok();
    load_from(start_dir.as_deref())
}

/// Same as [`load`] but with an explicit start directory instead of the
/// process CWD. Useful for testing without mutating global state.
pub fn load_from(start_dir: Option<&Path>) -> AgentsMd {
    let user = load_user_level();
    let project = load_project_level(start_dir);
    AgentsMd { user, project }
}

// ---------------------------------------------------------------------------
// Extension struct
// ---------------------------------------------------------------------------

/// A [`CapabilityProvider`] that injects AGENTS.md content as system-prompt
/// instructions.
pub struct AgentsMd {
    user: Option<(PathBuf, String)>,
    project: Option<(PathBuf, String)>,
}

impl CapabilityProvider for AgentsMd {
    fn id(&self) -> &str {
        "shirl:agents_md"
    }

    fn capabilities(&self) -> Vec<Capability> {
        let mut caps = Vec::new();

        if let Some((path, content)) = &self.user {
            caps.push(Capability::Prompt(PromptSpec::new(
                "agents_md_user",
                format_prompt(path, content),
            )));
        }

        if let Some((path, content)) = &self.project {
            caps.push(Capability::Prompt(PromptSpec::new(
                "agents_md_project",
                format_prompt(path, content),
            )));
        }

        caps
    }
}

fn format_prompt(path: &Path, content: &str) -> String {
    format!("Instructions from: {}\n{}", path.display(), content)
}

// ---------------------------------------------------------------------------
// User-level loading (~/.shirl/AGENTS.md)
// ---------------------------------------------------------------------------

fn load_user_level() -> Option<(PathBuf, String)> {
    let candidate = crate::paths::config_home().ok()?.join(FILE_NAME);
    read_if_exists(&candidate)
}

// ---------------------------------------------------------------------------
// Project-level loading (walk from start_dir to git root)
// ---------------------------------------------------------------------------

fn load_project_level(start_dir: Option<&Path>) -> Option<(PathBuf, String)> {
    let start = start_dir?;
    // Search from `start` (most specific) outward to the git root.
    crate::discovery::project_dirs(start)
        .iter()
        .find_map(|dir| find_agents_md_in(dir))
}

/// Look for `AGENTS.md` inside `dir`. An exact-case match wins; otherwise the
/// first case-insensitive variant (e.g. `agents.md`) is used.
fn find_agents_md_in(dir: &Path) -> Option<(PathBuf, String)> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut case_variant: Option<(PathBuf, String)> = None;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if !name.eq_ignore_ascii_case(FILE_NAME) {
            continue;
        }
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(hit) = read_if_exists(&path) else {
            continue;
        };
        if name == FILE_NAME {
            return Some(hit);
        }
        case_variant.get_or_insert(hit);
    }
    case_variant
}

/// Read a file as UTF-8 text if it exists and is a regular file.
fn read_if_exists(path: &Path) -> Option<(PathBuf, String)> {
    if !path.is_file() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    if content.trim().is_empty() {
        return None;
    }
    Some((path.to_path_buf(), content))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: write AGENTS.md to `dir`.
    fn write_agents_md(dir: &Path, content: &str) {
        fs::write(dir.join("AGENTS.md"), content).unwrap();
    }

    /// Helper: write `agents.md` (lowercase) to `dir`.
    fn write_agents_md_lowercase(dir: &Path, content: &str) {
        fs::write(dir.join("agents.md"), content).unwrap();
    }

    fn init_git(dir: &Path) {
        fs::create_dir_all(dir.join(".git")).unwrap();
    }

    // -- format_prompt --

    #[test]
    fn format_prompt_includes_path_header_and_content() {
        let path = PathBuf::from("/home/user/.shirl/AGENTS.md");
        let result = format_prompt(&path, "use tabs not spaces");
        assert!(result.starts_with("Instructions from: /home/user/.shirl/AGENTS.md\n"));
        assert!(result.ends_with("use tabs not spaces"));
    }

    // -- capabilities() --

    #[test]
    fn no_agents_md_anywhere_produces_no_capabilities() {
        let ext = AgentsMd {
            user: None,
            project: None,
        };
        assert!(ext.capabilities().is_empty());
    }

    #[test]
    fn user_level_only_produces_one_prompt() {
        let content = "prefer short functions";
        let path = PathBuf::from("/home/user/.shirl/AGENTS.md");
        let ext = AgentsMd {
            user: Some((path.clone(), content.to_string())),
            project: None,
        };

        let caps = ext.capabilities();
        assert_eq!(caps.len(), 1);
        let text = match &caps[0] {
            Capability::Prompt(p) => &p.text,
            _ => panic!("expected Prompt capability"),
        };
        assert!(text.contains("Instructions from: /home/user/.shirl/AGENTS.md"));
        assert!(text.contains(content));
    }

    #[test]
    fn project_level_only_produces_one_prompt() {
        let content = "follow existing patterns";
        let path = PathBuf::from("/project/AGENTS.md");
        let ext = AgentsMd {
            user: None,
            project: Some((path.clone(), content.to_string())),
        };

        let caps = ext.capabilities();
        assert_eq!(caps.len(), 1);
        let text = match &caps[0] {
            Capability::Prompt(p) => &p.text,
            _ => panic!("expected Prompt capability"),
        };
        assert!(text.contains("Instructions from: /project/AGENTS.md"));
        assert!(text.contains(content));
    }

    #[test]
    fn both_levels_produce_two_prompts_user_first() {
        let user_path = PathBuf::from("/home/user/.shirl/AGENTS.md");
        let project_path = PathBuf::from("/project/AGENTS.md");
        let ext = AgentsMd {
            user: Some((user_path, "user instructions".to_string())),
            project: Some((project_path, "project instructions".to_string())),
        };

        let caps = ext.capabilities();
        assert_eq!(caps.len(), 2);

        // First should be user-level
        match &caps[0] {
            Capability::Prompt(p) => {
                assert_eq!(p.id, "agents_md_user");
                assert!(p.text.contains("user instructions"));
            }
            _ => panic!("expected Prompt"),
        }

        // Second should be project-level
        match &caps[1] {
            Capability::Prompt(p) => {
                assert_eq!(p.id, "agents_md_project");
                assert!(p.text.contains("project instructions"));
            }
            _ => panic!("expected Prompt"),
        }
    }

    // -- find_agents_md_in --

    #[test]
    fn find_agents_md_finds_exact_case() {
        let tmp = tempfile::tempdir().unwrap();
        write_agents_md(tmp.path(), "hello");
        let result = find_agents_md_in(tmp.path());
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, "hello");
    }

    #[test]
    fn find_agents_md_is_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        write_agents_md_lowercase(tmp.path(), "lowercase");
        let result = find_agents_md_in(tmp.path());
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, "lowercase");
    }

    #[test]
    fn find_agents_md_returns_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let result = find_agents_md_in(tmp.path());
        assert!(result.is_none());
    }

    // -- read_if_exists --

    #[test]
    fn read_if_exists_returns_none_for_missing_file() {
        assert!(read_if_exists(Path::new("/no/such/file")).is_none());
    }

    #[test]
    fn read_if_exists_returns_none_for_empty_content() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("AGENTS.md"), "   \n\t\n  ").unwrap();
        assert!(read_if_exists(&tmp.path().join("AGENTS.md")).is_none());
    }

    #[test]
    fn read_if_exists_returns_content_for_real_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("AGENTS.md"), "use rustfmt").unwrap();
        let result = read_if_exists(&tmp.path().join("AGENTS.md"));
        assert_eq!(result.unwrap().1, "use rustfmt");
    }

    // -- load_project_level via load_from (no CWD mutation) --

    #[test]
    fn load_from_finds_agents_md_in_start_dir() {
        let tmp = tempfile::tempdir().unwrap();
        init_git(tmp.path());
        write_agents_md(tmp.path(), "project rules");

        let ext = load_from(Some(tmp.path()));
        assert!(ext.project.is_some());
        assert_eq!(ext.project.unwrap().1, "project rules");
    }

    #[test]
    fn load_from_finds_agents_md_in_parent_when_start_dir_is_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        init_git(repo_root);
        write_agents_md(repo_root, "root-level rules");

        let sub = repo_root.join("crates").join("shirl-core");
        fs::create_dir_all(&sub).unwrap();

        let ext = load_from(Some(&sub));
        assert!(ext.project.is_some());
        assert_eq!(ext.project.unwrap().1, "root-level rules");
    }

    #[test]
    fn load_from_prefers_start_dir_agents_md_over_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        init_git(repo_root);
        write_agents_md(repo_root, "root-level rules");

        let sub = repo_root.join("subdir");
        fs::create_dir_all(&sub).unwrap();
        write_agents_md(&sub, "subdir rules");

        let ext = load_from(Some(&sub));
        assert!(ext.project.is_some());
        assert_eq!(ext.project.unwrap().1, "subdir rules");
    }

    #[test]
    fn load_from_returns_none_when_no_agents_md() {
        let tmp = tempfile::tempdir().unwrap();
        init_git(tmp.path());

        let ext = load_from(Some(tmp.path()));
        assert!(ext.project.is_none());
    }
}
