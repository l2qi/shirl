// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Filesystem-walk helpers shared by the disk-discovered providers
//! (`agents_md`, `custom_commands`, `skills`).

use std::path::{Path, PathBuf};

/// Walk upward from `start` until we find a `.git` directory or file. Returns
/// the directory that contains `.git`, or `None` if we hit the filesystem root
/// without finding one.
pub(crate) fn find_git_root(start: &Path) -> Option<&Path> {
    let mut dir = start;
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return None,
        }
    }
}

/// The directories to search for project-level configuration: from `start` up
/// to (and including) the git root, nearest first. When `start` is not inside a
/// git repo, the walk stops at `start` itself.
pub(crate) fn project_dirs(start: &Path) -> Vec<PathBuf> {
    let root = find_git_root(start).unwrap_or(start).to_path_buf();
    let mut dirs = Vec::new();
    let mut current = start;
    loop {
        dirs.push(current.to_path_buf());
        if current == root {
            break;
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn init_git(dir: &Path) {
        fs::create_dir_all(dir.join(".git")).unwrap();
    }

    #[test]
    fn find_git_root_returns_dir_containing_git() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        init_git(repo_root);
        let sub = repo_root.join("crates").join("shirl-core").join("src");
        fs::create_dir_all(&sub).unwrap();

        assert_eq!(find_git_root(&sub), Some(repo_root));
    }

    #[test]
    fn find_git_root_returns_none_when_no_git_ancestor() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("a").join("b");
        fs::create_dir_all(&sub).unwrap();

        assert_eq!(find_git_root(&sub), None);
    }

    #[test]
    fn project_dirs_walks_from_start_up_to_git_root() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_root = tmp.path();
        init_git(repo_root);
        let sub = repo_root.join("crates").join("shirl-core");
        fs::create_dir_all(&sub).unwrap();

        let dirs = project_dirs(&sub);
        assert_eq!(dirs.first().unwrap(), &sub); // nearest first
        assert_eq!(dirs.last().unwrap(), repo_root); // root included
    }

    #[test]
    fn project_dirs_without_git_yields_only_start() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("a").join("b");
        fs::create_dir_all(&sub).unwrap();

        assert_eq!(project_dirs(&sub), vec![sub]);
    }
}
