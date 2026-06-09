// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! File-path picker driven by `@` in the input line.
//!
//! On trigger, walks the project directory (respecting `.gitignore`), fuzzy-matches
//! entries against the user's filter, and presents them as an inline picker. The
//! selected path is spliced back into the input buffer.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use shirl_ui::{Command, FileEntry, FilePickerState, SharedIo};

// ---------------------------------------------------------------------------
// File enumeration
// ---------------------------------------------------------------------------

/// A cached snapshot of the project's file tree. Built once per session on
/// first `@` trigger; reused thereafter.
#[derive(Clone, Default)]
pub struct FileListCache {
    entries: Option<Arc<Vec<FileEntry>>>,
    /// CWD at the time the cache was built — invalidated on directory change.
    built_for_cwd: Option<PathBuf>,
}

impl FileListCache {
    /// Return the cached entries, building them first if needed.
    pub fn get(&mut self, cwd: &Path) -> Arc<Vec<FileEntry>> {
        if let Some(ref cached_cwd) = self.built_for_cwd {
            if cached_cwd == cwd {
                if let Some(ref entries) = self.entries {
                    return Arc::clone(entries);
                }
            }
        }
        let entries = Arc::new(walk_files(cwd));
        self.entries = Some(Arc::clone(&entries));
        self.built_for_cwd = Some(cwd.to_path_buf());
        entries
    }
}

/// Maximum tree depth to descend when collecting files. Keeps the picker
/// responsive in monorepos with deep `node_modules`-style trees.
const MAX_WALK_DEPTH: usize = 10;

/// Recursively walk `cwd`, respecting `.gitignore`, and collect relative paths.
fn walk_files(cwd: &Path) -> Vec<FileEntry> {
    let mut entries = Vec::new();
    let walker = ignore::WalkBuilder::new(cwd)
        .hidden(false)
        .max_depth(Some(MAX_WALK_DEPTH))
        .build();
    for result in walker {
        let Ok(entry) = result else { continue };
        let Some(rel) = entry.path().strip_prefix(cwd).ok() else {
            continue;
        };
        let rel_str = rel.to_string_lossy();
        if rel_str.is_empty() {
            // Root entry.
            continue;
        }
        let is_dir = entry.file_type().is_some_and(|ft| ft.is_dir());
        let display = if is_dir {
            format!("{}/", rel_str)
        } else {
            rel_str.to_string()
        };
        entries.push(FileEntry {
            path: display,
            is_dir,
        });
    }
    // Directories first, then alphabetical within each group.
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then_with(|| a.path.cmp(&b.path)));
    entries
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

/// A dead-simple case-insensitive substring matcher. Good enough for v1 —
/// real fuzzy matching (skipping characters) can be added later.
fn matches_filter(entry: &str, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    entry
        .to_ascii_lowercase()
        .contains(&filter.to_ascii_lowercase())
}

/// Filter and rank entries against `filter`. Returns up to `limit` entries,
/// ranked by filename-match quality (limit applied after sorting so the
/// best matches survive even when there are many candidates).
fn filter_entries(all: &[FileEntry], filter: &str, limit: usize) -> Vec<FileEntry> {
    let mut scored: Vec<(u8, &FileEntry)> = all
        .iter()
        .filter(|e| matches_filter(&e.path, filter))
        .map(|e| (filename_match_score(&e.path, filter), e))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.path.cmp(&b.1.path)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, e)| e.clone())
        .collect()
}

/// Score how well `filter` matches the filename portion of `path`.
/// Higher is better. Exact filename match = 3, prefix match = 2,
/// contains = 1, no filename match = 0.
fn filename_match_score(path: &str, filter: &str) -> u8 {
    let fname = path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .to_ascii_lowercase();
    let f = filter.to_ascii_lowercase();
    if fname == f {
        3
    } else if fname.starts_with(&f) {
        2
    } else if fname.contains(&f) {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Inline picker sub-loop
// ---------------------------------------------------------------------------

/// Maximum number of entries to show in the picker.
const PICKER_LIMIT: usize = 50;

/// Drive the inline file picker. Called from the main event loop for any
/// picker-related `Command`. Holds the IO lock once across state mutation,
/// path splicing, and redraw so the picker can't render in a half-updated
/// state.
///
/// Returns `Some(PickerAction)` when the picker closed (accepted or
/// cancelled); the caller uses this to decide whether further action is
/// needed beyond the splicing this function already performed.
pub async fn dispatch(
    io: &SharedIo,
    cache: &mut FileListCache,
    cwd: &Path,
    cmd: &Command,
) -> Result<Option<PickerAction>> {
    let mut guard = io.lock().await;

    let action = match cmd {
        Command::FilePickerFilter(filter) => {
            let all = cache.get(cwd);
            let filtered = filter_entries(&all, filter, PICKER_LIMIT);
            if let Some(ref mut fp) = guard.file_picker {
                fp.filter = filter.clone();
                fp.entries = filtered;
                fp.selected = 0;
                fp.scroll = 0;
            } else {
                guard.file_picker = Some(FilePickerState::new(filter.clone(), filtered));
            }
            None
        }
        Command::SelectMove(delta) => {
            if let Some(ref mut fp) = guard.file_picker {
                fp.move_selection(*delta);
            }
            None
        }
        Command::FilePickerAccept => {
            let path = guard
                .file_picker
                .as_ref()
                .and_then(|fp| fp.selected_entry())
                .map(|e| e.path.clone());
            guard.file_picker = None;
            match path {
                Some(p) => {
                    guard.insert_file_mention(&p);
                    Some(PickerAction::Accepted)
                }
                None => Some(PickerAction::Cancelled),
            }
        }
        Command::FilePickerClose => {
            if guard.file_picker.is_none() {
                return Ok(None);
            }
            guard.file_picker = None;
            Some(PickerAction::Cancelled)
        }
        _ => None,
    };

    let _ = guard.draw();
    Ok(action)
}

/// Result of a picker interaction.
#[derive(Debug)]
pub enum PickerAction {
    /// The user selected an entry; the path is already spliced into the
    /// input buffer.
    Accepted,
    /// The user dismissed the picker.
    Cancelled,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_filter_basic() {
        assert!(matches_filter("src/main.rs", "main"));
        assert!(matches_filter("src/main.rs", ""));
        assert!(!matches_filter("src/main.rs", "zzz"));
    }

    #[test]
    fn matches_filter_case_insensitive() {
        assert!(matches_filter("src/Main.rs", "main"));
        assert!(matches_filter("src/main.rs", "MAIN"));
    }

    #[test]
    fn filename_match_scoring() {
        assert_eq!(filename_match_score("src/main.rs", "main.rs"), 3);
        assert_eq!(filename_match_score("src/main.rs", "main"), 2);
        assert_eq!(filename_match_score("src/main.rs", "ai"), 1);
        assert_eq!(filename_match_score("src/main.rs", "xyz"), 0);
    }

    #[test]
    fn filter_entries_prefers_filename_match() {
        let entries = vec![
            FileEntry {
                path: "contains_foo/bar.rs".to_string(),
                is_dir: false,
            },
            FileEntry {
                path: "foo.rs".to_string(),
                is_dir: false,
            },
            FileEntry {
                path: "src/foo/mod.rs".to_string(),
                is_dir: false,
            },
        ];
        let result = filter_entries(&entries, "foo", 10);
        assert_eq!(result[0].path, "foo.rs");
    }

    /// Regression: a perfect filename match must survive the `limit` cut
    /// even when there are many lower-quality matches in front of it in
    /// the input slice.
    #[test]
    fn filter_entries_keeps_best_match_under_limit() {
        let mut entries: Vec<FileEntry> = (0..20)
            .map(|i| FileEntry {
                path: format!("noise/contains_foo_{i}.txt"),
                is_dir: false,
            })
            .collect();
        // Best match (filename == filter) appears last in the input slice.
        entries.push(FileEntry {
            path: "deep/path/foo".to_string(),
            is_dir: false,
        });
        let result = filter_entries(&entries, "foo", 5);
        assert_eq!(result.len(), 5);
        assert_eq!(
            result[0].path, "deep/path/foo",
            "best filename match should be ranked first regardless of input order"
        );
    }
}
