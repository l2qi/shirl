// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! On-disk-backed input history ring buffer.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(super) struct History {
    entries: Vec<String>,
    path: PathBuf,
    cap: usize,
}

impl History {
    /// Load history from `path` (returning an empty buffer if the file is
    /// missing). New entries are persisted to the same path.
    pub(super) fn load(path: PathBuf, cap: usize) -> Self {
        let entries = read_entries(&path);
        Self { entries, path, cap }
    }

    pub(super) fn entries(&self) -> &[String] {
        &self.entries
    }

    /// Append `line` unless it equals the most recent entry. Trims to
    /// capacity and persists on every change.
    pub(super) fn push(&mut self, line: String) {
        if self.entries.last().map(String::as_str) == Some(line.as_str()) {
            return;
        }
        self.entries.push(line);
        if self.entries.len() > self.cap {
            let drop = self.entries.len() - self.cap;
            self.entries.drain(..drop);
        }
        self.persist();
    }

    fn persist(&self) {
        let Ok(mut file) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
        else {
            return;
        };
        // Restrict to owner read/write only — history may contain sensitive
        // data if redaction is ever incomplete.
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
        for line in &self.entries {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Default location: `~/.shirl/history.txt`, or `./.shirl_history` if no home
/// directory is available.
pub(super) fn default_history_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        let dir = home.join(".shirl");
        let _ = fs::create_dir_all(&dir);
        dir.join("history.txt")
    } else {
        PathBuf::from(".shirl_history")
    }
}

fn read_entries(path: &Path) -> Vec<String> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.txt");
        (dir, path)
    }

    #[test]
    fn missing_file_loads_empty() {
        let (_dir, path) = tmp_path();
        let h = History::load(path, 100);
        assert!(h.entries().is_empty());
    }

    #[test]
    fn push_dedups_consecutive_duplicates() {
        let (_dir, path) = tmp_path();
        let mut h = History::load(path, 100);
        h.push("a".to_string());
        h.push("a".to_string());
        h.push("b".to_string());
        assert_eq!(h.entries(), &["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn push_caps_to_capacity() {
        let (_dir, path) = tmp_path();
        let mut h = History::load(path, 3);
        for n in 0..5 {
            h.push(n.to_string());
        }
        assert_eq!(
            h.entries(),
            &["2".to_string(), "3".to_string(), "4".to_string()]
        );
    }

    #[test]
    fn push_persists_and_load_roundtrips() {
        let (_dir, path) = tmp_path();
        {
            let mut h = History::load(path.clone(), 100);
            h.push("first".to_string());
            h.push("second".to_string());
        }
        let h = History::load(path, 100);
        assert_eq!(h.entries(), &["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn empty_lines_in_file_are_skipped_on_load() {
        let (_dir, path) = tmp_path();
        std::fs::write(&path, "first\n\nsecond\n").unwrap();
        let h = History::load(path, 100);
        assert_eq!(h.entries(), &["first".to_string(), "second".to_string()]);
    }
}
