// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Extension that loads user-defined slash commands from disk and contributes
//! them as `Activation::ByCommand` prompt capabilities.
//!
//! A custom command is a Markdown file whose body is a prompt template. Invoking
//! `/name [args]` renders the template (substituting `$ARGUMENTS`, or appending
//! the args) and submits the result as a user turn — the command is *content*,
//! not behavior, so it carries no handler.
//!
//! Discovery (project shadows user by command name):
//! - **User-level**: `~/.shirl/commands/*.md`
//! - **Project-level**: the nearest `.agents/commands/*.md` walking from the
//!   start directory up to the git root.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sweet_agent::{Capability, CapabilityProvider, PromptSpec};

/// A user-defined slash command: a name and the prompt template it expands to.
struct CustomCommand {
    name: String,
    template: String,
}

/// A [`CapabilityProvider`] that exposes user-defined slash commands as
/// `Activation::ByCommand` prompts.
pub struct CustomCommandsProvider {
    commands: Vec<CustomCommand>,
}

impl CustomCommandsProvider {
    /// Discover custom commands from the user-level and project-level command
    /// directories, relative to the process CWD. Commands whose name appears in
    /// `reserved` (built-in and mode commands) are skipped with a warning.
    pub fn load(reserved: &[&str]) -> Self {
        let start = std::env::current_dir().ok();
        let user_dir = dirs::home_dir().map(|h| h.join(".shirl").join("commands"));
        Self::load_from(user_dir.as_deref(), start.as_deref(), reserved)
    }

    /// Same as [`load`](Self::load) but with explicit directories, so tests can
    /// avoid touching the home directory or mutating the process CWD.
    fn load_from(user_dir: Option<&Path>, project_start: Option<&Path>, reserved: &[&str]) -> Self {
        // Keyed by name; project entries overwrite user entries. BTreeMap keeps
        // a stable, sorted order for `/capabilities` and tests.
        let mut by_name: BTreeMap<String, String> = BTreeMap::new();

        if let Some(dir) = user_dir {
            for (name, template) in read_command_dir(dir) {
                by_name.insert(name, template);
            }
        }
        if let Some(start) = project_start {
            if let Some(dir) = nearest_commands_dir(start) {
                for (name, template) in read_command_dir(&dir) {
                    by_name.insert(name, template);
                }
            }
        }

        let commands = by_name
            .into_iter()
            .filter(|(name, _)| {
                if reserved.contains(&name.as_str()) {
                    eprintln!(
                        "shirl: skipping custom command '/{name}' — name is reserved by a built-in"
                    );
                    false
                } else {
                    true
                }
            })
            .map(|(name, template)| CustomCommand { name, template })
            .collect();

        Self { commands }
    }
}

impl CapabilityProvider for CustomCommandsProvider {
    fn id(&self) -> &str {
        "shirl:custom_commands"
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.commands
            .iter()
            .map(|cmd| Capability::Prompt(PromptSpec::command(&cmd.name, &cmd.template)))
            .collect()
    }
}

/// Render a command template against the supplied argument string. If the
/// template contains `$ARGUMENTS` it is substituted; otherwise non-empty args
/// are appended after a blank line, and an empty arg string leaves the template
/// untouched.
pub fn render_template(template: &str, args: &str) -> String {
    if template.contains("$ARGUMENTS") {
        template.replace("$ARGUMENTS", args)
    } else if args.is_empty() {
        template.to_string()
    } else {
        format!("{template}\n\n{args}")
    }
}

/// The nearest `.agents/commands` directory walking from `start` to the git
/// root, if any.
fn nearest_commands_dir(start: &Path) -> Option<PathBuf> {
    crate::discovery::project_dirs(start)
        .into_iter()
        .map(|d| d.join(".agents").join("commands"))
        .find(|p| p.is_dir())
}

/// Read top-level `*.md` files from `dir` as `(stem, body)` pairs. Empty or
/// whitespace-only files are skipped; non-`.md` entries and subdirectories are
/// ignored (no recursion).
fn read_command_dir(dir: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }
        out.push((stem.to_string(), content));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(dir: &Path, name: &str, content: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(name), content).unwrap();
    }

    fn names(provider: &CustomCommandsProvider) -> Vec<&str> {
        provider.commands.iter().map(|c| c.name.as_str()).collect()
    }

    #[test]
    fn empty_dirs_produce_no_commands() {
        let provider = CustomCommandsProvider::load_from(None, None, &[]);
        assert!(provider.capabilities().is_empty());
    }

    #[test]
    fn loads_user_level_commands() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "summarize.md", "Summarize:\n\n$ARGUMENTS");

        let provider = CustomCommandsProvider::load_from(Some(tmp.path()), None, &[]);
        assert_eq!(names(&provider), vec!["summarize"]);

        let caps = provider.capabilities();
        assert_eq!(caps.len(), 1);
        match &caps[0] {
            Capability::Prompt(p) => {
                assert_eq!(
                    p.activation,
                    sweet_agent::Activation::ByCommand("summarize".into())
                );
                assert_eq!(p.text, "Summarize:\n\n$ARGUMENTS");
            }
            _ => panic!("expected Prompt capability"),
        }
    }

    #[test]
    fn loads_project_level_commands() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join(".git")).unwrap();
        let cmd_dir = tmp.path().join(".agents").join("commands");
        write(&cmd_dir, "deploy.md", "deploy it");

        let provider = CustomCommandsProvider::load_from(None, Some(tmp.path()), &[]);
        assert_eq!(names(&provider), vec!["deploy"]);
    }

    #[test]
    fn project_level_shadows_user_level() {
        let user = tempfile::tempdir().unwrap();
        write(user.path(), "deploy.md", "user version");

        let project = tempfile::tempdir().unwrap();
        fs::create_dir_all(project.path().join(".git")).unwrap();
        let cmd_dir = project.path().join(".agents").join("commands");
        write(&cmd_dir, "deploy.md", "project version");

        let provider =
            CustomCommandsProvider::load_from(Some(user.path()), Some(project.path()), &[]);
        assert_eq!(provider.commands.len(), 1);
        assert_eq!(provider.commands[0].template, "project version");
    }

    #[test]
    fn non_md_files_and_empty_files_are_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "keep.md", "real");
        write(tmp.path(), "notes.txt", "ignored");
        write(tmp.path(), "blank.md", "   \n\t  ");

        let provider = CustomCommandsProvider::load_from(Some(tmp.path()), None, &[]);
        assert_eq!(names(&provider), vec!["keep"]);
    }

    #[test]
    fn reserved_names_are_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "compact.md", "shadow built-in");
        write(tmp.path(), "summarize.md", "fine");

        let provider = CustomCommandsProvider::load_from(Some(tmp.path()), None, &["compact"]);
        assert_eq!(names(&provider), vec!["summarize"]);
    }

    #[test]
    fn render_template_substitutes_arguments() {
        assert_eq!(
            render_template("Summarize:\n\n$ARGUMENTS", "the auth module"),
            "Summarize:\n\nthe auth module"
        );
    }

    #[test]
    fn render_template_appends_when_no_placeholder() {
        assert_eq!(
            render_template("Review this", "src/lib.rs"),
            "Review this\n\nsrc/lib.rs"
        );
    }

    #[test]
    fn render_template_verbatim_when_no_args() {
        assert_eq!(render_template("Just do it", ""), "Just do it");
    }
}
