// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Extension that discovers skills from disk and contributes them to the agent
//! via progressive disclosure.
//!
//! A skill is a directory containing a `SKILL.md` (frontmatter `name`,
//! `description`, optional `alwaysApply`) plus optional `scripts/`,
//! `references/`, and `assets/` resource directories.
//!
//! - **Discoverable** (default): only a one-line catalog entry (name +
//!   description + directory + resources) enters the system prompt. The model
//!   reads the full `SKILL.md` on demand via the file tools when a task matches.
//! - **Always-on** (`alwaysApply: true`): the full body is injected into the
//!   system prompt every turn, like AGENTS.md.
//!
//! Discovery (project shadows user by skill name):
//! - **User-level**: `~/.shirl/skills/*/SKILL.md`
//! - **Project-level**: the nearest `.agents/skills/*/SKILL.md` walking from the
//!   start directory up to the git root.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sweet_agent::{Capability, CapabilityProvider, PromptSpec};

const SKILL_FILE: &str = "SKILL.md";
const RESOURCE_DIRS: [&str; 3] = ["scripts", "references", "assets"];

/// A discovered skill. Internal to this module — consumers see only the
/// capabilities `SkillsProvider` produces, not the skills themselves.
struct Skill {
    name: String,
    description: String,
    skill_dir: PathBuf,
    body: String,
    always_apply: bool,
    /// Resource paths relative to `skill_dir`, e.g. `"scripts/deploy.sh"`.
    resources: Vec<String>,
}

/// A [`CapabilityProvider`] that exposes discovered skills: a catalog prompt for
/// discoverable skills plus a full-body prompt for each `alwaysApply` skill.
pub struct SkillsProvider {
    skills: Vec<Skill>,
}

impl SkillsProvider {
    /// Discover skills from the user-level and project-level skill directories,
    /// relative to the process CWD.
    pub fn load() -> Self {
        let start = std::env::current_dir().ok();
        let user_dir = dirs::home_dir().map(|h| h.join(".shirl").join("skills"));
        Self::load_from(user_dir.as_deref(), start.as_deref())
    }

    /// Same as [`load`](Self::load) but with explicit directories, so tests can
    /// avoid touching the home directory or mutating the process CWD.
    fn load_from(user_dir: Option<&Path>, project_start: Option<&Path>) -> Self {
        let mut by_name: BTreeMap<String, Skill> = BTreeMap::new();
        if let Some(dir) = user_dir {
            for skill in read_skills_dir(dir) {
                by_name.insert(skill.name.clone(), skill);
            }
        }
        if let Some(start) = project_start {
            if let Some(dir) = nearest_skills_dir(start) {
                for skill in read_skills_dir(&dir) {
                    by_name.insert(skill.name.clone(), skill);
                }
            }
        }
        Self {
            skills: by_name.into_values().collect(),
        }
    }
}

impl CapabilityProvider for SkillsProvider {
    fn id(&self) -> &str {
        "shirl:skills"
    }

    fn capabilities(&self) -> Vec<Capability> {
        let mut caps = Vec::new();

        // Always-on skills: inject the full body every turn.
        for skill in self.skills.iter().filter(|s| s.always_apply) {
            caps.push(Capability::Prompt(PromptSpec::new(
                format!("skill_{}", skill.name),
                skill.body.clone(),
            )));
        }

        // Discoverable skills: a single catalog the model reads on demand.
        let discoverable: Vec<&Skill> = self.skills.iter().filter(|s| !s.always_apply).collect();
        if !discoverable.is_empty() {
            caps.push(Capability::Prompt(PromptSpec::new(
                "skills_catalog",
                format_catalog(&discoverable),
            )));
        }

        caps
    }
}

/// The nearest `.agents/skills` directory walking from `start` to the git root.
fn nearest_skills_dir(start: &Path) -> Option<PathBuf> {
    crate::discovery::project_dirs(start)
        .into_iter()
        .map(|d| d.join(".agents").join("skills"))
        .find(|p| p.is_dir())
}

/// Parse every immediate subdirectory of `dir` that contains a `SKILL.md`.
fn read_skills_dir(dir: &Path) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join(SKILL_FILE).is_file() {
            if let Some(skill) = parse_skill(&path) {
                out.push(skill);
            }
        }
    }
    out
}

/// Parse a single skill directory. Returns `None` (with a warning) when the
/// skill cannot yield a usable description.
fn parse_skill(skill_dir: &Path) -> Option<Skill> {
    let dir_name = skill_dir.file_name()?.to_str()?.to_string();
    let content = std::fs::read_to_string(skill_dir.join(SKILL_FILE)).ok()?;
    if content.trim().is_empty() {
        return None;
    }

    let (front_matter, body) = split_front_matter(&content);
    let mut name = None;
    let mut description = None;
    let mut always_apply = false;
    if let Some(fm) = front_matter {
        let lines: Vec<&str> = fm.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let Some((key, value)) = lines[i].split_once(':') else {
                i += 1;
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            // A YAML block scalar (`>`, `>-`, `|`, `|-`, …) carries its value
            // on the following indented lines, not after the colon. Fold those
            // continuation lines into the value (joined with spaces — adequate
            // for the single-line metadata fields skills use).
            let resolved = if value.starts_with('>') || value.starts_with('|') {
                let (folded, consumed) = fold_block_scalar(&lines[i + 1..]);
                i += 1 + consumed;
                folded
            } else {
                i += 1;
                value.to_string()
            };
            match key {
                "name" if !resolved.is_empty() => name = Some(resolved),
                "description" if !resolved.is_empty() => description = Some(resolved),
                "alwaysApply" => always_apply = resolved.eq_ignore_ascii_case("true"),
                _ => {}
            }
        }
    }

    let name = name.unwrap_or_else(|| dir_name.clone());
    if name != dir_name {
        eprintln!("shirl: skill '{name}' does not match its directory name '{dir_name}'");
    }

    let description = description.or_else(|| first_description_line(&body));
    let Some(description) = description else {
        eprintln!("shirl: skipping skill '{name}' — no description in frontmatter or body");
        return None;
    };

    Some(Skill {
        name,
        description,
        skill_dir: skill_dir.to_path_buf(),
        body,
        always_apply,
        resources: enumerate_resources(skill_dir),
    })
}

/// Split leading `---`-delimited YAML-ish frontmatter from the body. Returns
/// `(None, whole_content)` when there is no well-formed frontmatter block.
fn split_front_matter(content: &str) -> (Option<String>, String) {
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, content.to_string());
    }
    let mut front_matter = String::new();
    let mut body = Vec::new();
    let mut closed = false;
    for line in lines {
        if !closed {
            if line.trim() == "---" {
                closed = true;
                continue;
            }
            front_matter.push_str(line);
            front_matter.push('\n');
        } else {
            body.push(line);
        }
    }
    if !closed {
        return (None, content.to_string());
    }
    (Some(front_matter), body.join("\n"))
}

/// Fold the indented continuation lines of a YAML block scalar into a single
/// space-joined string. `lines` is the slice immediately following the
/// `key: >`/`key: |` indicator line. Consumes leading indented (or blank)
/// lines and stops at the first line that is non-empty and not indented,
/// returning the folded value and the number of lines consumed.
fn fold_block_scalar(lines: &[&str]) -> (String, usize) {
    let mut parts: Vec<&str> = Vec::new();
    let mut consumed = 0;
    for line in lines {
        let indented = line.starts_with(char::is_whitespace);
        if line.trim().is_empty() {
            consumed += 1;
            continue;
        }
        if !indented {
            break;
        }
        parts.push(line.trim());
        consumed += 1;
    }
    (parts.join(" "), consumed)
}

/// First non-empty, non-heading line of `body`, used as a description fallback.
fn first_description_line(body: &str) -> Option<String> {
    body.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
}

/// Collect resource files one level deep in the skill's `scripts/`,
/// `references/`, and `assets/` directories, as paths relative to `skill_dir`.
fn enumerate_resources(skill_dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for sub in RESOURCE_DIRS {
        let Ok(entries) = std::fs::read_dir(skill_dir.join(sub)) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.path().is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    out.push(format!("{sub}/{name}"));
                }
            }
        }
    }
    out.sort();
    out
}

fn format_catalog(skills: &[&Skill]) -> String {
    let mut out = String::from(
        "Available skills — when a task matches a skill's description, read its SKILL.md to load \
full instructions; resolve resource paths against the skill's directory:\n",
    );
    for skill in skills {
        out.push_str(&format!(
            "\n- name: {}\n  description: {}\n  skill_dir: {}\n",
            skill.name,
            skill.description,
            skill.skill_dir.display()
        ));
        if !skill.resources.is_empty() {
            out.push_str(&format!("  resources: {}\n", skill.resources.join(", ")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create `skills_dir/<name>/SKILL.md` with `content`.
    fn write_skill(skills_dir: &Path, name: &str, content: &str) -> PathBuf {
        let dir = skills_dir.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("SKILL.md"), content).unwrap();
        dir
    }

    #[test]
    fn parses_frontmatter_and_body() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(
            tmp.path(),
            "deploy",
            "---\nname: deploy\ndescription: Ship the app\nalwaysApply: false\n---\nStep 1. Do it.",
        );
        let skill = parse_skill(&dir).unwrap();
        assert_eq!(skill.name, "deploy");
        assert_eq!(skill.description, "Ship the app");
        assert!(!skill.always_apply);
        assert_eq!(skill.body, "Step 1. Do it.");
    }

    #[test]
    fn folded_block_scalar_description_is_joined() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(
            tmp.path(),
            "deploy",
            "---\nname: deploy\ndescription: >-\n  Ship the app to prod\n  when the user asks.\nalwaysApply: false\n---\nbody",
        );
        let skill = parse_skill(&dir).unwrap();
        assert_eq!(
            skill.description,
            "Ship the app to prod when the user asks."
        );
        assert!(!skill.always_apply);
    }

    #[test]
    fn literal_block_scalar_description_is_joined() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(
            tmp.path(),
            "deploy",
            "---\nname: deploy\ndescription: |\n  line one\n  line two\n---\nbody",
        );
        let skill = parse_skill(&dir).unwrap();
        assert_eq!(skill.description, "line one line two");
    }

    #[test]
    fn no_frontmatter_uses_dir_name_and_first_body_line() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(tmp.path(), "notes", "# Heading\n\nUse the notes carefully.");
        let skill = parse_skill(&dir).unwrap();
        assert_eq!(skill.name, "notes");
        assert_eq!(skill.description, "Use the notes carefully.");
        assert!(!skill.always_apply);
    }

    #[test]
    fn skill_without_description_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(tmp.path(), "empty", "---\nname: empty\n---\n");
        assert!(parse_skill(&dir).is_none());
    }

    #[test]
    fn always_apply_skill_emits_body_capability_and_no_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "house-rules",
            "---\nname: house-rules\ndescription: repo rules\nalwaysApply: true\n---\nAlways run fmt.",
        );
        let provider = SkillsProvider::load_from(Some(tmp.path()), None);
        let caps = provider.capabilities();
        assert_eq!(caps.len(), 1);
        match &caps[0] {
            Capability::Prompt(p) => {
                assert_eq!(p.id, "skill_house-rules");
                assert_eq!(p.text, "Always run fmt.");
            }
            _ => panic!("expected Prompt"),
        }
    }

    #[test]
    fn discoverable_skill_emits_catalog_with_dir_and_resources() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(
            tmp.path(),
            "js-frontend",
            "---\nname: js-frontend\ndescription: JS conventions\n---\nbody",
        );
        fs::create_dir_all(dir.join("scripts")).unwrap();
        fs::write(dir.join("scripts").join("build.sh"), "#!/bin/sh").unwrap();

        let provider = SkillsProvider::load_from(Some(tmp.path()), None);
        let caps = provider.capabilities();
        assert_eq!(caps.len(), 1);
        let text = match &caps[0] {
            Capability::Prompt(p) => {
                assert_eq!(p.id, "skills_catalog");
                &p.text
            }
            _ => panic!("expected Prompt"),
        };
        assert!(text.contains("name: js-frontend"));
        assert!(text.contains("description: JS conventions"));
        assert!(text.contains(&dir.display().to_string()));
        assert!(text.contains("scripts/build.sh"));
    }

    #[test]
    fn resource_enumeration_ignores_other_subdirs() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_skill(tmp.path(), "s", "---\nname: s\ndescription: d\n---\nb");
        for sub in ["scripts", "references", "assets", "other"] {
            fs::create_dir_all(dir.join(sub)).unwrap();
            fs::write(dir.join(sub).join("f.txt"), "x").unwrap();
        }
        let resources = enumerate_resources(&dir);
        assert_eq!(
            resources,
            vec!["assets/f.txt", "references/f.txt", "scripts/f.txt"]
        );
    }

    #[test]
    fn directory_without_skill_md_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("not-a-skill")).unwrap();
        let provider = SkillsProvider::load_from(Some(tmp.path()), None);
        assert!(provider.capabilities().is_empty());
    }

    #[test]
    fn project_level_shadows_user_level() {
        let user = tempfile::tempdir().unwrap();
        write_skill(
            user.path(),
            "deploy",
            "---\nname: deploy\ndescription: user deploy\n---\nuser body",
        );
        let project = tempfile::tempdir().unwrap();
        fs::create_dir_all(project.path().join(".git")).unwrap();
        let skills_dir = project.path().join(".agents").join("skills");
        write_skill(
            &skills_dir,
            "deploy",
            "---\nname: deploy\ndescription: project deploy\nalwaysApply: true\n---\nproject body",
        );

        let provider = SkillsProvider::load_from(Some(user.path()), Some(project.path()));
        assert_eq!(provider.skills.len(), 1);
        assert_eq!(provider.skills[0].description, "project deploy");
        assert!(provider.skills[0].always_apply);
    }

    #[test]
    fn empty_provider_has_no_capabilities() {
        let provider = SkillsProvider::load_from(None, None);
        assert!(provider.capabilities().is_empty());
    }
}
