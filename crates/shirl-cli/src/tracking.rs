// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Glue between `shirl-core`'s [`PlanTracker`] and `shirl-agents`. Lives in the
//! binary so neither library crate depends on the other: the CLI builds the
//! tracker, attaches its tool + reminder to interactive Main agents, and bundles
//! it into the headless [`Tracking`] (implementing `shirl-agents`'s
//! [`ReportStore`] over the tracker).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shirl_agents::agents::AgentKind;
use shirl_agents::headless::{ReportStore, Tracking};
use shirl_core::PlanTracker;
use sweet_agent::Agent;
use sweet_core::{Model, Role, Session, SessionId};

/// Load the workflow tracker for a session, or `None` if the home directory
/// can't be resolved (workflow features then degrade off - unreachable in
/// practice, since the session itself lives under the same home directory).
pub(crate) fn load_tracker(session_id: &SessionId) -> Option<PlanTracker> {
    shirl_core::session_dir(session_id)
        .ok()
        .map(PlanTracker::load)
}

/// Sandbox read roots the agent may read but not write, without re-exposing the
/// rest of the home directory:
/// - the workflow tracker's plan/review files under `~/.shirl/sessions`, and
/// - ancestor `.cargo` directories (see [`ancestor_cargo_dirs`]).
pub(crate) fn sandbox_read_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = shirl_core::sessions_root().ok().into_iter().collect();
    if let Ok(cwd) = std::env::current_dir() {
        roots.extend(ancestor_cargo_dirs(&cwd));
    }
    roots
}

/// Cargo discovers configuration by walking from the working directory up
/// through every ancestor, reading each `.cargo/config.toml` it finds. Under
/// the sandbox only the project root is readable, so an ancestor `.cargo` (e.g.
/// a `[patch]` overlay shared across sibling crates in a parent dir) is denied
/// and the build fails with a permission error. Expose those ancestor `.cargo`
/// dirs read-only so cargo's config walk succeeds. The project root's own
/// `.cargo` is already readable, so only strict ancestors need adding.
///
/// Paths are returned as found; `OsSandbox` canonicalizes every read root
/// (resolving symlinks) so the in-process file tools and the sandboxed command
/// runner both see the same resolved directory.
fn ancestor_cargo_dirs(cwd: &Path) -> Vec<PathBuf> {
    cwd.ancestors()
        .skip(1) // strict ancestors; cwd itself is already in the write root
        .map(|dir| dir.join(".cargo"))
        .filter(|cargo| cargo.is_dir())
        .collect()
}

/// Sandbox write roots the agent may write as well as read, without opening up
/// the rest of the home directory. `cargo build` populates its registry cache,
/// git checkouts, and `.package-cache` lock under `$CARGO_HOME` (default
/// `~/.cargo`), which lives outside the project root - without write access
/// there every fetch fails with `Operation not permitted`. The default
/// `~/.cargo` is already a *read* root (a known tool dir the sandbox exposes);
/// a custom `$CARGO_HOME` is made readable by the write root itself (sweet folds
/// every write root into the read set). Either way this adds write access on top
/// of read access. Returned only when the directory exists, and used only under
/// the sandbox (a no-op when the sandbox policy is Off).
pub(crate) fn sandbox_write_roots() -> Vec<PathBuf> {
    existing_dirs(cargo_home())
}

/// Resolve `$CARGO_HOME`, falling back to `~/.cargo` - cargo's own default.
fn cargo_home() -> Option<PathBuf> {
    cargo_home_from(
        std::env::var_os("CARGO_HOME").map(PathBuf::from),
        dirs::home_dir(),
    )
}

/// The env-var-vs-default resolution, split out for testing without touching
/// process-wide environment state. An empty `CARGO_HOME` is treated as unset -
/// cargo itself ignores an empty value and falls back to `~/.cargo`.
fn cargo_home_from(env_cargo_home: Option<PathBuf>, home: Option<PathBuf>) -> Option<PathBuf> {
    env_cargo_home
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| home.map(|h| h.join(".cargo")))
}

/// Keep only a candidate root that currently exists as a directory. This backs
/// the "no-op when the dir doesn't exist" guarantee for the write root; the
/// existence check is split out so it can be tested without depending on
/// process-wide `$CARGO_HOME`/home state (both sandbox runners also skip
/// non-existent roots, so this is defense in depth for the doc contract).
fn existing_dirs(candidate: Option<PathBuf>) -> Vec<PathBuf> {
    candidate.filter(|dir| dir.is_dir()).into_iter().collect()
}

/// Attach the `write_todos` tool and the per-turn reminder to a Main agent.
pub(crate) fn attach(agent: Agent<Arc<dyn Model>>, tracker: &PlanTracker) -> Agent<Arc<dyn Model>> {
    agent
        .with_tool(tracker.write_todos_tool())
        .with_dynamic_prompt(tracker.dynamic_prompt())
}

/// Build the headless [`Tracking`] bundle from one tracker. The tool, reminder,
/// and store all share the tracker's state.
pub(crate) fn headless_tracking(tracker: PlanTracker) -> Tracking {
    Tracking {
        todos_tool: tracker.write_todos_tool(),
        reminder: tracker.dynamic_prompt(),
        store: Arc::new(CliReportStore(tracker)),
        worker_post_build: None,
    }
}

/// The most recent non-empty assistant message text in `session` - the report a
/// Plan/Review agent produced before handing off to Main.
pub(crate) fn last_assistant_text(session: &dyn Session) -> Option<String> {
    session
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && !m.text_content().is_empty())
        .map(|m| m.text_content())
}

/// A resolved Plan/Review->Main handover: the report text to persist and the
/// user instruction (if any) that selects which of its items to act on.
pub(crate) struct Handover {
    pub report: String,
    pub instruction: Option<String>,
}

/// Resolve which text is the handed-over report and which (if any) is a separate
/// user selection over it.
///
/// When the outgoing agent left assistant text, that text *is* the report and
/// `step_with` is a separate selection instruction (`/fix only item 3`) that
/// must be preserved. When there is no assistant text - a model-driven handoff
/// whose payload itself is the report - `step_with` becomes the report and the
/// model chooses the items. Returns `None` only when there is nothing to persist
/// (no assistant text and no `step_with`).
pub(crate) fn resolve_handover(
    session_report: Option<String>,
    step_with: Option<String>,
) -> Option<Handover> {
    match session_report {
        Some(report) => Some(Handover {
            report,
            instruction: step_with,
        }),
        None => step_with.map(|report| Handover {
            report,
            instruction: None,
        }),
    }
}

/// Directive injected into Main once a plan/review is persisted.
///
/// `instruction` is the user's own request that triggered the handover (e.g.
/// `/fix only item 3`, `/approve`). When present it is authoritative - the
/// model must record exactly the items it names, not the whole report - so it
/// leads, and the file pointer follows. When absent (a model-driven handoff
/// whose payload *was* the report), the model picks the items itself.
pub(crate) fn report_directive(kind: AgentKind, path: &Path, instruction: Option<&str>) -> String {
    let what = match kind {
        AgentKind::Plan => "plan",
        AgentKind::Review => "review",
        AgentKind::Main => "report",
    };
    match instruction {
        Some(instruction) => format!(
            "{instruction}\n\n\
             The full {what} is saved to {}. The request above is authoritative about WHICH \
             items to act on - call write_todos with exactly those (all of them if it asks for \
             all, only the named ones if it is selective), then work through them, re-reading \
             the file with ReadFile whenever you need full detail.",
            path.display()
        ),
        None => format!(
            "The {what} has been saved to {}. Decide which items to act on, call write_todos with \
             that list, then work through it - re-read the file with ReadFile whenever you need \
             full detail.",
            path.display()
        ),
    }
}

struct CliReportStore(PlanTracker);

impl ReportStore for CliReportStore {
    fn save_plan(&self, content: &str) {
        if let Err(e) = self.0.save_plan(content) {
            tracing::warn!("failed to persist plan to tracker: {e}");
        }
    }

    fn save_review(&self, content: &str) {
        if let Err(e) = self.0.save_review(content) {
            tracing::warn!("failed to persist review to tracker: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_with_instruction_makes_the_selection_authoritative() {
        // Regression: `/fix only item 3` must not be discarded - the model
        // should record exactly that item, not the whole review.
        let directive = report_directive(
            AgentKind::Review,
            Path::new("/s/reviews/r.md"),
            Some("Address these review items: only item 3"),
        );
        assert!(directive.starts_with("Address these review items: only item 3"));
        assert!(directive.contains("authoritative about WHICH"));
        assert!(directive.contains("/s/reviews/r.md"));
    }

    #[test]
    fn directive_without_instruction_lets_the_model_choose() {
        let directive = report_directive(AgentKind::Plan, Path::new("/s/plans/p.md"), None);
        assert!(directive.contains("Decide which items to act on"));
        assert!(!directive.contains("authoritative about WHICH"));
        assert!(directive.contains("/s/plans/p.md"));
    }

    #[test]
    fn handover_with_assistant_text_keeps_step_with_as_the_selection() {
        // The Plan agent left a report in the transcript; `/fix only item 3`
        // is a separate selection that must survive as the instruction.
        let h = resolve_handover(
            Some("# Plan\n\n1. a\n2. b\n3. c".to_string()),
            Some("only item 3".to_string()),
        )
        .unwrap();
        assert!(h.report.starts_with("# Plan"));
        assert_eq!(h.instruction.as_deref(), Some("only item 3"));
    }

    #[test]
    fn handover_without_assistant_text_treats_step_with_as_the_report() {
        // A model-driven handoff with no assistant text: the payload itself is
        // the report and there is no separate selection.
        let h = resolve_handover(None, Some("the whole plan".to_string())).unwrap();
        assert_eq!(h.report, "the whole plan");
        assert!(h.instruction.is_none());
    }

    #[test]
    fn handover_with_nothing_to_persist_is_none() {
        assert!(resolve_handover(None, None).is_none());
    }

    #[test]
    fn ancestor_cargo_dirs_finds_ancestor_and_skips_cwd() {
        // A `.cargo` in a parent dir (the alset-dev `[patch]` overlay case) must
        // be surfaced, while the working dir's own `.cargo` is skipped - it is
        // already inside the readable project root.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join(".cargo")).unwrap();
        let cwd = root.join("a/b");
        std::fs::create_dir_all(cwd.join(".cargo")).unwrap();

        let dirs = ancestor_cargo_dirs(&cwd);

        // Paths are returned as found (OsSandbox canonicalizes read roots), so
        // compare against the same non-canonicalized joins the walk produces.
        let ancestor = root.join(".cargo");
        let own = cwd.join(".cargo");
        assert!(
            dirs.contains(&ancestor),
            "ancestor .cargo should be surfaced: {dirs:?}"
        );
        assert!(
            !dirs.contains(&own),
            "the cwd's own .cargo must be skipped: {dirs:?}"
        );
    }

    #[test]
    fn ancestor_cargo_dirs_empty_without_ancestor_config() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path().join("x/y/z");
        std::fs::create_dir_all(&cwd).unwrap();
        assert!(ancestor_cargo_dirs(&cwd).is_empty());
    }

    #[test]
    fn cargo_home_prefers_env_over_default() {
        // An explicit CARGO_HOME wins - matching cargo's own precedence.
        let env = PathBuf::from("/custom/cargo");
        let home = PathBuf::from("/home/user");
        assert_eq!(
            cargo_home_from(Some(env.clone()), Some(home)),
            Some(env),
            "CARGO_HOME must override the ~/.cargo default"
        );
    }

    #[test]
    fn cargo_home_falls_back_to_dot_cargo_under_home() {
        let home = PathBuf::from("/home/user");
        assert_eq!(
            cargo_home_from(None, Some(home.clone())),
            Some(home.join(".cargo")),
            "without CARGO_HOME the default is ~/.cargo"
        );
    }

    #[test]
    fn cargo_home_none_without_env_or_home() {
        assert_eq!(cargo_home_from(None, None), None);
    }

    #[test]
    fn cargo_home_ignores_empty_env() {
        // An empty CARGO_HOME is treated as unset, matching cargo, so the
        // ~/.cargo default still wins rather than yielding an empty path.
        let home = PathBuf::from("/home/user");
        assert_eq!(
            cargo_home_from(Some(PathBuf::new()), Some(home.clone())),
            Some(home.join(".cargo")),
            "empty CARGO_HOME must fall back to ~/.cargo"
        );
    }

    #[test]
    fn existing_dirs_keeps_a_real_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        assert_eq!(existing_dirs(Some(dir.clone())), vec![dir]);
    }

    #[test]
    fn existing_dirs_drops_a_missing_path() {
        // The existence guard behind sandbox_write_roots' "no-op when the dir
        // doesn't exist" contract: a resolved-but-absent $CARGO_HOME yields no
        // write root rather than one pointing at nothing.
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        assert!(existing_dirs(Some(missing)).is_empty());
    }

    #[test]
    fn existing_dirs_drops_none() {
        assert!(existing_dirs(None).is_empty());
    }
}
