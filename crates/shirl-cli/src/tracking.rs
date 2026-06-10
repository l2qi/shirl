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
/// can't be resolved (workflow features then degrade off — unreachable in
/// practice, since the session itself lives under the same home directory).
pub(crate) fn load_tracker(session_id: &SessionId) -> Option<PlanTracker> {
    shirl_core::session_dir(session_id)
        .ok()
        .map(PlanTracker::load)
}

/// Sandbox read roots that let the agent read back the workflow tracker's
/// plan/review files under `~/.shirl/sessions` (read-only) without re-exposing
/// the rest of the home directory. Empty if the home dir can't be resolved.
pub(crate) fn sandbox_read_roots() -> Vec<PathBuf> {
    shirl_core::sessions_root().ok().into_iter().collect()
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

/// The most recent non-empty assistant message text in `session` — the report a
/// Plan/Review agent produced before handing off to Main.
pub(crate) fn last_assistant_text(session: &dyn Session) -> Option<String> {
    session
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && !m.text_content().is_empty())
        .map(|m| m.text_content())
}

/// A resolved Plan/Review→Main handover: the report text to persist and the
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
/// must be preserved. When there is no assistant text — a model-driven handoff
/// whose payload itself is the report — `step_with` becomes the report and the
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
/// `/fix only item 3`, `/approve`). When present it is authoritative — the
/// model must record exactly the items it names, not the whole report — so it
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
             items to act on — call write_todos with exactly those (all of them if it asks for \
             all, only the named ones if it is selective), then work through them, re-reading \
             the file with ReadFile whenever you need full detail.",
            path.display()
        ),
        None => format!(
            "The {what} has been saved to {}. Decide which items to act on, call write_todos with \
             that list, then work through it — re-read the file with ReadFile whenever you need \
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
        // Regression: `/fix only item 3` must not be discarded — the model
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
}
