// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Workflow tracker: persists handed-over plans/reviews and the live todo list,
//! and surfaces them to the main agent as a compaction-proof reminder.
//!
//! The main agent drifts off a plan on long tasks because the plan, injected as
//! a single user message, gets summarized away once history compaction kicks in.
//! Two channels survive compaction: on-disk files and the per-turn system
//! prompt. This module uses both - the report is written to disk (the agent can
//! re-read it with `ReadFile` any time) and a live todo list + a pointer to the
//! report is re-rendered into the system instructions every turn via
//! [`DynamicPrompt`].
//!
//! State lives under the session directory (`~/.shirl/sessions/<id>/`):
//!
//! ```text
//! plans/<YYYYMMDD-HHMMSS>-<slug>.md
//! reviews/<YYYYMMDD-HHMMSS>-<slug>.md
//! tracker.json   (the active source + the todo list)
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sweet_agent::DynamicPrompt;
use sweet_core::permission::ToolRisk;
use sweet_core::tool::{ToolError, ToolHandler, ToolSpec};

/// Which workflow agent produced the report currently driving the main agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Plan,
    Review,
}

impl SourceKind {
    fn label(self) -> &'static str {
        match self {
            SourceKind::Plan => "plan",
            SourceKind::Review => "review",
        }
    }

    /// Session-dir subdirectory the report files live in.
    fn subdir(self) -> &'static str {
        match self {
            SourceKind::Plan => "plans",
            SourceKind::Review => "reviews",
        }
    }
}

/// Pointer to the one report driving current work, relative to the session dir.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReportRef {
    kind: SourceKind,
    path: PathBuf,
}

/// Status of a single todo item.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    #[default]
    Pending,
    InProgress,
    Done,
}

impl TodoStatus {
    fn marker(self) -> &'static str {
        match self {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Done => "[x]",
        }
    }
}

/// A stored todo item. `id` is a 1-based display index assigned on write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TodoItem {
    id: u32,
    text: String,
    status: TodoStatus,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TrackerState {
    #[serde(default)]
    source: Option<ReportRef>,
    #[serde(default)]
    todos: Vec<TodoItem>,
}

struct TrackerInner {
    session_dir: PathBuf,
    state: Mutex<TrackerState>,
}

impl TrackerInner {
    fn tracker_path(&self) -> PathBuf {
        self.session_dir.join("tracker.json")
    }

    fn persist(&self, state: &TrackerState) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.session_dir)?;
        let json = serde_json::to_string_pretty(state).map_err(std::io::Error::other)?;
        std::fs::write(self.tracker_path(), json)
    }

    /// Lock the state, recovering the guard if a previous holder panicked. The
    /// tracker is a plain data record (todos + a report pointer); a poisoned
    /// lock can't leave it logically inconsistent, so recovering is safe and
    /// keeps a workflow feature from taking down the whole session.
    fn lock_state(&self) -> std::sync::MutexGuard<'_, TrackerState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Per-session workflow state. Cheap to clone (`Arc` inside) - clone it to hand
/// the same state to the `write_todos` tool, the [`DynamicPrompt`], and the
/// caller that persists incoming plans/reviews.
#[derive(Clone)]
pub struct PlanTracker {
    inner: Arc<TrackerInner>,
}

impl PlanTracker {
    /// Open the tracker for a session directory, loading `tracker.json` if it
    /// exists (resume) or starting empty. A malformed file is ignored and
    /// treated as empty rather than failing the session.
    pub fn load(session_dir: PathBuf) -> Self {
        let state = std::fs::read_to_string(session_dir.join("tracker.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self {
            inner: Arc::new(TrackerInner {
                session_dir,
                state: Mutex::new(state),
            }),
        }
    }

    /// Persist a handed-over plan and make it the active source. Returns the
    /// absolute path of the written file.
    pub fn save_plan(&self, content: &str) -> std::io::Result<PathBuf> {
        self.save(SourceKind::Plan, content)
    }

    /// Persist a handed-over review and make it the active source. Returns the
    /// absolute path of the written file.
    pub fn save_review(&self, content: &str) -> std::io::Result<PathBuf> {
        self.save(SourceKind::Review, content)
    }

    fn save(&self, kind: SourceKind, content: &str) -> std::io::Result<PathBuf> {
        let rel = PathBuf::from(kind.subdir()).join(report_filename(content));
        let abs = self.inner.session_dir.join(&rel);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&abs, content)?;

        let mut state = self.inner.lock_state();
        state.source = Some(ReportRef { kind, path: rel });
        self.inner.persist(&state)?;
        Ok(abs)
    }

    /// The `write_todos` tool, sharing this tracker's state. The model replaces
    /// its whole todo list on each call.
    pub fn write_todos_tool(&self) -> ToolSpec {
        ToolSpec::new(
            "write_todos",
            "Record or update your working todo list for the current task. Pass the \
             COMPLETE list every call - it REPLACES the previous one. Use it after \
             receiving a plan or review to track the items you will address (a subset of \
             the report's items is fine - honor what the user asked for), and for \
             multi-step direct work. Set each item's status (pending, in_progress, done) \
             and keep it current as you work. The list is shown back to you every turn \
             until all items are done. Skip it for trivial one-step requests.",
            serde_json::to_value(schemars::schema_for!(WriteTodosArgs)).expect("schema"),
            WriteTodosHandler {
                inner: self.inner.clone(),
            },
        )
        .with_risk(ToolRisk::ReadOnly)
    }

    /// This tracker as a per-turn dynamic prompt for the agent's system
    /// instructions.
    pub fn dynamic_prompt(&self) -> Arc<dyn DynamicPrompt> {
        Arc::new(self.clone())
    }
}

impl DynamicPrompt for PlanTracker {
    fn render(&self) -> Option<String> {
        let state = self.inner.lock_state();
        render_state(&self.inner.session_dir, &state)
    }
}

/// Render the reminder block, or `None` when there is nothing to anchor (no
/// active report and no todos) - so trivial direct work carries zero overhead.
fn render_state(session_dir: &Path, state: &TrackerState) -> Option<String> {
    if state.source.is_none() && state.todos.is_empty() {
        return None;
    }

    let mut out = String::new();
    if let Some(src) = &state.source {
        let abs = session_dir.join(&src.path);
        out.push_str(&format!(
            "## Active {}\nSource file: {} - re-read it with ReadFile for full detail.\n",
            src.kind.label(),
            abs.display(),
        ));
    }
    if !state.todos.is_empty() {
        out.push_str(
            "\n## Todo list (work through these; do not start unrelated work until every \
             item is done)\n",
        );
        for item in &state.todos {
            out.push_str(&format!(
                "{} {}. {}\n",
                item.status.marker(),
                item.id,
                item.text,
            ));
        }
    }
    Some(out.trim_end().to_string())
}

#[derive(Deserialize, JsonSchema)]
struct WriteTodosArgs {
    /// The complete todo list. Each call replaces the previous list entirely.
    todos: Vec<TodoArg>,
}

#[derive(Deserialize, JsonSchema)]
struct TodoArg {
    /// Short, imperative description of the work item.
    text: String,
    /// Item status. Defaults to `pending`.
    #[serde(default)]
    status: TodoStatus,
}

struct WriteTodosHandler {
    inner: Arc<TrackerInner>,
}

#[sweet_core::async_trait]
impl ToolHandler for WriteTodosHandler {
    async fn call(&self, args: serde_json::Value) -> Result<String, ToolError> {
        let args: WriteTodosArgs = serde_json::from_value(args)?;
        let todos: Vec<TodoItem> = args
            .todos
            .into_iter()
            .enumerate()
            .map(|(i, t)| TodoItem {
                id: (i + 1) as u32,
                text: t.text,
                status: t.status,
            })
            .collect();

        let mut state = self.inner.lock_state();
        state.todos = todos;
        self.inner
            .persist(&state)
            .map_err(|e| ToolError::Execution(e.to_string().into()))?;
        Ok(render_state(&self.inner.session_dir, &state)
            .unwrap_or_else(|| "Todo list cleared.".to_string()))
    }
}

/// Build a sortable, human-readable filename from the report's first heading
/// line: `<YYYYMMDD-HHMMSS>-<slug>.md`.
fn report_filename(content: &str) -> String {
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
    format!("{ts}-{}.md", slug(content))
}

fn slug(content: &str) -> String {
    let title = content
        .lines()
        .map(|l| l.trim_start_matches('#').trim())
        .find(|l| !l.is_empty())
        .unwrap_or("");

    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.extend(ch.to_lowercase());
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
        if slug.len() >= 40 {
            break;
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "report".to_string()
    } else {
        slug.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_todos(tracker: &PlanTracker, items: &[(&str, TodoStatus)]) {
        let mut state = tracker.inner.state.lock().unwrap();
        state.todos = items
            .iter()
            .enumerate()
            .map(|(i, (text, status))| TodoItem {
                id: (i + 1) as u32,
                text: (*text).to_string(),
                status: *status,
            })
            .collect();
        tracker.inner.persist(&state).unwrap();
    }

    #[test]
    fn render_is_none_when_empty() {
        let dir = TempDir::new().unwrap();
        let tracker = PlanTracker::load(dir.path().to_path_buf());
        assert_eq!(tracker.render(), None);
    }

    #[test]
    fn save_plan_writes_file_and_sets_source() {
        let dir = TempDir::new().unwrap();
        let tracker = PlanTracker::load(dir.path().to_path_buf());

        let path = tracker.save_plan("# Add auth\n\n1. do x").unwrap();
        assert!(path.exists());
        assert!(path.starts_with(dir.path().join("plans")));
        assert!(path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("add-auth"));

        let rendered = tracker.render().unwrap();
        assert!(rendered.contains("## Active plan"));
        assert!(rendered.contains(&path.display().to_string()));
    }

    #[test]
    fn save_review_replaces_plan_as_source() {
        let dir = TempDir::new().unwrap();
        let tracker = PlanTracker::load(dir.path().to_path_buf());
        tracker.save_plan("# Plan").unwrap();
        tracker.save_review("# Review findings").unwrap();

        let state = tracker.inner.state.lock().unwrap();
        assert_eq!(state.source.as_ref().unwrap().kind, SourceKind::Review);
    }

    #[test]
    fn todos_render_with_status_markers() {
        let dir = TempDir::new().unwrap();
        let tracker = PlanTracker::load(dir.path().to_path_buf());
        write_todos(
            &tracker,
            &[
                ("done item", TodoStatus::Done),
                ("current", TodoStatus::InProgress),
                ("later", TodoStatus::Pending),
            ],
        );
        let rendered = tracker.render().unwrap();
        assert!(rendered.contains("[x] 1. done item"));
        assert!(rendered.contains("[~] 2. current"));
        assert!(rendered.contains("[ ] 3. later"));
    }

    #[test]
    fn load_restores_persisted_state() {
        let dir = TempDir::new().unwrap();
        {
            let tracker = PlanTracker::load(dir.path().to_path_buf());
            tracker.save_plan("# Resume me").unwrap();
            write_todos(&tracker, &[("item", TodoStatus::InProgress)]);
        }
        // Fresh load from the same dir - simulates --resume.
        let reloaded = PlanTracker::load(dir.path().to_path_buf());
        let rendered = reloaded.render().unwrap();
        assert!(rendered.contains("## Active plan"));
        assert!(rendered.contains("[~] 1. item"));
    }

    #[test]
    fn render_combines_active_source_and_todos() {
        let dir = TempDir::new().unwrap();
        let tracker = PlanTracker::load(dir.path().to_path_buf());
        let path = tracker.save_plan("# Refactor X\n\nDetails here").unwrap();
        write_todos(
            &tracker,
            &[
                ("done item", TodoStatus::Done),
                ("remaining", TodoStatus::Pending),
            ],
        );

        let rendered = tracker.render().unwrap();
        // Active source section present with absolute path.
        assert!(rendered.starts_with("## Active plan\nSource file: "));
        assert!(rendered.contains(&path.display().to_string()));
        // Todos section follows, separated by a blank line.
        assert!(rendered.contains("\n\n## Todo list"));
        assert!(rendered.contains("[x] 1. done item"));
        assert!(rendered.contains("[ ] 2. remaining"));
        // Source comes before todos.
        let src_pos = rendered.find("## Active plan").unwrap();
        let todo_pos = rendered.find("## Todo list").unwrap();
        assert!(src_pos < todo_pos);
    }

    #[test]
    fn slug_falls_back_when_no_title() {
        assert_eq!(slug(""), "report");
        assert_eq!(slug("\n\n  "), "report");
        assert_eq!(slug("# Fix the Bug!"), "fix-the-bug");
    }

    // The whole point of the feature: the reminder rides in the system prompt,
    // which is rebuilt from the tracker every turn - so it survives compaction
    // wiping the session, the exact failure that made Main drift off the plan.
    #[tokio::test]
    async fn reminder_survives_session_being_cleared() {
        use sweet_agent::test_util::MockModel;
        use sweet_agent::Agent;

        let dir = TempDir::new().unwrap();
        let tracker = PlanTracker::load(dir.path().to_path_buf());
        tracker.save_plan("# Build feature\n\n1. step one").unwrap();
        write_todos(&tracker, &[("step one", TodoStatus::InProgress)]);

        let mut agent = Agent::new(MockModel::with_replies(["a", "b"]))
            .with_instructions("base")
            .with_dynamic_prompt(tracker.dynamic_prompt());

        agent.step("hi").await.unwrap();
        // Simulate compaction dropping the whole transcript.
        let _ = agent.take_session();
        agent.step("again").await.unwrap();

        let calls = agent.model().calls();
        let system = &calls[1][0];
        assert_eq!(system.role, sweet_core::Role::System);
        let text = system.text_content();
        assert!(text.contains("## Active plan"), "got: {text}");
        assert!(text.contains("[~] 1. step one"), "got: {text}");
    }
}
