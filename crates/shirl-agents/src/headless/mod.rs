// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

mod implement_sub;
mod orchestrator;
mod plan_sub;
mod review_sub;

use std::collections::HashMap;
use std::sync::Arc;

use sweet_agent::{Agent, DynamicPrompt, ExtensionRegistry, TurnResult};
use sweet_core::sandbox::Sandbox;
use sweet_core::tool::{ToolError, ToolSpec};
use sweet_core::SharedSessionHandle;
use sweet_core::{InMemorySession, MemoryItem, Message, Model, Role, Session};

use crate::agents::{self, AgentKind, SharedWebSearchBackend};

pub use orchestrator::ORCHESTRATOR_PROMPT;

/// Sink for persisting a worker's handed-over report to durable storage.
///
/// Implemented by the binary (over `shirl-core`'s tracker) and injected here so
/// `shirl-agents` stays decoupled from `shirl-core`. The plan/review workers
/// call it with their final report so it survives the implement worker's own
/// history compaction.
pub trait ReportStore: Send + Sync {
    fn save_plan(&self, content: &str);
    fn save_review(&self, content: &str);
}

/// The workflow-tracking pieces the orchestrator threads into its workers. All
/// three are derived from one tracker by the binary, so they share state: the
/// plan/review workers persist via `store`, and the implement worker's Main
/// agent gets the `todos_tool` plus the `reminder` that re-renders the live
/// list into its system prompt every turn.
#[derive(Clone)]
pub struct Tracking {
    pub todos_tool: ToolSpec,
    pub reminder: Arc<dyn DynamicPrompt>,
    pub store: Arc<dyn ReportStore>,
    /// Optional post-build hook applied to each worker agent after
    /// construction. The binary fills this with auto-compaction installation
    /// so workers don't exceed the context window on long tool-call chains.
    pub worker_post_build: Option<WorkerPostBuild>,
}

/// The shared dependencies every headless worker needs to build its child
/// agent. Bundled into one struct so each worker spec takes a single `deps`
/// argument; cloned once per worker (all fields are cheap `Arc`/`Option`
/// handles).
#[derive(Clone)]
pub(crate) struct WorkerDeps {
    pub model: Arc<dyn Model>,
    pub sandbox: Arc<dyn Sandbox>,
    pub extensions: Arc<ExtensionRegistry>,
    pub web_search: Option<SharedWebSearchBackend>,
    pub mcp_specs: Arc<Vec<ToolSpec>>,
    pub parent_session: SharedSessionHandle,
    /// Optional post-build hook applied to each worker agent after
    /// construction. The binary fills this with auto-compaction installation
    /// so workers don't exceed the context window on long tool-call chains.
    pub post_build: Option<WorkerPostBuild>,
}

/// Post-build hook applied to each worker agent after construction.
/// The binary fills this with auto-compaction installation so workers
/// don't exceed the context window on long tool-call chains.
pub type WorkerPostBuild =
    Arc<dyn Fn(Agent<Arc<dyn Model>>) -> Agent<Arc<dyn Model>> + Send + Sync>;

// Same construction-wiring shape as `agents::build_agent`.
#[allow(clippy::too_many_arguments)]
pub fn build_orchestrator(
    model: Arc<dyn Model>,
    extensions: Arc<ExtensionRegistry>,
    web_search: Option<SharedWebSearchBackend>,
    session: Box<dyn Session>,
    mcp_specs: &[ToolSpec],
    sandbox: Arc<dyn Sandbox>,
    tracking: Tracking,
    memory: Option<&crate::MemoryWiring>,
) -> Agent<Arc<dyn Model>> {
    let agent = orchestrator::build(
        model, extensions, web_search, session, mcp_specs, sandbox, tracking,
    );
    // The orchestrator's session is the persisted top-level transcript, so it
    // gets the Main memory policy. Workers run on ephemeral child sessions
    // and deliberately get none (see run_worker_turn).
    crate::memory::apply_memory(agent, AgentKind::Main, memory)
}

/// Shared invocation logic for the three headless subagents (plan, implement,
/// review).
///
/// Builds a fresh child agent of `kind` whose session is pre-loaded with a
/// snapshot of the orchestrator's transcript, runs a single turn with
/// `user_input`, and returns the child's final assistant message content for
/// the orchestrator to consume as a tool result. `TurnResult::Handoff` from a
/// child is surfaced as a `ToolError::Execution` — handoffs are an
/// interactive-mode concept with no meaning in headless runs.
/// `extra_tools` and `reminders` augment the child agent after construction —
/// the implement worker passes the `write_todos` tool and the todo reminder so
/// the workflow state anchors its Main agent; plan/review pass none.
pub(crate) async fn run_worker_turn(
    kind: AgentKind,
    deps: &WorkerDeps,
    user_input: String,
    extra_tools: Vec<ToolSpec>,
    reminders: Vec<Arc<dyn DynamicPrompt>>,
) -> Result<String, ToolError> {
    let child_session = child_session_from_snapshot(&deps.parent_session)?;
    let mut agent = agents::build_agent(
        kind,
        deps.model.clone(),
        &deps.extensions,
        deps.web_search.clone(),
        child_session,
        &deps.mcp_specs,
        deps.sandbox.clone(),
        // Workers run on ephemeral child sessions: no long-term memory, and
        // in particular no distillation from scratch transcripts.
        None,
    );
    if let Some(post_build) = &deps.post_build {
        agent = post_build(agent);
    }
    for tool in extra_tools {
        agent = agent.with_tool(tool);
    }
    for reminder in reminders {
        agent = agent.with_dynamic_prompt(reminder);
    }

    let result = agent
        .step(user_input)
        .await
        .map_err(|e| ToolError::Execution(e.to_string().into()))?;
    match result {
        TurnResult::Message(msg) => Ok(msg.text_content()),
        TurnResult::Handoff { target, payload } => Err(ToolError::Execution(
            format!(
                "handoffs not supported in headless subagents (target: {target}{})",
                payload
                    .as_deref()
                    .map(|p| format!(", payload: {p}"))
                    .unwrap_or_default()
            )
            .into(),
        )),
    }
}

/// Build a child session that mirrors the orchestrator's transcript without
/// leaking tool-call references the child can't satisfy.
///
/// The orchestrator's transcript contains assistant messages with `tool_calls`
/// for `plan`/`implement`/`review` and paired `Role::Tool` results — tools the
/// child agent does not have registered. Passing them through raw confuses
/// schema-strict providers (OpenAI tool_choice validation, Anthropic
/// tool_use_id checks).
///
/// Replacement strategy:
/// - `Role::Tool` (worker report) → synthetic assistant message tagged with
///   the originating tool name, so the child still sees prior worker reports.
/// - Assistant carriers with `tool_calls` → keep any text content, drop the
///   tool_calls themselves. Pure carriers (empty content) are skipped.
/// - Other messages → pass through with `tool_calls`/`tool_call_id` cleared.
///
/// Per-message metadata (`token_count`, `context_tokens`, `compacted`) is
/// zeroed because those values belong to the orchestrator's API calls, not
/// the child's.
fn child_session_from_snapshot(
    handle: &SharedSessionHandle,
) -> Result<Box<dyn Session>, ToolError> {
    let mut session = InMemorySession::new();
    let mut id_to_name: HashMap<String, String> = HashMap::new();

    for msg in handle.snapshot_messages() {
        match msg.role {
            Role::Tool => {
                let name = msg
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| id_to_name.get(id))
                    .cloned()
                    .unwrap_or_else(|| "worker".to_string());
                let synthetic =
                    Message::assistant(format!("[Prior `{name}` report]\n{}", msg.text_content()));
                push_msg(&mut session, synthetic)?;
            }
            Role::Assistant if !msg.tool_calls.is_empty() => {
                for tc in &msg.tool_calls {
                    id_to_name.insert(tc.id.clone(), tc.name.clone());
                }
                if !msg.text_content().is_empty() {
                    push_msg(&mut session, strip_metadata(&msg))?;
                }
            }
            _ => {
                push_msg(&mut session, strip_metadata(&msg))?;
            }
        }
    }
    Ok(Box::new(session))
}

/// Strip tool-call fields and per-message metadata from `msg`.
///
/// Tool calls and IDs are removed because the child doesn't register the
/// orchestrator's tools, and schema-strict providers reject dangling
/// references. Token counts and the compacted flag belong to the
/// orchestrator's API calls, not the child's.
fn strip_metadata(msg: &Message) -> Message {
    Message {
        tool_calls: Vec::new(),
        tool_call_id: None,
        token_count: None,
        context_tokens: None,
        compacted: false,
        ..msg.clone()
    }
}

fn push_msg(session: &mut InMemorySession, msg: Message) -> Result<(), ToolError> {
    session
        .push(MemoryItem::Message(msg))
        .map_err(|e| ToolError::Execution(e.to_string().into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_agent::test_util::{MockModel, MockReply};
    use sweet_core::sandbox::DirectSandbox;
    use sweet_core::{SharedSession, ToolCall};

    fn handle_with(messages: Vec<Message>) -> SharedSessionHandle {
        let (mut shared, handle) = SharedSession::new(Box::new(InMemorySession::new()));
        for m in messages {
            shared.push(MemoryItem::Message(m)).unwrap();
        }
        handle
    }

    fn worker_deps(model: Arc<dyn Model>, parent_session: SharedSessionHandle) -> WorkerDeps {
        WorkerDeps {
            model,
            sandbox: Arc::new(DirectSandbox::new()),
            extensions: Arc::new(ExtensionRegistry::new()),
            web_search: None,
            mcp_specs: Arc::new(Vec::new()),
            parent_session,
            post_build: None,
        }
    }

    #[test]
    fn snapshot_filter_strips_tool_calls_and_inlines_reports() {
        let assistant_call = Message {
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "plan".into(),
                arguments: serde_json::json!({"task": "x"}),
            }],
            ..Message::assistant("")
        };
        let tool_result = Message::tool_result("call_1", "plan output");

        let handle = handle_with(vec![Message::user("do task"), assistant_call, tool_result]);

        let child = child_session_from_snapshot(&handle).unwrap();
        let msgs = child.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text_content(), "do task");
        assert!(msgs[0].tool_calls.is_empty());
        assert_eq!(msgs[1].role, Role::Assistant);
        assert!(msgs[1].text_content().contains("[Prior `plan` report]"));
        assert!(msgs[1].text_content().contains("plan output"));
        assert!(msgs[1].tool_calls.is_empty());
        assert!(msgs[1].tool_call_id.is_none());
    }

    #[test]
    fn snapshot_filter_preserves_assistant_text_alongside_tool_calls() {
        let assistant_mixed = Message {
            tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "implement".into(),
                arguments: serde_json::json!({}),
            }],
            ..Message::assistant("thinking out loud")
        };
        let handle = handle_with(vec![assistant_mixed]);

        let child = child_session_from_snapshot(&handle).unwrap();
        let msgs = child.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text_content(), "thinking out loud");
        assert!(msgs[0].tool_calls.is_empty());
    }

    #[test]
    fn snapshot_filter_uses_fallback_name_for_orphan_tool_result() {
        let orphan = Message::tool_result("missing", "result");
        let handle = handle_with(vec![orphan]);
        let child = child_session_from_snapshot(&handle).unwrap();
        let msgs = child.messages();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].text_content().contains("[Prior `worker` report]"));
    }

    #[tokio::test]
    async fn run_worker_turn_returns_assistant_message_content() {
        let model: Arc<dyn Model> = Arc::new(MockModel::with_replies(["worker reply"]));
        let (_shared, handle) = SharedSession::new(Box::new(InMemorySession::new()));
        let deps = worker_deps(model, handle);

        let result = run_worker_turn(
            AgentKind::Plan,
            &deps,
            "do the thing".to_string(),
            Vec::new(),
            Vec::new(),
        )
        .await
        .unwrap();

        assert_eq!(result, "worker reply");
    }

    #[tokio::test]
    async fn run_worker_turn_surfaces_handoff_as_descriptive_error() {
        // Force the Plan agent to invoke its transfer_to_main handoff. The
        // agent loop turns the handoff tool's `ToolError::Handoff` into a
        // `TurnResult::Handoff`, which run_worker_turn re-wraps as a
        // `ToolError::Execution` with the target and payload spelled out.
        let model: Arc<dyn Model> =
            Arc::new(MockModel::with_scripted(vec![MockReply::ToolCalls(vec![
                ToolCall {
                    id: "c1".into(),
                    name: "transfer_to_main".into(),
                    arguments: serde_json::json!({ "content": "approved plan" }),
                },
            ])]));
        let (_shared, handle) = SharedSession::new(Box::new(InMemorySession::new()));
        let deps = worker_deps(model, handle);

        let err = run_worker_turn(
            AgentKind::Plan,
            &deps,
            "plan it".to_string(),
            Vec::new(),
            Vec::new(),
        )
        .await
        .unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("handoffs not supported"), "got: {msg}");
        assert!(msg.contains("main"), "got: {msg}");
        assert!(msg.contains("approved plan"), "got: {msg}");
    }
}
