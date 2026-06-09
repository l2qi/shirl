// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use sweet_agent::{DynamicPrompt, SubagentContext, SubagentHandler, SubagentSpec};
use sweet_core::tool::{ToolError, ToolSpec};

use crate::agents::AgentKind;
use crate::headless::{run_worker_turn, WorkerDeps};

#[derive(Deserialize, JsonSchema)]
struct ImplementInput {
    /// What to implement. The worker sees a snapshot of the orchestrator's
    /// conversation — including every prior worker's final report — so it has
    /// the plan in context. Use this parameter for new instructions or
    /// clarifications, not to re-state the plan verbatim.
    instructions: String,
}

pub(crate) fn implement_subagent_spec(
    deps: WorkerDeps,
    todos_tool: ToolSpec,
    reminder: Arc<dyn DynamicPrompt>,
) -> SubagentSpec {
    SubagentSpec::new(
        "implement",
        "Write code, edit files, and run commands. The worker sees a snapshot \
         of the orchestrator's conversation history (including the plan's \
         final report) and acts on the instructions parameter.",
        serde_json::to_value(schemars::schema_for!(ImplementInput))
            .expect("schema for ImplementInput"),
        ImplementHandler {
            deps,
            todos_tool,
            reminder,
        },
    )
}

struct ImplementHandler {
    deps: WorkerDeps,
    todos_tool: ToolSpec,
    reminder: Arc<dyn DynamicPrompt>,
}

#[sweet_core::async_trait]
impl SubagentHandler for ImplementHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: ImplementInput = serde_json::from_value(args)?;
        run_worker_turn(
            AgentKind::Main,
            &self.deps,
            input.instructions,
            vec![self.todos_tool.clone()],
            vec![self.reminder.clone()],
        )
        .await
    }
}
