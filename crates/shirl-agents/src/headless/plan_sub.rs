// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use sweet_agent::{SubagentContext, SubagentHandler, SubagentSpec};
use sweet_core::permission::ToolRisk;
use sweet_core::tool::ToolError;

use crate::agents::AgentKind;
use crate::headless::{run_worker_turn, ReportStore, WorkerDeps};

#[derive(Deserialize, JsonSchema)]
struct PlanInput {
    /// Description of the task to plan. Be specific about files, requirements, and constraints.
    task: String,
}

pub(crate) fn plan_subagent_spec(deps: WorkerDeps, store: Arc<dyn ReportStore>) -> SubagentSpec {
    SubagentSpec::new(
        "plan",
        "Produce a numbered implementation plan. The task parameter describes \
         what needs planning - be specific about files, requirements, and constraints.",
        serde_json::to_value(schemars::schema_for!(PlanInput)).expect("schema for PlanInput"),
        PlanHandler { deps, store },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct PlanHandler {
    deps: WorkerDeps,
    store: Arc<dyn ReportStore>,
}

#[sweet_core::async_trait]
impl SubagentHandler for PlanHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: PlanInput = serde_json::from_value(args)?;
        let report = run_worker_turn(
            AgentKind::Plan,
            &self.deps,
            input.task,
            Vec::new(),
            Vec::new(),
        )
        .await?;
        // Persist the plan so the implement worker can re-anchor on it even
        // after its own history compaction.
        self.store.save_plan(&report);
        Ok(report)
    }
}
