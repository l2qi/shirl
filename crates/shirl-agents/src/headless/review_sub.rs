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
struct ReviewInput {
    /// What to focus the review on. E.g. "Review changes to bar.rs for
    /// correctness" or "Check for edge cases in the error handling."
    focus: String,
}

pub(crate) fn review_subagent_spec(deps: WorkerDeps, store: Arc<dyn ReportStore>) -> SubagentSpec {
    SubagentSpec::new(
        "review",
        "Review code for correctness and quality. The focus parameter says \
         what to review, e.g. \"Review changes to bar.rs for correctness\" or \
         \"Check for edge cases in the error handling.\"",
        serde_json::to_value(schemars::schema_for!(ReviewInput)).expect("schema for ReviewInput"),
        ReviewHandler { deps, store },
    )
    .with_risk(ToolRisk::ReadOnly)
}

struct ReviewHandler {
    deps: WorkerDeps,
    store: Arc<dyn ReportStore>,
}

#[sweet_core::async_trait]
impl SubagentHandler for ReviewHandler {
    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: SubagentContext,
    ) -> Result<String, ToolError> {
        let input: ReviewInput = serde_json::from_value(args)?;
        let report = run_worker_turn(
            AgentKind::Review,
            &self.deps,
            input.focus,
            Vec::new(),
            Vec::new(),
        )
        .await?;
        self.store.save_review(&report);
        Ok(report)
    }
}
