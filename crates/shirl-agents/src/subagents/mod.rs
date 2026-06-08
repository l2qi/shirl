// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sweet_agent::{Agent, SubagentContext, TurnResult};
use sweet_core::{Model, ToolError};

pub mod diagnose;
pub mod explain;
pub mod explore;
pub mod testgen;
pub mod web_research;

/// Run a one-shot leaf subagent.
///
/// `configure` receives an `Agent` pre-loaded with `instructions` and should
/// attach the tools the subagent needs. A `TurnResult::Handoff` is rejected as
/// an error since leaf subagents are not part of the handoff graph.
pub(crate) async fn run_leaf(
    name: &'static str,
    instructions: &'static str,
    prompt: String,
    ctx: SubagentContext,
    configure: impl FnOnce(Agent<Arc<dyn Model>>) -> Agent<Arc<dyn Model>>,
) -> Result<String, ToolError> {
    let model = ctx.parent_model.ok_or_else(|| {
        ToolError::Execution(
            format!(
                "{name} subagent requires a shared parent model \
                 (parent must use Agent::new_shared)"
            )
            .into(),
        )
    })?;

    let mut agent = configure(Agent::new(model).with_instructions(instructions));

    let outcome = agent
        .step(prompt)
        .await
        .map_err(|e| ToolError::Execution(e.to_string().into()))?;
    match outcome {
        TurnResult::Message(m) => Ok(m.text_content()),
        TurnResult::Handoff { .. } => Err(ToolError::Execution(
            format!("unexpected handoff from {name}").into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_agent::SubagentContext;

    /// All subagents must reject invocation when the parent didn't share its
    /// model (i.e. parent built with `Agent::new` instead of `new_shared`).
    /// The error message mentions the subagent name so the user can act on it.
    #[tokio::test]
    async fn run_leaf_errors_without_parent_model() {
        let ctx = SubagentContext {
            depth: 1,
            parent_model: None,
        };
        let err = run_leaf("explore", "instr", "prompt".into(), ctx, |a| a)
            .await
            .expect_err("must reject missing parent_model");
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution variant");
        };
        let rendered = msg.to_string();
        assert!(rendered.contains("explore"));
        assert!(rendered.contains("shared parent model"));
    }
}
