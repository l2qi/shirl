// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Long-term memory wiring for the shirl agents.
//!
//! The binary builds one [`MemoryWiring`] at startup (store, scopes,
//! distiller) and passes it to every agent build; the per-kind policy lives
//! here:
//!
//! - **Main** — full memory tools, per-turn recall, and (when a distiller is
//!   configured) automatic distillation.
//! - **Plan** — `memory_search` plus recall: planning benefits from past
//!   decisions but stays read-only, so no save/update/delete.
//! - **Review** — recall only.
//!
//! The headless orchestrator gets the Main policy: its session is the
//! persisted top-level transcript, while its workers run on ephemeral child
//! sessions that must not write long-term memory.

use std::sync::Arc;

use sweet_agent::{
    memory_distiller_capabilities, memory_recall_capabilities, Agent, MemoryDistiller, MemoryRecall,
};
use sweet_core::{Memory, MemoryScope, Model};
use sweet_memory::{memory_search_tool, memory_tools, MemoryToolset};

use crate::agents::AgentKind;

/// Store plus scope binding for one shirl run, built by the binary.
#[derive(Clone)]
pub struct MemoryWiring {
    pub store: Arc<dyn Memory>,
    /// Personal scope, searched everywhere.
    pub user_scope: MemoryScope,
    /// Current codebase's scope (git-root keyed); saves land here.
    pub project_scope: MemoryScope,
    /// Session id recorded as provenance on saves.
    pub session_id: String,
    /// Maximum memories injected into the system prompt per turn.
    pub recall_limit: usize,
    /// Present when auto-distillation is enabled. `Arc`-shared so the
    /// watermark survives agent rebuilds (mode switches) and the binary can
    /// flush pending items at session boundaries via
    /// [`MemoryDistiller::run_now`].
    pub distiller: Option<Arc<MemoryDistiller>>,
}

impl MemoryWiring {
    fn searchable_scopes(&self) -> Vec<MemoryScope> {
        vec![self.project_scope.clone(), self.user_scope.clone()]
    }

    fn toolset(&self) -> MemoryToolset {
        MemoryToolset::new(Arc::clone(&self.store), self.project_scope.clone())
            .with_searchable_scopes(self.searchable_scopes())
            .with_source_session(&self.session_id)
    }
}

/// Attach the kind-appropriate memory capabilities to `agent`. A `None`
/// wiring (memory disabled) is a no-op.
pub(crate) fn apply_memory<M: Model>(
    agent: Agent<M>,
    kind: AgentKind,
    wiring: Option<&MemoryWiring>,
) -> Agent<M> {
    let Some(wiring) = wiring else {
        return agent;
    };

    let recall = Arc::new(
        MemoryRecall::new(Arc::clone(&wiring.store), wiring.searchable_scopes())
            .with_limit(wiring.recall_limit),
    );
    let mut agent = agent
        .with_dynamic_prompt(recall.clone())
        .with_capabilities(memory_recall_capabilities(recall));

    match kind {
        AgentKind::Main => {
            agent = agent.with_tools(memory_tools(wiring.toolset()));
            if let Some(distiller) = &wiring.distiller {
                agent =
                    agent.with_capabilities(memory_distiller_capabilities(Arc::clone(distiller)));
            }
        }
        AgentKind::Plan => {
            agent = agent.with_tool(memory_search_tool(wiring.toolset()));
        }
        AgentKind::Review => {}
    }

    agent
}

#[cfg(test)]
mod tests {
    use super::*;
    use sweet_core::EphemeralMemory;

    fn wiring(distill: bool) -> MemoryWiring {
        let store: Arc<dyn Memory> = Arc::new(EphemeralMemory::new());
        MemoryWiring {
            user_scope: MemoryScope::User("default".into()),
            project_scope: MemoryScope::Project("/repo".into()),
            session_id: "s1".into(),
            recall_limit: 5,
            distiller: distill.then(|| {
                Arc::new(MemoryDistiller::new(
                    Arc::clone(&store),
                    MemoryScope::Project("/repo".into()),
                    sweet_agent::DistillConfig::default(),
                ))
            }),
            store,
        }
    }

    fn tool_names<M: Model>(agent: &Agent<M>) -> Vec<String> {
        agent.tools().iter().map(|t| t.name.clone()).collect()
    }

    fn test_agent() -> Agent<sweet_agent::test_util::MockModel> {
        Agent::new(sweet_agent::test_util::MockModel::with_replies(["ok"]))
    }

    #[test]
    fn main_gets_all_memory_tools() {
        let agent = apply_memory(test_agent(), AgentKind::Main, Some(&wiring(true)));
        let names = tool_names(&agent);
        for tool in [
            "memory_save",
            "memory_search",
            "memory_update",
            "memory_delete",
        ] {
            assert!(names.contains(&tool.to_string()), "missing {tool}");
        }
    }

    #[test]
    fn plan_gets_search_only() {
        let agent = apply_memory(test_agent(), AgentKind::Plan, Some(&wiring(false)));
        let names = tool_names(&agent);
        assert!(names.contains(&"memory_search".to_string()));
        assert!(!names.contains(&"memory_save".to_string()));
        assert!(!names.contains(&"memory_update".to_string()));
        assert!(!names.contains(&"memory_delete".to_string()));
    }

    #[test]
    fn review_gets_no_memory_tools() {
        let agent = apply_memory(test_agent(), AgentKind::Review, Some(&wiring(false)));
        let names = tool_names(&agent);
        assert!(!names.iter().any(|n| n.starts_with("memory_")));
    }

    #[test]
    fn none_wiring_is_a_noop() {
        let agent = apply_memory(test_agent(), AgentKind::Main, None);
        assert!(!tool_names(&agent).iter().any(|n| n.starts_with("memory_")));
    }
}
