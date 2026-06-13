// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

mod main_agent;
mod plan_agent;
mod review_agent;

use std::sync::Arc;

use sweet_agent::handoff::{HandoffContext, HandoffHandler, HandoffResult, HandoffSpec};
use sweet_agent::{Agent, ExtensionRegistry, ToolCapabilities};
use sweet_core::sandbox::Sandbox;
use sweet_core::tool::ToolError;
use sweet_core::{Model, Session};
use sweet_tools::{WebSearch, WebSearchBackend};

use shirl_tools::{
    directory_size_tool, directory_tree_tool, get_file_info_tool, glob_tool, grep_tool,
    head_file_tool, list_directory_tool, read_file_tool, tail_file_tool,
};

/// Shared handle to a web-search backend. Cloning yields another reference
/// to the same backend instance, so the parent agent's `WebSearch` tool and
/// the `web_research` subagent can both use it without duplication.
pub type SharedWebSearchBackend = Arc<dyn WebSearchBackend>;

/// The three peer agents users can interact with through the REPL.
///
/// The headless orchestrator is intentionally not a member: it is constructed
/// only by [`crate::headless::build_orchestrator`], never reached by user
/// mode switches, and treated as `Main` for model/web-search resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgentKind {
    Main,
    Plan,
    Review,
}

impl AgentKind {
    pub fn from_target(target: &str) -> Option<Self> {
        match target {
            "main" => Some(Self::Main),
            "plan" => Some(Self::Plan),
            "review" => Some(Self::Review),
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Plan => "plan",
            Self::Review => "review",
        }
    }
}

pub struct ModeSwitch {
    pub target: AgentKind,
    pub step_with: Option<String>,
}

pub enum ModeCommand {
    Switch(ModeSwitch),
    Invalid(String),
    NotModeCommand,
}

pub fn resolve_mode_command(name: &str, args: &str, current: AgentKind) -> ModeCommand {
    match name {
        "plan" => {
            if current != AgentKind::Main {
                return ModeCommand::Invalid(format!(
                    "/plan is only available in main mode (currently in {} mode)",
                    current.display_name()
                ));
            }
            ModeCommand::Switch(ModeSwitch {
                target: AgentKind::Plan,
                step_with: if args.trim().is_empty() {
                    None
                } else {
                    Some(args.to_string())
                },
            })
        }
        "review" => {
            if current != AgentKind::Main {
                return ModeCommand::Invalid(format!(
                    "/review is only available in main mode (currently in {} mode)",
                    current.display_name()
                ));
            }
            ModeCommand::Switch(ModeSwitch {
                target: AgentKind::Review,
                step_with: if args.trim().is_empty() {
                    None
                } else {
                    Some(args.to_string())
                },
            })
        }
        "approve" => {
            if current != AgentKind::Plan {
                return ModeCommand::Invalid(format!(
                    "/approve is only available in plan mode (currently in {} mode)",
                    current.display_name()
                ));
            }
            ModeCommand::Switch(ModeSwitch {
                target: AgentKind::Main,
                step_with: Some("The user approved the plan above. Implement it.".to_string()),
            })
        }
        "fix" => {
            if current != AgentKind::Review {
                return ModeCommand::Invalid(format!(
                    "/fix is only available in review mode (currently in {} mode)",
                    current.display_name()
                ));
            }
            let prompt = if args.trim().is_empty() {
                "Address all review items.".to_string()
            } else {
                format!("Address these review items: {}", args)
            };
            ModeCommand::Switch(ModeSwitch {
                target: AgentKind::Main,
                step_with: Some(prompt),
            })
        }
        "back" => {
            if current == AgentKind::Main {
                return ModeCommand::Invalid(
                    "/back is only available in plan or review mode".to_string(),
                );
            }
            ModeCommand::Switch(ModeSwitch {
                target: AgentKind::Main,
                step_with: None,
            })
        }
        _ => ModeCommand::NotModeCommand,
    }
}

/// Shared read-only tool set used by plan, review, orchestrator, and explore.
///
/// Eliminates the four-way duplication of the same tool list by
/// centralising the read-only baseline. Callers add their own extras
/// (e.g. `HttpFetch`, `WebSearch`) on top.
pub(crate) fn read_only_tools(
    fs: std::sync::Arc<dyn sweet_core::sandbox::Filesystem>,
) -> ToolCapabilities {
    ToolCapabilities::new("read-only")
        .with_tool(read_file_tool(fs.clone()))
        .with_tool(glob_tool(fs.clone()))
        .with_tool(grep_tool(fs.clone()))
        .with_tool(directory_tree_tool(fs.clone()))
        .with_tool(list_directory_tool(fs.clone()))
        .with_tool(get_file_info_tool(fs.clone()))
        .with_tool(directory_size_tool(fs.clone()))
        .with_tool(head_file_tool(fs.clone()))
        .with_tool(tail_file_tool(fs))
}

// Construction wiring: every argument is a distinct dependency the binary
// resolves differently; bundling them into a struct would only move the list.
#[allow(clippy::too_many_arguments)]
pub fn build_agent(
    kind: AgentKind,
    model: Arc<dyn Model>,
    extensions: &ExtensionRegistry,
    web_search: Option<SharedWebSearchBackend>,
    session: Box<dyn Session>,
    mcp_specs: &[sweet_core::ToolSpec],
    sandbox: Arc<dyn Sandbox>,
    memory: Option<&crate::MemoryWiring>,
) -> Agent<Arc<dyn Model>> {
    let agent = match kind {
        AgentKind::Main => {
            main_agent::build(model, extensions, web_search, session, mcp_specs, sandbox)
        }
        AgentKind::Plan => {
            plan_agent::build(model, extensions, web_search, session, mcp_specs, sandbox)
        }
        AgentKind::Review => {
            review_agent::build(model, extensions, web_search, session, mcp_specs, sandbox)
        }
    };
    crate::memory::apply_memory(agent, kind, memory)
}

/// Bundle the dynamically discovered MCP tool specs into a capability provider.
/// Shared by every agent builder so MCP tools enter through one code path.
pub(crate) fn mcp_capabilities(mcp_specs: &[sweet_core::ToolSpec]) -> ToolCapabilities {
    ToolCapabilities::new("mcp").with_tools(mcp_specs.iter().cloned())
}

/// Append the optional web-search tool to a tool bundle, leaving it untouched
/// when no backend is configured for the agent. Takes the backend by shared
/// reference so the same instance can also be handed to the `web_research`
/// subagent.
fn with_web_search(
    tools: ToolCapabilities,
    backend: Option<SharedWebSearchBackend>,
) -> ToolCapabilities {
    match backend {
        Some(b) => tools.with_tool(WebSearch::new(b)),
        None => tools,
    }
}

struct PassthroughHandoff {
    target: &'static str,
    field: &'static str,
}

#[sweet_core::async_trait]
impl HandoffHandler for PassthroughHandoff {
    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: HandoffContext,
    ) -> Result<HandoffResult, ToolError> {
        let payload = args
            .get(self.field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(HandoffResult::Transfer {
            target: self.target.to_string(),
            payload: if payload.is_empty() {
                None
            } else {
                Some(payload)
            },
        })
    }
}

pub(crate) fn handoff_to_plan() -> HandoffSpec {
    HandoffSpec::new(
        "transfer_to_plan",
        "Hand off to the planning agent. MUST be your FIRST action (before reading files or running any other tool) when the user uses 'plan' or 'design' as a verb directed at a task — e.g., 'plan how to add X', 'plan this refactor', 'design this change'. May also be called when a task is clearly multi-file and would benefit from upfront design (announce the switch in one sentence first). Do NOT use when the user references an existing plan as a noun ('here is the plan, please implement it'), or for tasks you can implement directly.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the task to plan"
                }
            },
            "required": ["task"]
        }),
        PassthroughHandoff {
            target: "plan",
            field: "task",
        },
    )
}

pub(crate) fn handoff_to_review() -> HandoffSpec {
    HandoffSpec::new(
        "transfer_to_review",
        "Hand off to the code review agent. MUST be your FIRST action (before reading files, running git, or invoking any other tool) when the user uses 'review' or 'audit' as a verb directed at code, files, changes, a diff, or a module — e.g., 'review this', 'review the changes', 'review the diff', 'audit the auth module'. Pass the user's request as the `focus` argument. Do NOT use when the user references existing review content as a noun ('look at the review feedback below'), or for ad-hoc inspection during normal coding.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What to focus the review on (optional)"
                }
            }
        }),
        PassthroughHandoff {
            target: "review",
            field: "focus",
        },
    )
}

pub(crate) fn handoff_to_main() -> HandoffSpec {
    HandoffSpec::new(
        "transfer_to_main",
        "Return to the main coding agent. Call when the user explicitly approves your plan \
        or asks to fix review findings. Content: the complete plan or review items, verbatim.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The complete approved plan or complete review items to fix, verbatim. Any surrounding discussion context can be summarized briefly."
                }
            },
            "required": ["content"]
        }),
        PassthroughHandoff {
            target: "main",
            field: "content",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn switch(cmd: ModeCommand) -> ModeSwitch {
        match cmd {
            ModeCommand::Switch(s) => s,
            ModeCommand::Invalid(msg) => panic!("expected Switch, got Invalid({msg})"),
            ModeCommand::NotModeCommand => panic!("expected Switch, got NotModeCommand"),
        }
    }

    fn invalid(cmd: ModeCommand) -> String {
        match cmd {
            ModeCommand::Invalid(msg) => msg,
            ModeCommand::Switch(_) => panic!("expected Invalid, got Switch"),
            ModeCommand::NotModeCommand => panic!("expected Invalid, got NotModeCommand"),
        }
    }

    #[test]
    fn plan_from_main_with_args_targets_plan_and_passes_args() {
        let s = switch(resolve_mode_command(
            "plan",
            "build feature X",
            AgentKind::Main,
        ));
        assert_eq!(s.target, AgentKind::Plan);
        assert_eq!(s.step_with.as_deref(), Some("build feature X"));
    }

    #[test]
    fn plan_from_main_with_empty_args_just_switches() {
        let s = switch(resolve_mode_command("plan", "   ", AgentKind::Main));
        assert_eq!(s.target, AgentKind::Plan);
        assert!(s.step_with.is_none());
    }

    #[test]
    fn plan_from_non_main_is_invalid() {
        for kind in [AgentKind::Plan, AgentKind::Review] {
            let msg = invalid(resolve_mode_command("plan", "x", kind));
            assert!(msg.contains("main mode"), "{msg}");
        }
    }

    #[test]
    fn review_from_main_targets_review() {
        let s = switch(resolve_mode_command(
            "review",
            "check auth",
            AgentKind::Main,
        ));
        assert_eq!(s.target, AgentKind::Review);
        assert_eq!(s.step_with.as_deref(), Some("check auth"));
    }

    #[test]
    fn review_from_non_main_is_invalid() {
        let msg = invalid(resolve_mode_command("review", "", AgentKind::Plan));
        assert!(msg.contains("main mode"));
    }

    #[test]
    fn approve_from_plan_targets_main_with_implement_step() {
        let s = switch(resolve_mode_command("approve", "", AgentKind::Plan));
        assert_eq!(s.target, AgentKind::Main);
        assert!(s.step_with.unwrap().contains("approved"));
    }

    #[test]
    fn approve_from_non_plan_is_invalid() {
        for kind in [AgentKind::Main, AgentKind::Review] {
            let msg = invalid(resolve_mode_command("approve", "", kind));
            assert!(msg.contains("plan mode"), "{msg}");
        }
    }

    #[test]
    fn fix_from_review_targets_main_with_review_items() {
        let s = switch(resolve_mode_command(
            "fix",
            "item 1, item 2",
            AgentKind::Review,
        ));
        assert_eq!(s.target, AgentKind::Main);
        let step = s.step_with.unwrap();
        assert!(step.contains("item 1, item 2"), "{step}");
    }

    #[test]
    fn fix_from_review_with_empty_args_uses_default() {
        let s = switch(resolve_mode_command("fix", "", AgentKind::Review));
        assert_eq!(s.target, AgentKind::Main);
        assert!(s.step_with.unwrap().contains("review"));
    }

    #[test]
    fn fix_from_non_review_is_invalid() {
        for kind in [AgentKind::Main, AgentKind::Plan] {
            let msg = invalid(resolve_mode_command("fix", "", kind));
            assert!(msg.contains("review mode"), "{msg}");
        }
    }

    #[test]
    fn back_from_plan_or_review_targets_main_without_step() {
        for kind in [AgentKind::Plan, AgentKind::Review] {
            let s = switch(resolve_mode_command("back", "", kind));
            assert_eq!(s.target, AgentKind::Main);
            assert!(s.step_with.is_none());
        }
    }

    #[test]
    fn back_from_main_is_invalid() {
        let msg = invalid(resolve_mode_command("back", "", AgentKind::Main));
        assert!(msg.contains("plan or review"));
    }

    #[test]
    fn unknown_name_is_not_mode_command() {
        assert!(matches!(
            resolve_mode_command("clear", "", AgentKind::Main),
            ModeCommand::NotModeCommand
        ));
    }
}
