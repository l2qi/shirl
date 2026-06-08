// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sweet_agent::{Agent, ExtensionRegistry, ToolCapabilities};
use sweet_core::sandbox::Sandbox;
use sweet_core::tool::ToolSpec;
use sweet_core::SharedSession;
use sweet_core::{Model, Session};

use crate::agents::{mcp_capabilities, SharedWebSearchBackend};
use crate::headless::{implement_sub, plan_sub, review_sub, Tracking, WorkerDeps};

use shirl_tools::{
    directory_size_tool, directory_tree_tool, get_file_info_tool, glob_tool, grep_tool,
    head_file_tool, list_directory_tool, read_file_tool, tail_file_tool,
};

pub const ORCHESTRATOR_PROMPT: &str =
    "You are Shirl, a coding assistant. In headless mode you orchestrate \
    three specialist workers to complete the user's task without a human \
    present.\n\
    \n\
    Your workers are tools you can call:\n\
    \n\
    - plan(task): Produces a numbered implementation plan. The task \
      parameter describes what needs planning — be specific about files, \
      requirements, and constraints.\n\
    - implement(instructions): Writes code, edits files, and runs commands. \
      The instructions parameter tells the worker what to do.\n\
    - review(focus): Reviews code for correctness and quality. The focus \
      parameter says what to review, e.g. \"Review changes to bar.rs for \
      correctness\" or \"Check for edge cases in the error handling.\"\n\
    \n\
    Each worker sees a snapshot of your conversation up to the moment you \
    call it, including every prior worker's final report (your tool-result \
    messages). What it does not see is a worker's internal scratchpad — \
    only the report. So when you call implement, you can rely on it having \
    the plan's full text; you do not need to re-quote it. Do include any \
    new constraints or clarifications in the instructions parameter.\n\
    \n\
    You decide which workers to call, in what order, and whether to iterate:\n\
    \n\
    - Default workflow: plan → implement → review → implement-to-fix → done.\n\
    - Skip plan for trivial changes (typo fix, one-liner) or pure questions.\n\
    - Skip review when no files were changed.\n\
    - Loop review → implement at most twice if review finds substantive \
      issues. Stop earlier if the next round would be churn.\n\
    - Call plan again if implement reveals the plan was wrong.\n\
    \n\
    When done, send a final message summarising what was done, any \
    remaining caveats, and the files touched. Be terse — this output goes \
    to a script or pipe.";

pub(crate) fn build(
    model: Arc<dyn Model>,
    extensions: Arc<ExtensionRegistry>,
    web_search: Option<SharedWebSearchBackend>,
    session: Box<dyn Session>,
    mcp_specs: &[ToolSpec],
    sandbox: Arc<dyn Sandbox>,
    tracking: Tracking,
) -> Agent<Arc<dyn Model>> {
    let fs = sandbox.fs();

    let read_only_tools = ToolCapabilities::new("orchestrator")
        .with_tool(read_file_tool(fs.clone()))
        .with_tool(glob_tool(fs.clone()))
        .with_tool(grep_tool(fs.clone()))
        .with_tool(directory_tree_tool(fs.clone()))
        .with_tool(list_directory_tool(fs.clone()))
        .with_tool(get_file_info_tool(fs.clone()))
        .with_tool(directory_size_tool(fs.clone()))
        .with_tool(head_file_tool(fs.clone()))
        .with_tool(tail_file_tool(fs.clone()));
    let mcp_tools = mcp_capabilities(mcp_specs);
    // One Arc'd vec shared by all three worker specs — avoids three full
    // `Vec<ToolSpec>` clones each time the orchestrator is built.
    let shared_mcp_specs = Arc::new(mcp_specs.to_vec());

    // Wrap the orchestrator's session so every append mirrors into a shared
    // handle. Each subagent handler holds a clone and snapshots the
    // orchestrator's transcript on invocation.
    let (shared_session, session_handle) = SharedSession::new(session);

    // The dependencies shared by all three workers. Cloned once per worker.
    let deps = WorkerDeps {
        model: model.clone(),
        sandbox,
        extensions: extensions.clone(),
        web_search,
        mcp_specs: shared_mcp_specs,
        parent_session: session_handle,
    };

    Agent::new_shared(model)
        .with_instructions(ORCHESTRATOR_PROMPT)
        .with_session_boxed(Box::new(shared_session))
        .with_capability_provider(&read_only_tools)
        .with_capability_provider(&mcp_tools)
        .with_subagent(plan_sub::plan_subagent_spec(
            deps.clone(),
            tracking.store.clone(),
        ))
        .with_subagent(implement_sub::implement_subagent_spec(
            deps.clone(),
            tracking.todos_tool.clone(),
            tracking.reminder.clone(),
        ))
        .with_subagent(review_sub::review_subagent_spec(deps, tracking.store))
        .with_extension_registry(&extensions)
}
