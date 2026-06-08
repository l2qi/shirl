// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sweet_agent::{Agent, ExtensionRegistry, ToolCapabilities};
use sweet_core::sandbox::Sandbox;
use sweet_core::{Model, Session};
use sweet_tools::HttpFetch;

use super::{handoff_to_main, mcp_capabilities, with_web_search, SharedWebSearchBackend};
use crate::subagents::{explore::explore_spec, web_research::web_research_spec};
use shirl_tools::{
    directory_size_tool, directory_tree_tool, get_file_info_tool, glob_tool, grep_tool,
    head_file_tool, list_directory_tool, read_file_tool, tail_file_tool,
};

const PLAN_PROMPT: &str =
    "You are Shirl's planning agent. You produce numbered implementation plans. \
    You do NOT implement — you only have read-only tools.\n\
    \n\
    Use explore for codebase lookup, Glob and DirectoryTree for project structure, \
    HeadFile and TailFile to preview files, Grep to search for symbols or patterns. \
    HttpFetch and WebSearch are available for documentation and research.\n\
    \n\
    1. Investigate the codebase and produce a numbered plan: file paths, what to \
       change, [WARN] risks, assumptions\n\
    2. Present the plan and wait for the user's feedback\n\
    3. If the user amends the plan, update it and present again\n\
    \n\
    Only when the user explicitly approves the plan, use transfer_to_main with the \
    complete approved plan verbatim. When the plan is final, tell the user to say \
    /approve or \"go ahead\" to implement.";

pub fn build(
    model: Arc<dyn Model>,
    extensions: &ExtensionRegistry,
    web_search: Option<SharedWebSearchBackend>,
    session: Box<dyn Session>,
    mcp_specs: &[sweet_core::ToolSpec],
    sandbox: Arc<dyn Sandbox>,
) -> Agent<Arc<dyn Model>> {
    let fs = sandbox.fs();

    let tools = ToolCapabilities::new("plan")
        .with_tool(read_file_tool(fs.clone()))
        .with_tool(glob_tool(fs.clone()))
        .with_tool(grep_tool(fs.clone()))
        .with_tool(directory_tree_tool(fs.clone()))
        .with_tool(list_directory_tool(fs.clone()))
        .with_tool(get_file_info_tool(fs.clone()))
        .with_tool(directory_size_tool(fs.clone()))
        .with_tool(head_file_tool(fs.clone()))
        .with_tool(tail_file_tool(fs.clone()))
        .with_tool(HttpFetch::default());
    let tools = with_web_search(tools, web_search.clone());
    let mcp_tools = mcp_capabilities(mcp_specs);

    Agent::new_shared(model)
        .with_instructions(PLAN_PROMPT)
        .with_session_boxed(session)
        .with_capability_provider(&tools)
        .with_capability_provider(&mcp_tools)
        .with_subagent(explore_spec(sandbox.clone()))
        .with_subagent(web_research_spec(web_search))
        .with_handoff(handoff_to_main())
        .with_extension_registry(extensions)
}
