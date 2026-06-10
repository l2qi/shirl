// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sweet_agent::{Agent, ExtensionRegistry};
use sweet_core::sandbox::Sandbox;
use sweet_core::{Model, Session};
use sweet_tools::HttpFetch;

use super::{
    handoff_to_main, mcp_capabilities, read_only_tools, with_web_search, SharedWebSearchBackend,
};

const REVIEW_PROMPT: &str = "You are Shirl's code review agent. \
    You do NOT fix code — you only have read-only tools.\n\
    \n\
    Use Grep and Glob to find related files, DirectoryTree for project context, \
    HeadFile and TailFile to check imports and file ends. HttpFetch and WebSearch \
    are available for looking up best practices or known issues.\n\
    \n\
    1. Read code, search for patterns, and present findings by severity: \
       [CRITICAL], [WARN], [NOTE]. Each finding: file path, line reference, \
       description, suggested fix.\n\
    2. Present your findings and wait for the user's response\n\
    3. If the user responds to findings, update the review\n\
    \n\
    Only when the user explicitly asks to fix, use transfer_to_main with the \
    complete list of items to fix verbatim. When the review is final, tell the \
    user to say /fix or \"fix these\" to apply fixes.";

pub fn build(
    model: Arc<dyn Model>,
    extensions: &ExtensionRegistry,
    web_search: Option<SharedWebSearchBackend>,
    session: Box<dyn Session>,
    mcp_specs: &[sweet_core::ToolSpec],
    sandbox: Arc<dyn Sandbox>,
) -> Agent<Arc<dyn Model>> {
    let fs = sandbox.fs();

    let tools = read_only_tools(fs).with_tool(HttpFetch::default());
    let tools = with_web_search(tools, web_search.clone());
    let mcp_tools = mcp_capabilities(mcp_specs);

    Agent::new_shared(model)
        .with_instructions(REVIEW_PROMPT)
        .with_session_boxed(session)
        .with_capability_provider(&tools)
        .with_capability_provider(&mcp_tools)
        .with_handoff(handoff_to_main())
        .with_extension_registry(extensions)
}
