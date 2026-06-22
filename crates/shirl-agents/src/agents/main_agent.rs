// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sweet_agent::{Agent, ExtensionRegistry, ToolCapabilities};
use sweet_core::sandbox::Sandbox;
use sweet_core::{Model, Session};
use sweet_tools::{Calculator, HttpFetch};

use super::{
    handoff_to_plan, handoff_to_review, mcp_capabilities, with_web_search, SharedWebSearchBackend,
};
use crate::subagents::{
    diagnose::diagnose_spec, explain::explain_spec, explore::explore_spec, testgen::testgen_spec,
    web_research::web_research_spec,
};
use shirl_tools::{
    bash_tool, create_directory_tool, directory_size_tool, directory_tree_tool, edit_file_tool,
    get_file_info_tool, glob_tool, grep_tool, head_file_tool, list_directory_tool, move_file_tool,
    patch_tool, read_file_tool, tail_file_tool, write_file_tool,
};

const MAIN_PROMPT: &str = "You are Shirl, a coding assistant.\n\
    \n\
    ROUTING - evaluate this BEFORE any other action on every user turn:\n\
    \n\
    1. If the user uses \"review\" or \"audit\" as a verb directed at code, files, \
       changes, a diff, or a module - examples: \"review this\", \"review the \
       changes\", \"review the diff\", \"audit the auth module\" - your FIRST and \
       ONLY action this turn MUST be to call transfer_to_review. Do NOT read files, \
       run git, run grep, or gather any data first. Pass the user's request as the \
       `focus` argument and hand off.\n\
    2. If the user uses \"plan\" or \"design\" as a verb directed at a task - \
       examples: \"plan how to add X\", \"plan this refactor\", \"design this \
       change\" - your FIRST and ONLY action this turn MUST be to call \
       transfer_to_plan. Do NOT start implementing.\n\
    3. Verb vs. noun - stay in main when the user is referencing existing content: \
       \"here is the review feedback below\" (noun, don't switch), \"here is the \
       plan, please implement it\" (noun, don't switch).\n\
    4. If the task is clearly multi-file or non-trivial and you'd genuinely benefit \
       from upfront design before coding, you MAY call transfer_to_plan. First emit \
       one short sentence explaining why you're switching.\n\
    5. For everything else, stay in main mode and just do the work. Don't suggest \
       slash commands; the user knows about them.\n\
    \n\
    When working in main mode:\n\
    - Be concise and accurate\n\
    - Prefer short answers; show code only when it's the cleanest response\n\
    - Before making changes, understand the existing code by reading relevant files\n\
    - Make minimal, targeted edits rather than rewriting large sections\n\
    - Run relevant tests after making changes to verify correctness\n\
    - When a task is ambiguous, ask for clarification rather than guessing\n\
    \n\
    Staying on track with write_todos:\n\
    - When you receive a saved plan or review (you'll be told its file path), FIRST call \
      write_todos to record the items you'll act on - a subset is fine if the user asked \
      for only some items; honor exactly what they asked for. Then work the list top to \
      bottom, marking each in_progress as you start it and done when finished. Finish only \
      when every item is done.\n\
    - For direct requests with no plan/review, call write_todos only when the task is \
      genuinely multi-step (roughly three or more distinct actions, or spans multiple \
      files). For a one-line change, a quick edit, or a question, just do it - no todos.\n\
    - Your active plan/review and current todo list are re-shown to you every turn; trust \
      that list and re-read the source file when you need full detail. Do not drift onto \
      unrelated work while items remain.\n\
    \n\
    You have specialized subagents available as tools. Use them when they would produce \
    better results than doing the work directly:\n\
    - **explore**: investigate the codebase - find files, locate symbol definitions, trace \
      usages, and answer structural questions\n\
    - **web_research**: fetch URLs (and search the web, when configured) to look up \
      external documentation, APIs, or best practices\n\
    - **diagnose**: debug errors or failing tests by systematically investigating the codebase\n\
    - **explain**: explain what a specific code region does in plain language\n\
    - **testgen**: generate test cases for a specific function or module";

pub fn build(
    model: Arc<dyn Model>,
    extensions: &ExtensionRegistry,
    web_search: Option<SharedWebSearchBackend>,
    session: Box<dyn Session>,
    mcp_specs: &[sweet_core::ToolSpec],
    sandbox: Arc<dyn Sandbox>,
) -> Agent<Arc<dyn Model>> {
    let fs = sandbox.fs();
    let runner = sandbox.runner();

    let coding_tools = ToolCapabilities::new("coding")
        .with_tool(read_file_tool(fs.clone()))
        .with_tool(write_file_tool(fs.clone()))
        .with_tool(edit_file_tool(fs.clone()))
        .with_tool(glob_tool(fs.clone()))
        .with_tool(grep_tool(fs.clone()))
        .with_tool(bash_tool(runner.clone()))
        .with_tool(patch_tool(fs.clone()))
        .with_tool(move_file_tool(fs.clone()))
        .with_tool(create_directory_tool(fs.clone()))
        .with_tool(directory_tree_tool(fs.clone()))
        .with_tool(get_file_info_tool(fs.clone()))
        .with_tool(directory_size_tool(fs.clone()))
        .with_tool(head_file_tool(fs.clone()))
        .with_tool(tail_file_tool(fs.clone()))
        .with_tool(list_directory_tool(fs.clone()))
        .with_tool(Calculator::default())
        .with_tool(HttpFetch::default());
    let coding_tools = with_web_search(coding_tools, web_search.clone());
    let mcp_tools = mcp_capabilities(mcp_specs);

    Agent::new_shared(model)
        .with_instructions(MAIN_PROMPT)
        .with_session_boxed(session)
        .with_capability_provider(&coding_tools)
        .with_capability_provider(&mcp_tools)
        .with_subagent(explore_spec(sandbox.clone()))
        .with_subagent(web_research_spec(web_search))
        .with_subagent(diagnose_spec(sandbox.clone()))
        .with_subagent(explain_spec(sandbox.clone()))
        .with_subagent(testgen_spec(sandbox.clone()))
        .with_handoff(handoff_to_plan())
        .with_handoff(handoff_to_review())
        .with_extension_registry(extensions)
}
