// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Agent definitions, system prompts, and tool wiring for the Shirl coding
//! assistant.
//!
//! [`agents`] holds the three peer agents (main, plan, review) plus the
//! handoff and mode-switch machinery; [`subagents`] holds the leaf agents
//! (explore, diagnose, explain, testgen, web_research) that those agents
//! invoke as tools; [`headless`] holds the orchestrator agent and its
//! subagent workers for headless (`-p`) mode.

pub mod agents;
pub mod headless;
pub mod subagents;
