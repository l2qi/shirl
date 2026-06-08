// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test for Sweet's model-only runloop: build an `Agent` with a
//! system prompt, drive it through `sweet_agent::run` against an in-memory
//! `AgentIo`, and verify multi-turn replies round-trip.
//!
//! The real `CrosstermIo` is not exercised here because it requires a TTY.
//! The runloop itself is covered by `sweet-agent`'s tests.

use sweet_agent::test_util::{MockModel, VecIo};
use sweet_agent::{run, Agent};
use sweet_core::{Message, Role};

#[tokio::test]
async fn multi_turn_session_round_trips() {
    let model = MockModel::with_replies(["pong", "pong again"]);
    let mut agent = Agent::new(model).with_instructions("be terse");
    let mut io = VecIo::with_inputs(["ping", "ping again"]);

    run(&mut agent, &mut io).await.unwrap();

    let outputs = io.outputs();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0], Message::assistant("pong"));
    assert_eq!(outputs[1], Message::assistant("pong again"));

    let messages = agent.session().messages();
    // 2 user + 2 assistant (system is kept separate from session)
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, Role::User);
}
