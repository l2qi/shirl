// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Approval dialog loop — runs in the main event loop while the agent task
//! awaits the user's decision.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use shirl_tools::unified_diff;
use shirl_ui::{Command, SharedIo};
use sweet_core::permission::{ApprovalDecision, ApprovalPreview};
use sweet_core::ToolCall;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

/// What the user did with an approval prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome {
    /// The user answered the prompt; the decision was sent to the agent task.
    Answered,
    /// The user asked to cancel the whole turn (Esc / Ctrl+C). The caller
    /// should abort the in-flight turn.
    Cancelled,
}

/// Run the interactive approval dialog for a pending tool call.
///
/// Shows the approval prompt inline in the viewport (replacing the text input
/// area), drives viewport redraws so the working indicator keeps animating,
/// blocks on `cmd_rx` for the user's decision, clears the prompt, and returns
/// the outcome.
pub async fn run_approval_dialog(
    shared_io: &SharedIo,
    cmd_rx: &mut mpsc::Receiver<Command>,
    call: &ToolCall,
    risk: sweet_core::ToolRisk,
    response_tx: tokio::sync::oneshot::Sender<ApprovalDecision>,
) -> Result<ApprovalOutcome> {
    // The agent task that raised this request may already be gone — e.g. the
    // turn was cancelled while the request sat queued. Don't show a stale
    // prompt for a dead call.
    if response_tx.is_closed() {
        return Ok(ApprovalOutcome::Answered);
    }

    // Compute a rich diff/content preview *before* acquiring the lock so
    // file I/O doesn't block the input thread or the redraw tick.
    let preview = compute_preview(&call.name, &call.arguments).await;

    // Show the approval prompt inline in the viewport.
    {
        let mut io = shared_io.lock().await;
        // Preview rendering is cosmetic — errors are swallowed internally.
        io.flush_approval_preview(&preview);
        // Rendering errors are non-fatal: the approval dialog's hard
        // requirement is receiving the user's keystroke. A viewport redraw
        // failure must not drop response_tx and crash the program.
        let _ = io.set_approval(&call.name, risk, &call.arguments);
    }

    // Keep redrawing so the breathing working indicator stays animated while
    // the prompt is displayed.
    let mut redraw = tokio::time::interval(Duration::from_millis(150));
    redraw.set_missed_tick_behavior(MissedTickBehavior::Delay);

    // `Some(decision)` once the user answered; `None` to cancel the turn.
    let decision: Option<ApprovalDecision> = loop {
        tokio::select! {
            cmd = cmd_rx.recv() => match cmd {
                Some(Command::ApprovalKey('y' | 'Y')) => break Some(ApprovalDecision::Allow),
                Some(Command::ApprovalKey('a' | 'A')) => {
                    break Some(ApprovalDecision::AllowSession)
                }
                Some(Command::ApprovalKey('n' | 'N')) => break Some(ApprovalDecision::Deny),
                // Esc / Ctrl+C / EOF cancel the whole turn, not just this call.
                Some(Command::Cancel | Command::Exit) | None => break None,
                Some(Command::Submit(line)) => {
                    // A line submitted just before the prompt opened — stash it
                    // so the main loop processes it after the dialog rather
                    // than dropping it.
                    shared_io.lock().await.pending_command = Some(Command::Submit(line));
                }
                Some(
                    Command::CycleMode
                    | Command::Partial(_)
                    | Command::SelectMove(_)
                    | Command::Resize
                    | Command::ApprovalKey(_)
                    | Command::ToggleTranscript
                    | Command::FilePickerFilter(_)
                    | Command::FilePickerAccept
                    | Command::FilePickerClose,
                ) => {}
            },
            _ = redraw.tick() => {
                let mut io = shared_io.lock().await;
                let _ = io.draw();
            }
        }
    };

    // Clear the approval prompt and restore the normal input.
    {
        let mut io = shared_io.lock().await;
        // Non-fatal for the same reason as set_approval above.
        let _ = io.clear_approval();
    }

    match decision {
        Some(decision) => {
            // Ignore error: if the receiver was dropped the agent task was
            // cancelled — the caller handles that via the closed channel.
            let _ = response_tx.send(decision);
            Ok(ApprovalOutcome::Answered)
        }
        // Drop `response_tx` without sending; the caller aborts the turn.
        None => Ok(ApprovalOutcome::Cancelled),
    }
}

/// Compute a rich preview for the approval prompt based on tool name and
/// arguments. Only `edit_file` and `write_file` produce previews; everything
/// else returns `ApprovalPreview::None`.
async fn compute_preview(tool_name: &str, args: &serde_json::Value) -> ApprovalPreview {
    match tool_name {
        "edit_file" => compute_edit_preview(tool_name, args).await,
        "write_file" => compute_write_preview(tool_name, args).await,
        _ => ApprovalPreview::None,
    }
}

async fn compute_edit_preview(tool_name: &str, args: &serde_json::Value) -> ApprovalPreview {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => return ApprovalPreview::None,
    };

    let old_content = match tokio::fs::read_to_string(&path).await {
        Ok(c) => c,
        Err(_) => return ApprovalPreview::None,
    };

    // Compute the new content by applying the edits.
    let had_explicit_edits = args
        .get("edits")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let replace_all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        && !had_explicit_edits;

    let mut new_content = old_content.clone();

    if had_explicit_edits {
        let edits = match args.get("edits").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return ApprovalPreview::None,
        };
        for edit in edits {
            let old_text = match edit.get("old_text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return ApprovalPreview::None,
            };
            let new_text = match edit.get("new_text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return ApprovalPreview::None,
            };
            // EditFile rejects an explicit edit whose `old_text` does not
            // match exactly once. Mirror that so the preview is never shown
            // for a diff the tool will refuse to apply.
            if new_content.matches(old_text).count() != 1 {
                return ApprovalPreview::None;
            }
            new_content = new_content.replacen(old_text, new_text, 1);
        }
    } else {
        let old_string = args
            .get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new_string = args
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if replace_all {
            // EditFile rejects replace-all mode when nothing matches.
            if !new_content.contains(old_string) {
                return ApprovalPreview::None;
            }
            new_content = new_content.replace(old_string, new_string);
        } else {
            // Single-edit mode requires exactly one match, like the tool.
            if new_content.matches(old_string).count() != 1 {
                return ApprovalPreview::None;
            }
            new_content = new_content.replacen(old_string, new_string, 1);
        }
    }

    let diff = unified_diff(&old_content, &new_content, &path);
    ApprovalPreview::Diff {
        tool_name: tool_name.to_string(),
        path,
        diff,
    }
}

async fn compute_write_preview(tool_name: &str, args: &serde_json::Value) -> ApprovalPreview {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => return ApprovalPreview::None,
    };
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c.to_string(),
        None => return ApprovalPreview::None,
    };

    if Path::new(&path).exists() {
        // File already exists — show a diff.
        match tokio::fs::read_to_string(&path).await {
            Ok(old) => {
                let diff = unified_diff(&old, &content, &path);
                ApprovalPreview::Diff {
                    tool_name: tool_name.to_string(),
                    path,
                    diff,
                }
            }
            Err(_) => ApprovalPreview::NewFile { path, content },
        }
    } else {
        ApprovalPreview::NewFile { path, content }
    }
}
