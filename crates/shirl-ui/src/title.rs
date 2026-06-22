// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Session title generation and text sanitization.

use std::sync::Arc;

use sweet_core::Model;

/// Ask the model for a short session title given the conversation so far.
/// Returns `Ok(Some(title))` on success, `Ok(None)` when the conversation is
/// too short to title or the model reply sanitizes to an empty string.
///
/// Pure compute - no UI state is touched, so callers can spawn this without
/// holding the IO mutex across the model round-trip.
pub async fn compute_title(
    model: &Arc<dyn Model>,
    items: &[sweet_core::MemoryItem],
) -> sweet_core::Result<Option<String>> {
    if items.len() < 2 {
        return Ok(None);
    }
    let mut context = String::new();
    for item in items.iter().take(8) {
        let sweet_core::MemoryItem::Message(msg) = item;
        if msg.role == sweet_core::Role::Tool {
            continue;
        }
        let preview: String = msg.text_content().chars().take(200).collect();
        context.push_str(&format!("{}: {}\n", msg.role.as_str(), preview));
    }
    let prompt = format!(
        "Generate a short (3-6 word) title for this coding session based on the \
         conversation below. Return ONLY the title text, nothing else. \
         No quotes, no punctuation at the end.\n\n{}",
        context
    );
    let reply = model
        .complete(&[sweet_core::Message::user(prompt)], &[])
        .await?;
    let title = sanitize_title(&reply.text_content());
    Ok(if title.is_empty() { None } else { Some(title) })
}

/// Clean up a model-generated title: trim, strip wrapping quotes, drop
/// trailing punctuation, collapse to the first line. Models routinely return
/// `"Fixed bug."` or multi-line responses despite the prompt asking otherwise.
pub(crate) fn sanitize_title(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or("").trim();
    let unquoted = first_line
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| {
            first_line
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
        })
        .unwrap_or(first_line);
    unquoted
        .trim_end_matches(['.', '!', '?', ',', ';', ':'])
        .trim()
        .to_string()
}

/// Detect the current git branch name for the given working directory.
/// Returns `None` if `git` is unavailable, the directory is not a repo,
/// or HEAD is detached.
///
/// Shells out to `git` synchronously - call only at startup or in response
/// to explicit user action, never in a hot loop.
pub(crate) fn detect_git_branch(cwd: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    // "HEAD" means detached HEAD - nothing useful to display.
    if branch.is_empty() || branch == "HEAD" {
        return None;
    }
    Some(branch)
}

/// Collapse the cwd, replacing the home directory prefix with `~`.
pub(crate) fn short_cwd(cwd: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy().into_owned();
        if let Some(rest) = cwd.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    cwd.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_title_strips_common_model_decorations() {
        // Plain title passes through.
        assert_eq!(sanitize_title("Fix login bug"), "Fix login bug");
        // Surrounding quotes stripped.
        assert_eq!(sanitize_title("\"Fix login bug\""), "Fix login bug");
        assert_eq!(sanitize_title("'Fix login bug'"), "Fix login bug");
        // Trailing punctuation stripped.
        assert_eq!(sanitize_title("Fix login bug."), "Fix login bug");
        assert_eq!(sanitize_title("Fix login bug!"), "Fix login bug");
        // Combination: quotes + trailing punct.
        assert_eq!(sanitize_title("\"Fix login bug.\""), "Fix login bug");
        // Whitespace trimmed.
        assert_eq!(sanitize_title("  Fix login bug  "), "Fix login bug");
        // Multi-line: only first line kept.
        assert_eq!(
            sanitize_title("Fix login bug\nExtra explanation"),
            "Fix login bug"
        );
        // Empty input stays empty.
        assert_eq!(sanitize_title(""), "");
        assert_eq!(sanitize_title("   "), "");
    }
}
