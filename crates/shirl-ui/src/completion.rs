// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Slash-command completion and discovery hints.
//!
//! Provides ghost-text completion (suffix shown in dim gray after the cursor)
//! and a discovery hint line (lists matching commands when multiple matches
//! exist).

#[derive(Clone)]
pub struct CommandInfo {
    pub name: String,
    pub description: String,
}

/// Returns the built-in slash commands known at compile time.
pub fn built_in_commands() -> Vec<CommandInfo> {
    vec![
        CommandInfo {
            name: "approve".into(),
            description: "approve plan, return to main".into(),
        },
        CommandInfo {
            name: "back".into(),
            description: "return to main mode".into(),
        },
        CommandInfo {
            name: "capabilities".into(),
            description: "list the active agent's tools and commands".into(),
        },
        CommandInfo {
            name: "clear".into(),
            description: "clear session messages".into(),
        },
        CommandInfo {
            name: "compact".into(),
            description: "compact session history".into(),
        },
        CommandInfo {
            name: "fix".into(),
            description: "apply review fixes".into(),
        },
        CommandInfo {
            name: "model".into(),
            description: "show or switch model".into(),
        },
        CommandInfo {
            name: "help".into(),
            description: "show help".into(),
        },
        CommandInfo {
            name: "new".into(),
            description: "start new session".into(),
        },
        CommandInfo {
            name: "plan".into(),
            description: "switch to plan mode".into(),
        },
        CommandInfo {
            name: "provider".into(),
            description: "manage providers".into(),
        },
        CommandInfo {
            name: "review".into(),
            description: "switch to review mode".into(),
        },
    ]
}

/// Returns the slash-command portion of `input` and the known commands whose
/// name starts with it.
///
/// Returns `None` when `input` is not a bare slash command — no leading `/`,
/// or a space already separates the command from its arguments.
fn matching<'i, 'c>(
    input: &'i str,
    commands: &'c [CommandInfo],
) -> Option<(&'i str, Vec<&'c CommandInfo>)> {
    let cmd_part = input.strip_prefix('/')?;
    if cmd_part.contains(' ') {
        return None;
    }
    let matches = commands
        .iter()
        .filter(|c| c.name.starts_with(cmd_part))
        .collect();
    Some((cmd_part, matches))
}

/// Returns the completion suffix for a slash command input.
///
/// Only activates when `input` starts with `/` and the command portion
/// (before any space) is a prefix of one or more known commands.
/// Returns `None` when the command is already complete or there are no matches.
pub fn complete(input: &str, commands: &[CommandInfo]) -> Option<String> {
    let (cmd_part, matches) = matching(input, commands)?;
    if matches.is_empty() {
        return None;
    }
    let names: Vec<&str> = matches.iter().map(|c| c.name.as_str()).collect();
    let lcp = longest_common_prefix(&names);
    let suffix: String = lcp.chars().skip(cmd_part.chars().count()).collect();
    if suffix.is_empty() {
        None
    } else {
        Some(suffix)
    }
}

/// Returns a discovery hint listing matching commands and their descriptions
/// when there are multiple candidates. Returns `None` for zero or one match.
pub fn hint(input: &str, commands: &[CommandInfo]) -> Option<String> {
    let (_, matches) = matching(input, commands)?;
    if matches.len() <= 1 {
        return None;
    }
    Some(
        matches
            .iter()
            .map(|c| format!("{} — {}", c.name, c.description))
            .collect::<Vec<_>>()
            .join("  ·  "),
    )
}

fn longest_common_prefix(names: &[&str]) -> String {
    let Some((first, rest)) = names.split_first() else {
        return String::new();
    };
    let mut len = first.chars().count();
    for name in rest {
        let common = first
            .chars()
            .zip(name.chars())
            .take_while(|(a, b)| a == b)
            .count();
        len = len.min(common);
    }
    first.chars().take(len).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmds() -> Vec<CommandInfo> {
        built_in_commands()
    }

    #[test]
    fn complete_unique_match() {
        let suffix = complete("/provi", &cmds());
        assert_eq!(suffix, Some("der".to_string()));
    }

    #[test]
    fn complete_exact_match_returns_none() {
        assert_eq!(complete("/model", &cmds()), None);
    }

    #[test]
    fn complete_multiple_matches_returns_common_prefix_suffix() {
        let custom = vec![
            CommandInfo {
                name: "compact".into(),
                description: String::new(),
            },
            CommandInfo {
                name: "compile".into(),
                description: String::new(),
            },
        ];
        let suffix = complete("/co", &custom);
        assert_eq!(suffix, Some("mp".to_string()));
    }

    #[test]
    fn complete_no_match() {
        assert_eq!(complete("/xyz", &cmds()), None);
    }

    #[test]
    fn complete_non_slash_input() {
        assert_eq!(complete("hello", &cmds()), None);
    }

    #[test]
    fn complete_after_space_returns_none() {
        assert_eq!(complete("/model ", &cmds()), None);
    }

    #[test]
    fn complete_bare_slash() {
        let suffix = complete("/", &cmds());
        assert_eq!(suffix, None);
    }

    #[test]
    fn hint_multiple_matches() {
        let h = hint("/p", &cmds());
        assert_eq!(
            h,
            Some("plan — switch to plan mode  ·  provider — manage providers".to_string())
        );
    }

    #[test]
    fn hint_single_match_returns_none() {
        assert_eq!(hint("/model", &cmds()), None);
    }

    #[test]
    fn hint_no_match_returns_none() {
        assert_eq!(hint("/xyz", &cmds()), None);
    }

    #[test]
    fn hint_bare_slash_shows_all() {
        let h = hint("/", &cmds());
        assert!(h.is_some());
        let h = h.unwrap();
        assert!(h.contains("model"));
        assert!(h.contains("provider"));
        assert!(h.contains("plan"));
    }

    #[test]
    fn longest_common_prefix_empty() {
        assert_eq!(longest_common_prefix(&[]), "");
    }

    #[test]
    fn longest_common_prefix_single() {
        assert_eq!(longest_common_prefix(&["abc"]), "abc");
    }

    #[test]
    fn longest_common_prefix_divergent() {
        assert_eq!(longest_common_prefix(&["abc", "abd"]), "ab");
    }
}
