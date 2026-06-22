// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

use anyhow::{Context, Result};
use sweet_core::sandbox::SandboxPolicy;
use sweet_core::SessionId;
use tracing_subscriber::filter::EnvFilter;

/// Slash-command names that a custom command must not shadow: the built-in
/// command capabilities, the hardcoded CLI branches, and the mode commands.
/// `CustomCommandsProvider::load` skips files whose stem matches one of these.
pub(crate) const RESERVED_COMMANDS: &[&str] = &[
    "new",
    "clear",
    "compact",
    "model",
    "provider",
    "reasoning",
    "capabilities",
    "memory",
    "help",
    "plan",
    "review",
    "approve",
    "fix",
    "back",
];

const DEFAULT_OBSERVABILITY_FILTER: &str =
    "sweet_core=debug,sweet_agent=debug,sweet_tools=debug,sweet_llm=debug,shirl_cli=debug";

/// Parsed CLI arguments.
pub(crate) struct CliArgs {
    pub headless: bool,
    pub prompt: Option<String>,
    pub json: bool,
    pub diff: bool,
    pub resume: Option<SessionId>,
    pub permission_mode: sweet_core::PermissionMode,
    pub sandbox_policy: SandboxPolicy,
}

pub(crate) fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        default(info);
    }));
}

pub(crate) fn parse_args() -> Result<CliArgs> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let ParsedArgs {
        headless,
        mut prompt,
        json,
        diff,
        mut resume,
        continue_recent,
        permission_mode,
        sandbox_policy,
    } = parse_argv(argv)?;

    // --continue: resolve the most recent session
    if continue_recent {
        let sessions_dir = shirl_core::config_home()?.join("sessions");
        if sessions_dir.exists() {
            let mut entries: Vec<_> = std::fs::read_dir(&sessions_dir)
                .with_context(|| format!("reading {}", sessions_dir.display()))?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let id = SessionId::from_str(&name).ok()?;
                    let meta = e.metadata().ok()?;
                    Some((id, meta.modified().ok()?))
                })
                .collect();
            entries.sort_by_key(|(_, mtime)| std::cmp::Reverse(*mtime));
            resume = entries.into_iter().next().map(|(id, _)| id);
        }
        if resume.is_none() {
            anyhow::bail!("no sessions found in {}", sessions_dir.display());
        }
    }

    // Headless stdin: if -p was given without a prompt, read from stdin.
    // If stdin is piped and a prompt was given, append stdin to the prompt.
    if headless && prompt.is_none() {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            let stdin_content =
                std::io::read_to_string(std::io::stdin()).context("reading prompt from stdin")?;
            if !stdin_content.trim().is_empty() {
                prompt = Some(stdin_content);
            }
        }
    } else if headless && prompt.is_some() {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            let stdin_content =
                std::io::read_to_string(std::io::stdin()).context("reading stdin")?;
            if !stdin_content.trim().is_empty() {
                if let Some(p) = prompt.take() {
                    prompt = Some(format!("{p}\n\n{}", stdin_content.trim()));
                }
            }
        }
    }

    Ok(CliArgs {
        headless,
        prompt,
        json,
        diff,
        resume,
        permission_mode,
        sandbox_policy,
    })
}

pub(crate) fn init_observability(
    session_id: &SessionId,
) -> Result<Option<tracing_appender::non_blocking::WorkerGuard>> {
    if std::env::var("SHIRL_OBSERVABILITY").is_err() {
        return Ok(None);
    }
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(DEFAULT_OBSERVABILITY_FILTER));

    let log_dir = shirl_core::config_home()
        .context("cannot determine shirl sessions directory")?
        .join("sessions")
        .join(session_id.to_string());

    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("creating session directory {}", log_dir.display()))?;

    let log_path = log_dir.join("observability.log");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening observability log at {}", log_path.display()))?;

    let (non_blocking, guard) = tracing_appender::non_blocking(file);

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(non_blocking)
        .pretty()
        .try_init()
        .map_err(|e| anyhow::anyhow!("init observability: {e}"))?;

    Ok(Some(guard))
}

fn print_help() {
    let version = env!("CARGO_PKG_VERSION");
    println!("shirl {version}");
    println!();
    println!("Usage: shirl [OPTIONS]");
    println!("       shirl -p \"PROMPT\" [OPTIONS]");
    println!();
    println!("Mode:");
    println!("  -p, --print [PROMPT]    Run non-interactively. Reads prompt from stdin");
    println!("                          when omitted, or appends piped stdin to PROMPT.");
    println!();
    println!("Output (headless only):");
    println!("      --json              Emit one JSON object instead of text.");
    println!("      --diff              Append full unified diff of changes.");
    println!();
    println!("Permissions:");
    println!("      --accept-edits      Auto-approve file edits; ask for bash.");
    println!("      --auto              Auto-approve everything.");
    println!();
    println!("Sandbox:");
    println!("      --sandbox            Run with OS-level sandbox.");
    println!("      --restrict-network   Block network access for the session.");
    println!("                           Implies --sandbox.");
    println!();
    println!("Session:");
    println!("      --resume <ID>       Resume a prior session by id.");
    println!("  -c, --continue          Resume the most recent session.");
    println!();
    println!("Other:");
    println!("  -h, --help              Print this help and exit.");
    println!("  -V, --version           Print version and exit.");
    println!();
    println!("Exit codes: 0 success · 1 error · 2 usage · 5 config-incomplete");
}

/// Output of pure-flag parsing - no env or stdin touched yet.
///
/// `--continue` resolution and stdin reading happen in [`parse_args`] using
/// the values on this struct as inputs.
#[derive(Debug)]
struct ParsedArgs {
    headless: bool,
    prompt: Option<String>,
    json: bool,
    diff: bool,
    resume: Option<SessionId>,
    continue_recent: bool,
    permission_mode: sweet_core::PermissionMode,
    sandbox_policy: SandboxPolicy,
}

/// Pure parsing: takes an argv vector and validates flag combinations.
///
/// Returns a [`ParsedArgs`] with everything that does not depend on the
/// filesystem or stdin. Help/version cause `process::exit(0)` here, which is
/// why those branches are not unit-tested.
fn parse_argv(argv: Vec<String>) -> Result<ParsedArgs> {
    let mut i = 0;
    let mut headless = false;
    let mut prompt: Option<String> = None;
    let mut json = false;
    let mut diff = false;
    let mut resume: Option<SessionId> = None;
    let mut continue_recent = false;
    let mut accept_edits = false;
    let mut auto = false;
    let mut sandbox = false;
    let mut restrict_network = false;

    while i < argv.len() {
        match argv[i].as_str() {
            "-p" | "--print" => {
                headless = true;
                // The prompt is optional. Peek at the next arg: if it exists
                // and doesn't start with `-`, it's the prompt.
                if i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                    i += 1;
                    prompt = Some(argv[i].clone());
                }
            }
            "--json" => json = true,
            "--diff" => diff = true,
            "--resume" => {
                i += 1;
                let id = argv.get(i).context("missing session id after --resume")?;
                resume = Some(
                    SessionId::from_str(id)
                        .with_context(|| format!("invalid session id `{id}`"))?,
                );
            }
            "-c" | "--continue" => continue_recent = true,
            "--accept-edits" => accept_edits = true,
            "--auto" => auto = true,
            "--sandbox" => sandbox = true,
            "--restrict-network" => {
                restrict_network = true;
                sandbox = true;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("shirl {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("unrecognized argument `{other}`\nRun `shirl --help` for usage.")
            }
        }
        i += 1;
    }

    // Validation: --json and --diff require headless mode
    if json && !headless {
        anyhow::bail!("--json requires -p");
    }
    if diff && !headless {
        anyhow::bail!("--diff requires -p");
    }
    // --continue and --resume are mutually exclusive
    if continue_recent && resume.is_some() {
        anyhow::bail!("cannot use both --continue and --resume");
    }

    // Mutual exclusivity checks
    if auto && accept_edits {
        anyhow::bail!("cannot use both --auto and --accept-edits");
    }
    // No mutual-exclusivity check needed for --restrict-network: it
    // implies --sandbox, so both flags together is fine.

    // Permission mode: headless defaults to FullAuto unless an explicit mode
    // was requested.
    let permission_mode = if auto {
        sweet_core::PermissionMode::FullAuto
    } else if accept_edits {
        sweet_core::PermissionMode::AutoEdit
    } else if headless {
        sweet_core::PermissionMode::FullAuto
    } else {
        sweet_core::PermissionMode::Normal
    };

    // Network policy. Fixed for the session - neither macOS Seatbelt nor Linux
    // bwrap can filter network by host or IP at the kernel layer.
    let sandbox_policy = if restrict_network {
        SandboxPolicy::Restricted
    } else if sandbox {
        SandboxPolicy::Sandbox
    } else {
        SandboxPolicy::Off
    };

    Ok(ParsedArgs {
        headless,
        prompt,
        json,
        diff,
        resume,
        continue_recent,
        permission_mode,
        sandbox_policy,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::picker;

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_argv_defaults_to_interactive_normal_mode() {
        let parsed = parse_argv(argv(&[])).unwrap();
        assert!(!parsed.headless);
        assert!(parsed.prompt.is_none());
        assert!(!parsed.json && !parsed.diff);
        assert!(!parsed.continue_recent);
        assert!(parsed.resume.is_none());
        assert!(matches!(parsed.sandbox_policy, SandboxPolicy::Off));
        assert!(matches!(
            parsed.permission_mode,
            sweet_core::PermissionMode::Normal
        ));
    }

    #[test]
    fn parse_argv_p_with_inline_prompt_sets_headless_and_full_auto() {
        let parsed = parse_argv(argv(&["-p", "do thing"])).unwrap();
        assert!(parsed.headless);
        assert_eq!(parsed.prompt.as_deref(), Some("do thing"));
        assert!(matches!(
            parsed.permission_mode,
            sweet_core::PermissionMode::FullAuto
        ));
    }

    #[test]
    fn parse_argv_p_without_prompt_leaves_prompt_unset_for_stdin() {
        // -p followed by another flag should not consume that flag as prompt.
        let parsed = parse_argv(argv(&["-p", "--json"])).unwrap();
        assert!(parsed.headless);
        assert!(parsed.prompt.is_none());
        assert!(parsed.json);
    }

    #[test]
    fn parse_argv_long_print_form_works() {
        let parsed = parse_argv(argv(&["--print", "x"])).unwrap();
        assert!(parsed.headless);
        assert_eq!(parsed.prompt.as_deref(), Some("x"));
    }

    #[test]
    fn parse_argv_json_without_p_errors() {
        let err = parse_argv(argv(&["--json"])).unwrap_err();
        assert!(err.to_string().contains("--json requires -p"));
    }

    #[test]
    fn parse_argv_diff_without_p_errors() {
        let err = parse_argv(argv(&["--diff"])).unwrap_err();
        assert!(err.to_string().contains("--diff requires -p"));
    }

    #[test]
    fn parse_argv_continue_and_resume_are_mutually_exclusive() {
        let id = SessionId::new().to_string();
        let err = parse_argv(argv(&["--resume", &id, "-c"])).unwrap_err();
        assert!(err
            .to_string()
            .contains("cannot use both --continue and --resume"));
    }

    #[test]
    fn parse_argv_auto_and_accept_edits_are_mutually_exclusive() {
        let err = parse_argv(argv(&["--auto", "--accept-edits"])).unwrap_err();
        assert!(err
            .to_string()
            .contains("cannot use both --auto and --accept-edits"));
    }

    #[test]
    fn parse_argv_sandbox_flag_enables_sandbox() {
        let parsed = parse_argv(argv(&["--sandbox"])).unwrap();
        assert!(matches!(parsed.sandbox_policy, SandboxPolicy::Sandbox));
    }

    #[test]
    fn parse_argv_sandbox_with_restrict_network_accepted() {
        let parsed = parse_argv(argv(&["--sandbox", "--restrict-network"])).unwrap();
        assert!(matches!(parsed.sandbox_policy, SandboxPolicy::Restricted));
    }

    #[test]
    fn parse_argv_restrict_network_implies_sandbox() {
        let parsed = parse_argv(argv(&["--restrict-network"])).unwrap();
        assert!(matches!(parsed.sandbox_policy, SandboxPolicy::Restricted));
    }

    #[test]
    fn parse_argv_accept_edits_maps_to_auto_edit() {
        let parsed = parse_argv(argv(&["--accept-edits"])).unwrap();
        assert!(matches!(
            parsed.permission_mode,
            sweet_core::PermissionMode::AutoEdit
        ));
    }

    #[test]
    fn parse_argv_restrict_network_sets_restricted_policy() {
        let parsed = parse_argv(argv(&["--sandbox", "--restrict-network"])).unwrap();
        assert!(matches!(parsed.sandbox_policy, SandboxPolicy::Restricted));
    }

    #[test]
    fn parse_argv_resume_parses_session_id() {
        let id = SessionId::new();
        let parsed = parse_argv(argv(&["--resume", &id.to_string()])).unwrap();
        assert_eq!(
            parsed.resume.as_ref().map(|s| s.to_string()),
            Some(id.to_string())
        );
    }

    #[test]
    fn parse_argv_resume_rejects_invalid_id() {
        let err = parse_argv(argv(&["--resume", "not-a-uuid"])).unwrap_err();
        assert!(err.to_string().contains("invalid session id"));
    }

    #[test]
    fn parse_argv_resume_errors_when_missing_value() {
        let err = parse_argv(argv(&["--resume"])).unwrap_err();
        assert!(err.to_string().contains("missing session id"));
    }

    #[test]
    fn parse_argv_unknown_flag_errors() {
        let err = parse_argv(argv(&["--nope"])).unwrap_err();
        assert!(err.to_string().contains("unrecognized argument"));
    }

    #[test]
    fn parse_argv_continue_short_form_sets_flag() {
        let parsed = parse_argv(argv(&["-c"])).unwrap();
        assert!(parsed.continue_recent);
    }

    #[test]
    fn truncate_str_zero_max_returns_empty() {
        assert_eq!(picker::truncate_str("hello", 0), "");
    }

    #[test]
    fn truncate_str_one_max_is_ellipsis_when_truncated() {
        assert_eq!(picker::truncate_str("hello", 1), "…");
    }

    #[test]
    fn truncate_str_passes_through_when_shorter_than_max() {
        assert_eq!(picker::truncate_str("abc", 10), "abc");
        assert_eq!(picker::truncate_str("", 5), "");
    }

    #[test]
    fn truncate_str_passes_through_at_exact_length() {
        assert_eq!(picker::truncate_str("abcde", 5), "abcde");
    }

    #[test]
    fn truncate_str_truncates_with_ellipsis() {
        assert_eq!(picker::truncate_str("abcdefgh", 5), "abcd…");
        assert_eq!(picker::truncate_str("hello world", 8), "hello w…");
    }

    #[test]
    fn truncate_str_handles_multibyte_chars_by_char_count() {
        assert_eq!(picker::truncate_str("αβγδε", 5), "αβγδε");
        assert_eq!(picker::truncate_str("αβγδεζ", 5), "αβγδ…");
    }
}
