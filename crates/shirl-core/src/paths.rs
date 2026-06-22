// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Single source of truth for the per-user config directory base.
//!
//! Every on-disk path (sessions, config, auth, memory, discovered
//! commands/skills) hangs off one directory under the user's home - `~/.shirl`
//! by default. A fork that wants its own home (e.g. `~/.myapp`) calls
//! [`set_config_dir_name`] once at process start, before any path is resolved;
//! everything else flows through [`config_home`] / [`config_dir_name`].

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::Result;

/// Default config directory name (the leading-dot home subdirectory).
const DEFAULT_CONFIG_DIR_NAME: &str = ".shirl";

static CONFIG_DIR_NAME: OnceLock<String> = OnceLock::new();

/// Override the config directory name. Call once at process start, before any
/// path is resolved. A leading dot is added if omitted, so both `myapp` and
/// `.myapp` resolve to the same `~/.myapp` home (and a matching dotless brand) -
/// downstream path consumers rely on the leading-dot form. Idempotent: later
/// calls (and the implicit default) are ignored, so the first writer wins.
pub fn set_config_dir_name(name: impl Into<String>) {
    let _ = CONFIG_DIR_NAME.set(with_leading_dot(&name.into()));
}

/// Ensure a config dir name carries the leading dot it is canonically stored
/// with (`myapp` -> `.myapp`, `.myapp` -> `.myapp`).
fn with_leading_dot(name: &str) -> String {
    if name.starts_with('.') {
        name.to_string()
    } else {
        format!(".{name}")
    }
}

/// The configured directory name, e.g. `.shirl` (default) or `.myapp`.
pub fn config_dir_name() -> &'static str {
    CONFIG_DIR_NAME
        .get()
        .map(String::as_str)
        .unwrap_or(DEFAULT_CONFIG_DIR_NAME)
}

/// Absolute path to the config home, e.g. `~/.shirl`.
pub fn config_home() -> Result<PathBuf> {
    home_join(config_dir_name())
}

/// Join `name` onto the user's home directory. Pure helper so the join logic is
/// testable without touching the process-global [`CONFIG_DIR_NAME`].
fn home_join(name: &str) -> Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    Ok(home.join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_dir_name_defaults_to_shirl() {
        // Note: this asserts the default rather than calling `set_config_dir_name`,
        // which is process-global and set-once - a mutation here would leak into
        // sibling tests sharing the same process.
        assert_eq!(DEFAULT_CONFIG_DIR_NAME, ".shirl");
    }

    #[test]
    fn home_join_appends_dir_name() {
        let home = dirs::home_dir().expect("home dir for test");
        assert_eq!(home_join(".myapp").unwrap(), home.join(".myapp"));
        assert_eq!(home_join(".shirl").unwrap(), home.join(".shirl"));
    }

    #[test]
    fn with_leading_dot_normalizes_and_is_idempotent() {
        // A fork passing a dotless name still lands under the leading-dot home,
        // matching the dotless brand derived via `trim_start_matches('.')`.
        assert_eq!(with_leading_dot("myapp"), ".myapp");
        assert_eq!(with_leading_dot(".myapp"), ".myapp");
        assert_eq!(with_leading_dot(".shirl"), ".shirl");
    }
}
