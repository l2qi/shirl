// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Core logic for the Shirl coding assistant.

pub mod agents_md;
pub mod auth;
pub mod commands;
pub mod compaction;
pub mod config;
pub mod custom_commands;
mod discovery;
pub mod hooks;
pub mod media_input;
mod media_strip;
pub mod memory;
pub mod paths;
pub mod session;
pub mod skills;
pub mod tracker;

pub use agents_md::AgentsMd;
pub use auth::AuthStore;
pub use commands::{parse_slash_command, Clear, Compact, New};
pub use compaction::{
    compact_session, install_auto_compaction, CompactionConfig, DEFAULT_PRESERVE_RECENT,
};
pub use config::{AgentModelConfig, MemoryConfig, ReasoningPref, SamplingPref, ShirlConfig};
pub use custom_commands::CustomCommandsProvider;
pub use hooks::AutoCompactionProcedure;
pub use media_input::{has_files, has_images, resolve_media, Resolved};
pub use media_strip::install_media_strip;
pub use paths::{config_dir_name, config_home, set_config_dir_name};
pub use session::{session_dir, sessions_root, PersistedSession};
pub use skills::SkillsProvider;
pub use tracker::PlanTracker;
