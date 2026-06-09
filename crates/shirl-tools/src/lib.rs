// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Domain-specific tools for the Shirl coding assistant.
//!
//! All tools that interact with the filesystem or shell go through the
//! [`sweet_core::sandbox`] traits. Use the `*_tool()` factory functions
//! to construct tool specs with the appropriate runner or filesystem.

mod bash;
mod create_directory;
mod directory_size;
mod directory_tree;
mod edit_file;
mod get_file_info;
mod glob;
mod grep;
mod head_file;
mod list_directory;
mod move_file;
mod patch;
mod read_file;
mod tail_file;
mod write_file;

pub use bash::bash_tool;
pub use create_directory::create_directory_tool;
pub use directory_size::directory_size_tool;
pub use directory_tree::directory_tree_tool;
pub use edit_file::{edit_file_tool, unified_diff, EditOperation};
pub use get_file_info::get_file_info_tool;
pub use glob::glob_tool;
pub use grep::grep_tool;
pub use head_file::head_file_tool;
pub use list_directory::list_directory_tool;
pub use move_file::move_file_tool;
pub use patch::patch_tool;
pub use read_file::read_file_tool;
pub use tail_file::tail_file_tool;
pub use write_file::write_file_tool;
