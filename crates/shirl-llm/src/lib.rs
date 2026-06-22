// Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors
// SPDX-License-Identifier: Apache-2.0

//! Shirl-specific LLM provider management.
//!
//! This crate sits between `shirl-core` / `shirl-cli` and `sweet-llm`,
//! providing shirl-specific conveniences for working with providers:
//!
//! - [`catalog`] - fetch, parse, and cache the models.dev catalog
//! - [`factory`] - construct `Arc<dyn Model>` from catalog-derived parameters

pub mod catalog;
pub mod factory;

pub use catalog::{Catalog, CatalogModel, CatalogProvider, Protocol, ReasoningOption};
pub use factory::{build_embedder, build_model, can_disable_reasoning, ReasoningSettings};

/// Re-exported from `sweet-llm` so downstream crates that don't depend on
/// `sweet-llm` directly can name the sampling config `build_model` consumes.
pub use sweet_llm::SamplingConfig;
