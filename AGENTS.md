# Agent Guide

Operational brief for coding agents (Claude Code, etc.) working in this repo. Read this before touching anything.

## What lives here

Shirl is a terminal-based coding assistant CLI built on the [Sweet](https://github.com/sweet/sweet) AI agent framework. It provides:

- **Three peer agents**: main (coding), plan (structured planning), review (code review)
- **Five subagents**: explore, diagnose, explain, testgen, web\_research
- **Headless orchestrator**: `shirl -p` for non-interactive use
- **Terminal UI**: inline viewport ratatui TUI with file picker and clipboard paste
- **LLM catalog**: models.dev-backed provider discovery and factory
- **Long-term memory**: one `~/.shirl/memory.db` shared across sessions, with per-turn recall, agent memory tools, and background distillation

Dependency direction (one-way - never reverse):
```
shirl-cli -> shirl-{core,llm,ui,agents,tools} -> sweet-*
shirl-cli -> sweet-{agent,core,llm,sandbox,tools,mcp,session}
shirl-core -> sweet-{core,agent,session,memory}
shirl-llm -> sweet-{core,llm}
shirl-ui -> sweet-{agent,core}
shirl-agents -> sweet-{agent,core,tools,memory}; shirl-tools -> sweet-core
shirl-tools -> sweet-core
```

Sweet is an external dependency, pinned to a `v0.3.x` release tag as a git dependency in `Cargo.toml` (`[workspace.dependencies]`; local development can override it back to `../sweet/crates/` via the `[patch]` section). **Do not add sweet-* crates as members of this workspace.** Treat the sweet crate boundary as a repo boundary - no reaching across without a proper API.

## Before you write a line of code

### Ask first when

- The task requires an architectural decision or tradeoff (new abstraction, new crate, changing a public API).
- The scope is unclear or the right approach has non-obvious consequences.
- You'd need to change more than one crate's public surface to complete the task.

### Proceed without asking when

- The task is localized (a bug fix, a new tool, a new test).
- The change follows an established pattern (adding a tool, extending a builder, writing a test).
- The diff is small and obviously reversible.

## Quality bar

In order:

1. **Correctness** - tests must pass, including the full `--workspace` suite.
2. **Simplicity (KISS)** - the simplest solution that works. No defensive complexity.
3. **DRY** - no copy-paste logic. Shared code lives in the right crate.
4. **Cohesion** - every module/struct/fn does one thing.
5. **Test coverage** - new behavior needs tests. New error paths need tests.

Anti-patterns to avoid:
- Half-finished implementations (no TODO stubs committed).
- Abstractions for hypothetical future requirements.
- Comments that describe what the code does rather than why.
- `unwrap()` in production code paths. Use `?` or a typed error.

## Security - hard rules

- **Never hardcode API keys.** Keys are stored in `~/.shirl/auth.toml`, never in env vars or source.
- **Never commit secrets or `.env` files.**

## Established code patterns

### Error handling

- **Library crates** (`shirl-core`, `shirl-llm`, etc.): use `anyhow` for error propagation. Wrap external errors with `.context()` for clarity. Define `thiserror` error enums only when callers need to match on specific variants (e.g. `shirl-ui` clipboard errors).
- **Binary** (`shirl-cli`): use `anyhow`, propagate with `?`.

### Async

- Runtime is `tokio`. Only `shirl-cli` depends on it directly with full features.
- Libraries stay runtime-agnostic - no `#[tokio::main]`, no `tokio::spawn` in library code.

### Tool patterns

- **Stateful tools** (needing injected `Arc<dyn Filesystem>` or `Arc<dyn CommandRunner>`) use the factory pattern: a `xxx_tool(dep) -> ToolSpec` function creates a private handler struct implementing `ToolHandler` manually. See `shirl-tools` for the pattern.
- **Domain-specific tools** live in `shirl-tools`. **Universal tools** live in `sweet-tools`.
- `build_agent` in `shirl-agents` takes `Arc<dyn Sandbox>` and wires tools via factory functions.

### Agents and subagents

- **Three peer agents** (`Main`, `Plan`, `Review`) are in `shirl-agents/src/agents/`. Each has its own system prompt, tool set, and handoff tools.
- **Five subagents** (`explore`, `diagnose`, `explain`, `testgen`, `web_research`) are in `shirl-agents/src/subagents/`. Each takes `Arc<dyn Sandbox>`.
- **Headless orchestrator** is in `shirl-agents/src/headless/`. It has read-only file tools and three worker subagents (plan, implement, review). Workers share session state via `SharedSessionHandle`.
- Handoff tools (`transfer_to_plan`, `transfer_to_review`, `transfer_to_main`) are registered on each interactive agent. Handoffs are an interactive-mode concept; headless workers surface them as `ToolError::Execution`.

### Workflow tracker

`PlanTracker` in `shirl-core` is the durable workflow state for a session, stored under `~/.shirl/sessions/<id>/`:
- `plans/<ts>-<slug>.md` / `reviews/<ts>-<slug>.md` - handed-over reports
- `tracker.json` - active source (Plan | Review | None) and todo list

It connects to the main agent via a `DynamicPrompt` impl that re-renders into the system prompt every turn (compaction-proof). The tracker is wired by `shirl-cli`, keeping `shirl-core` and `shirl-agents` decoupled.

### Discovery providers

All three disk-discovered providers share the CWD->git-root walk in the private `discovery` module of `shirl-core`:

| Provider | User-level | Project-level |
|----------|-----------|---------------|
| Commands | `~/.shirl/commands/*.md` | `.agents/commands/*.md` |
| Skills | `~/.shirl/skills/*/SKILL.md` | `.agents/skills/*/SKILL.md` |
| AGENTS.md | `~/.shirl/AGENTS.md` | `AGENTS.md` (git-root walk) |

### Session management

`PersistedSession` (SQLite) lives in `shirl-core`. Sessions are stored at `~/.shirl/sessions/<id>/session.db`. `CodingAgent` wraps `sweet_agent::Agent` and installs auto-compaction as a `BeforeModelCall` hook. Compaction marks replaced rows archived in the same db (sweet >= 0.3.4) rather than deleting them; the Ctrl+O transcript popup shows the full history via `PersistedSession::full_items()`, hiding the synthetic compaction artifacts (which all carry `compacted = true`, including cleared-tool-result placeholders).

### Long-term memory

One `~/.shirl/memory.db` (`sweet_memory::SqliteMemory`, WAL - safe across concurrent shirl instances) holds every memory. Scopes: `MemoryScope::User("default")` (personal) and `MemoryScope::Project(<canonical git root>)` (per-codebase, same identity AGENTS.md discovery uses). Saves land in the project scope; recall and search see both.

- `shirl-core::memory` - `open_store`, `memory_db_path`, `user_scope`, `project_scope`; `[memory]` config (`enabled`, `embedder = "provider/model-id"`, `recall_limit`, `auto_distill`) lives on `ShirlConfig::memory`.
- `shirl-llm::build_embedder` - builds an `Arc<dyn Embedder>` for semantic recall (OpenAI/Gemini protocols only; keyword-only FTS5 recall when unset). Changing the configured embedder demotes existing memories to keyword-only recall - vectors are not re-embedded.
- `shirl-agents::MemoryWiring` + per-kind policy in `shirl-agents/src/memory.rs`: Main = tools + recall; Plan = `memory_search` + recall; Review = recall only; headless orchestrator = Main policy plus the in-turn distill procedure (non-interactive, blocking is fine); headless workers = none (ephemeral child sessions must not write long-term memory).
- Distillation never blocks the interactive UI: `shirl-cli::memory_cmd::spawn_distill` claims the pending span synchronously (shared `MemoryDistiller` watermark - survives mode switches, no double-distill) and runs `distill_span` on a detached task, at turn end (cadence-gated, ~12 items) and on `/new`; outcomes surface as a `✦ long-term memory: ...` scrollback line. Quit does **not** flush - an undistilled tail is picked up when the session is resumed. `auto_distill = false` disables the automatic passes only.
- `shirl-cli::memory_cmd` - builds the wiring at startup (warnings, never fatal) and implements `/memory` (`list`, `add`, `search`, `forget`, `distill`, `help`; debug builds additionally get `forget all <user|project>`). `/memory distill` first joins any in-flight background pass (tracked in a `DistillTask` slot on `RuntimeCtx`), then distills the remainder inline under the working indicator and prints what it wrote; it works regardless of `auto_distill`.

### LLM catalog

`shirl-llm` fetches the models.dev provider catalog (`~/.shirl/cache/`, 24h TTL) and maps providers to one of three `Protocol`s. `build_model` constructs an `Arc<dyn Model>` from the catalog entry.

### Tests

- **Unit tests**: `#[cfg(test)] mod tests { ... }` inside the source file.
- **Integration tests**: `crates/<crate>/tests/*.rs`.
- Async tests: `#[tokio::test]`.

## Mandatory pre-commit checklist

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps --all-features
```

## Git hygiene

- **Never run `git commit` or `git push` (or open a PR) without the owner's explicit approval.** A green pre-commit checklist is a prerequisite, not authorization.
- Do not amend or rewrite an existing commit on your own initiative - the owner may have pushed it. Default to a follow-up commit; if amending seems like the better call, ask first.
- Do not skip hooks (`--no-verify`). Fix the underlying issue.
- Write commit messages in the imperative mood, <= 72 chars for the subject.

## Crate-by-crate quick reference

### shirl-core

Public surface: `PersistedSession`, `CodingAgent`, `ShirlConfig`, `MemoryConfig`, `ReasoningPref`, `SamplingPref`, `AgentsMd`, `CustomCommandsProvider`, `SkillsProvider`, `PlanTracker`, `session_dir`, `memory` module (`open_store`, `user_scope`, `project_scope`, `memory_db_path`), `paths` module (`config_home`, `config_dir_name`, `set_config_dir_name`).

- `PersistedSession` - SQLite-backed session at `~/.shirl/sessions/<id>/session.db`
- `CodingAgent` - wraps `sweet_agent::Agent` with auto-compaction hook
- `PlanTracker` - durable workflow state; exposes `dynamic_prompt()` and `write_todos_tool()`
- `session_dir(id)` - resolves `~/.shirl/sessions/<id>/`
- `paths` - single source of truth for the config home (`~/.shirl` by default); `ReasoningPref`/`SamplingPref` are plain-data per-agent overrides merged into the model build by `shirl-cli`

### shirl-llm

Public surface: `Catalog`, `CatalogProvider`, `CatalogModel`, `Protocol`, `ReasoningOption`, `ReasoningReplay`, `ReasoningSettings`, `SamplingConfig`, `build_model`, `build_embedder`, `can_disable_reasoning`.

- `catalog` - fetch, parse, and cache models.dev provider/model catalog; `Protocol` includes `Cerebras`, `ReasoningOption` mirrors the models.dev `reasoning_options` dialects (toggle / effort / budget), and `ReasoningReplay` (from the models.dev `interleaved` field) records whether/how a model's prior reasoning must be replayed on the next request (`Omit` / `ReasoningContent` / `Reasoning` / `ReasoningDetails`)
- `factory` - `build_model` constructs an `Arc<dyn Model>` from a `Protocol`, model id, base URL, and API key, plus the dialect-correct reasoning (`ReasoningSettings`), the per-model `ReasoningReplay` (mapped to `sweet_llm::ReasoningHistoryKey` for OpenAI-protocol providers), and `SamplingConfig`; `can_disable_reasoning` is the shared predicate for whether reasoning has an off-switch
- **Reasoning-replay default for off-catalog models**: `ReasoningReplay` is only known for models in the models.dev catalog. Custom `[providers.*]` entries (and any model not found in the catalog) default to `Omit` - prior reasoning is not replayed. That is safe for the vast majority of models, but a custom reasoning model that *requires* its prior reasoning echoed back (e.g. a self-hosted Kimi-style model) will lose multi-turn reasoning continuity; route such models through a catalog provider instead.

### shirl-tools

Public surface: `bash_tool`, `read_file_tool`, `write_file_tool`, `edit_file_tool`, `glob_tool`, `grep_tool`, `patch_tool`, `move_file_tool`, `create_directory_tool`, `directory_tree_tool`, `get_file_info_tool`, `directory_size_tool`, `head_file_tool`, `tail_file_tool`, `list_directory_tool`, `EditOperation`, `unified_diff`.

All tools use the factory pattern, taking `Arc<dyn Filesystem>` or `Arc<dyn CommandRunner>` from `sweet-core::sandbox`. Tests in `tests/tools.rs`.

### shirl-ui

Public surface: `ReplIo`, `SharedIo`, `Command`, `PickerEntry`, `PickerSection`, `PickerRenderState`, `StatusInfo`, `compute_title`, `default_history_path`, `picker_popup_width`, `picker_row_width`, `FileEntry`, `FilePickerState`, `clipboard_image` module.

Implements `AgentIo` via `ReplIo` using ratatui's inline viewport mode. Status line pinned at bottom; history appended to scrollback.

Feature flags:

| Flag | Default | Description |
|------|---------|------------|
| `clipboard-image` | yes | `Ctrl+V` / `Alt+V` image paste via arboard + image |

### shirl-agents

Public surface (agents): `AgentKind`, `ModeCommand`, `ModeSwitch`, `build_agent`, `resolve_mode_command`, `SharedWebSearchBackend`. Crate root also exports `MemoryWiring`.
Public surface (subagents): `explore_spec`, `diagnose_spec`, `explain_spec`, `testgen_spec`, `web_research_spec`.
Public surface (headless): `build_orchestrator`, `ORCHESTRATOR_PROMPT`, `ReportStore`, `Tracking`.

- `build_agent` - takes `Arc<dyn Sandbox>` and wires all tools; returns `(Agent<...>, CommandRouter)`
- `AgentKind` - closed over `Main`, `Plan`, `Review` (headless orchestrator is NOT a member)
- Wired tools by agent: Main = full toolset + all subagents + handoffs; Plan = read-only + explore + web\_research + handoffs; Review = read-only + handoffs

### shirl-cli

Binary name: `shirl`. Entry point: `crates/shirl-cli/src/main.rs`.

Wires `shirl-core::CodingAgent` with tools and starts the REPL or headless runner. Contains model management (`model.rs`), MCP loading (`mcp.rs`), file picker (`file_picker.rs`), picker UI (`picker.rs`), and headless runner (`headless/`).

Config files: `~/.shirl/auth.toml` (API keys), `~/.shirl/config.toml` (model selection).

## What to update when you change things

| Change | Also update |
|--------|------------|
| New tool (domain-specific) | `shirl-tools/src/<tool>.rs`, re-export in `lib.rs`, wire in `shirl-agents/src/agents/main_agent.rs`, add test to `shirl-tools/tests/tools.rs` |
| New subagent | `shirl-agents/src/subagents/<name>.rs`, register via `with_subagent` in the agent builder |
| New headless worker | `shirl-agents/src/headless/<name>_sub.rs`, wire into `headless::orchestrator::build`, update orchestrator prompt |
| New peer agent (handoff) | `shirl-agents/src/agents/<name>.rs`, add to `agents/mod.rs`, register handoff tools, update `resolve_mode_command` |
| New slash command | `shirl-cli/src/main.rs` dispatch, `shirl-agents/src/agents/mod.rs` `resolve_mode_command`, add to `RESERVED_COMMANDS`, picker entry in `shirl-ui/src/completion.rs` |
| New CLI flag | `shirl-cli/src/main.rs` `parse_args` + `print_help`, plus `headless/mod.rs` if it changes headless behavior |
| MCP config change | `sweet-mcp/src/config.rs` (in the sweet repo), README |
| Public API change in sweet-core or sweet-agent | All downstream shirl crates using that API |
