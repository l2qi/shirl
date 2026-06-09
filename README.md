# shirl

A terminal-based coding assistant.

Shirl runs in your terminal, understands your codebase, and autonomously reads files, runs shell commands, edits code, and searches the web to complete coding tasks.

## Installation

Download a pre-built binary from the [Releases](https://github.com/shirl/shirl/releases) page, or build from source:

```bash
git clone https://github.com/shirl/shirl
cd shirl
cargo install --path crates/shirl-cli
```

## Quick Start

On first launch, shirl prompts you to configure a provider and API key. Keys are stored in `~/.shirl/auth.toml`.

```bash
shirl
```

## Key Features

- **Three peer agents**: main (coding), plan (structured planning), review (code review)
- **Five subagents**: explore, diagnose, explain, testgen, web\_research
- **Headless mode**: `shirl -p "your task"` for non-interactive use and scripting
- **MCP integration**: connect any MCP server via `~/.shirl/mcp.json`
- **OS sandbox**: `--sandbox` flag enables macOS Seatbelt / Linux Bubblewrap isolation
- **File picker**: type `@` in the prompt to autocomplete project file paths
- **Image paste**: `Ctrl+V` pastes clipboard images directly into the conversation
- **Session persistence**: conversation history stored in `~/.shirl/sessions/`

## Agent Modes

| Command | Switches to |
|---------|-------------|
| `/plan [task]` | Plan agent — structured planning |
| `/review [focus]` | Review agent — code review |
| `/approve` | Main agent — from plan |
| `/fix [items]` | Main agent — from review |
| `/back` | Main agent — from either |

## Headless Mode

```bash
# Run a task non-interactively
shirl -p "Add unit tests for src/parser.rs"

# Continue the most recent session
shirl -p "Now add integration tests" --continue

# JSON output
shirl -p "Refactor auth.rs" --json
```

## Crates

| Crate | Description |
|-------|-------------|
| `shirl-core` | Session persistence, compaction, slash commands, workflow tracker |
| `shirl-llm` | Provider catalog (models.dev) and factory |
| `shirl-tools` | Coding tools: bash, read/write/edit file, glob, grep, patch |
| `shirl-ui` | Terminal UI via ratatui (inline viewport, file picker, clipboard) |
| `shirl-agents` | Agent definitions, system prompts, subagents, headless orchestrator |
| `shirl-cli` | Binary: REPL, model management, MCP loading, `~/.shirl/` config |

## Configuration

| File | Purpose |
|------|---------|
| `~/.shirl/auth.toml` | API keys for LLM providers and MCP servers |
| `~/.shirl/config.toml` | Default provider/model and per-agent overrides |
| `~/.shirl/mcp.json` | MCP server definitions |
| `~/.shirl/AGENTS.md` | Personal instructions injected into every session |
| `.agents/commands/*.md` | Project-level slash commands |
| `.agents/skills/*/SKILL.md` | Project-level skills |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SHIRL_OBSERVABILITY` | Enable per-session observability logs |
| `RUST_LOG` | Observability filter override |
| `GH_TOKEN` | GitHub PAT, referenced as `${GH_TOKEN}` in mcp.json |

## License

Copyright (C) 2026 Ryuichi Intellectual Property LLC and the Shirl project contributors

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
