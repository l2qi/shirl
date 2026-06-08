---
name: debug-sandbox
description: >-
  Diagnose sandbox behaviour in shirl — why a tool or bash command works
  unsandboxed but fails under --sandbox / --restrict-network, why a read is
  denied, why ~/.shirl is (correctly) hidden, or why bwrap is missing. Covers how
  shirl-cli wires sweet's OsSandbox. For the macOS Seatbelt / Linux Bubblewrap /
  RestrictedFs internals, defer to the debug-sandbox skill in the sweet repo.
---

# Debugging shirl's sandbox

Shirl does not implement the sandbox — it **consumes** sweet's. The enforcement
code (`SeatbeltRunner`, `BubblewrapRunner`, `RestrictedFs`, `tool_paths`, SBPL
profiles, bwrap mounts) lives in `../sweet/crates/sweet-sandbox/`. For the
internals — profile rules, mount order, the `resolve_tool_roots` filter, the
device-node allowlist — read the **`debug-sandbox` skill in the sweet repo**
(`sweet/.agents/skills/debug-sandbox/`). This skill covers only shirl's wiring.

## How shirl constructs the sandbox

Two call sites build it (interactive + headless):

- `crates/shirl-cli/src/main.rs`
- `crates/shirl-cli/src/headless/mod.rs`

Both call:

```rust
OsSandbox::new(
    current_dir,            // project root: the only writable tree
    sandbox_policy,         // Off | Sandbox | Restricted
    tracking::sandbox_read_roots(),   // extra_read_roots
    vec![".shirl".to_string()],       // extra_secret_dirs
)
```

If `OsSandbox::new` errors (e.g. `bwrap` not installed on Linux), shirl prints a
warning and falls back to `DirectSandbox` (unsandboxed). `build_agent` in
`shirl-agents` then wires every tool through `sandbox.fs()` / `sandbox.runner()`.

## CLI flag → policy

| Flag | `SandboxPolicy` |
|---|---|
| (none) | `Off` — no OS sandbox (default) |
| `--sandbox` | `Sandbox` — OS sandbox, network allowed |
| `--restrict-network` | `Restricted` — OS sandbox, network blocked (implies `--sandbox`) |

Policy is fixed for the run; changing it means restarting `shirl`.

## The two shirl-specific arguments

### `extra_read_roots` — `~/.shirl/sessions`

`tracking::sandbox_read_roots()` returns the **sessions root** so the agent can
read back the workflow tracker's `plans/`/`reviews/` files even though the home
directory is otherwise hidden. Read-only on purpose — those files are written
out-of-band by the host process, never through a tool. The *root* (not the
per-session dir) is used so reads survive a `/new` that rotates the session id.

**Linux caveat:** `extra_read_roots` only affects the in-process `RestrictedFs`
(the `read_file`/`grep`/… tools). Bash commands run through `BubblewrapRunner`,
which `--tmpfs`-hides `$HOME`, so `cat ~/.shirl/sessions/.../plan.md` from a shell
tool still won't see it — the agent must use `read_file` (the tracker reminder
tells it to).

### `extra_secret_dirs` — `.shirl`

Shirl passes `vec![".shirl"]` so `~/.shirl` (which holds `auth.toml` — the
provider API keys) is **never** exposed to the sandbox, on top of sweet's
universal credential denylist (`.ssh`, `.aws`, …). Without this, a sandboxed tool
could read shirl's own keys. To see exactly what stays readable, run sweet's
`show-tool-roots.sh .shirl` (the `.shirl` arg mirrors shirl's config).

Note `auth.toml` lives directly under `~/.shirl`, *outside* `~/.shirl/sessions`,
so granting the sessions read-root never re-exposes credentials.

## Quick triage

| Symptom | Likely cause |
|---|---|
| A `read_file` of a project file is denied | path is outside the project root / not a tool root — see sweet's skill (`tool_paths`) |
| The agent can't read `~/.shirl/sessions/.../plan.md` via **bash** | expected on Linux — bash sees a tmpfs `$HOME`; use `read_file` |
| `~/.shirl` contents reachable inside the sandbox | bug — verify `vec![".shirl"]` is still passed at both call sites |
| `--restrict-network` set, but `HttpFetch`/`WebSearch` still reach the net | known gap — in-process `reqwest` bypasses the runner (sweet `known-gaps.md`) |
| Linux: runs unsandboxed despite `--sandbox` | `bwrap` not installed; shirl warned and fell back to `DirectSandbox` |

If the issue is in the enforcement itself (a profile rule, a mount, the tool-root
filter), fix it in `../sweet/crates/sweet-sandbox/` and follow the sweet skill.
