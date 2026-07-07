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
    current_dir,            // project root: always readable + writable
    sandbox_policy,         // Off | Sandbox | Restricted
    SandboxRoots {
        read: tracking::sandbox_read_roots(),   // extra read-only roots
        write: tracking::sandbox_write_roots(), // extra read+write roots
    },
    vec![".shirl".to_string()],       // extra_secret_dirs
)
```

The 3rd argument is a `SandboxRoots { read, write }` (sweet ≥ 0.3.7); before
that it was a bare `extra_read_roots: Vec<PathBuf>` positional.

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

## The three shirl-specific arguments

### `read` — `~/.shirl/sessions` + ancestor `.cargo` dirs

`tracking::sandbox_read_roots()` returns two kinds of read-only root:

- **The sessions root** so the agent can read back the workflow tracker's
  `plans/`/`reviews/` files even though the home directory is otherwise hidden.
  Those files are written out-of-band by the host process, never through a tool.
  The *root* (not the per-session dir) is used so reads survive a `/new` that
  rotates the session id.
- **Ancestor `.cargo` dirs** (`tracking::ancestor_cargo_dirs`) — cargo discovers
  config by walking from the working dir up through every ancestor, reading each
  `.cargo/config.toml`. A `[patch]` overlay in a parent dir (e.g. the alset-dev
  meta-repo layout) lives outside the project root, so without this the build
  fails when cargo's config walk hits a denied read. Strict ancestors only; the
  project root's own `.cargo` is already readable.

**Both layers honor these** (sweet ≥ 0.3.7, "Honor extra_read_roots in the OS
command runners"): the in-process `RestrictedFs` (the `read_file`/`grep`/… tools)
*and* the OS command runner, which `--ro-bind`s each read root on top of the
`$HOME` tmpfs. So a bash `cargo build` can read an ancestor `.cargo`, and a shell
`cat ~/.shirl/sessions/.../plan.md` now works too. `auth.toml` stays hidden — it
sits directly under `~/.shirl`, under no read or write root, and
`extra_secret_dirs` keeps `~/.shirl` out of the tool-root set so it stays that
way (it filters the readable roots, not a blanket read-denylist — see below).

### `write` — `$CARGO_HOME`

`tracking::sandbox_write_roots()` returns `$CARGO_HOME` (default `~/.cargo`, when
it exists) as a **read+write** root. `cargo build` must populate its registry
cache, git checkouts, and the `.package-cache` lock there; without write access
every fetch fails with `Operation not permitted`. The default `~/.cargo` is
already readable (a known tool dir); a non-default `$CARGO_HOME` is made readable
by the write root itself, since both sandbox layers fold write roots into their
read set. A write root is also *executable* (macOS Seatbelt grants `process-exec`;
Linux bind mounts are exec by default).

**Conscious tradeoff:** this makes the *whole* `$CARGO_HOME` tree writable and
every existing ancestor `.cargo` readable — deliberately broad, matching cargo's
unsandboxed behavior (it writes caches across the tree). A planted ancestor
`.cargo/config.toml` is a cargo config-injection concern, not a sandbox
exfiltration one; `~/.shirl` and the universal credential denylist stay hidden
regardless. A no-op when the sandbox policy is `Off` or the dir doesn't exist.

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
| A read root (`~/.shirl/sessions`, an ancestor `.cargo`) isn't visible to a tool **or** bash | it must be a live path returned by `sandbox_read_roots()`; ancestor `.cargo` must be a *strict* ancestor of cwd and exist. (Both layers honor read roots as of sweet 0.3.7.) |
| `cargo build` fails with `Operation not permitted` writing under `~/.cargo` | `$CARGO_HOME` not a write root — dir missing at launch, or `sandbox_write_roots()` not passed at both call sites |
| `cargo build` fails reading an ancestor `.cargo/config.toml` (`[patch]` overlay) | that ancestor `.cargo` isn't surfaced — see `ancestor_cargo_dirs`; only strict ancestors of cwd are added |
| `~/.shirl` contents reachable inside the sandbox | bug — verify `vec![".shirl"]` is still passed at both call sites |
| `--restrict-network` set, but `HttpFetch`/`WebSearch` still reach the net | known gap — in-process `reqwest` bypasses the runner (sweet `known-gaps.md`) |
| Linux: runs unsandboxed despite `--sandbox` | `bwrap` not installed; shirl warned and fell back to `DirectSandbox` |

If the issue is in the enforcement itself (a profile rule, a mount, the tool-root
filter), fix it in `../sweet/crates/sweet-sandbox/` and follow the sweet skill.
