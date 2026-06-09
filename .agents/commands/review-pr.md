Review a pull request using the `gh` CLI. The PR number is given in $ARGUMENTS.

**Resolve the upstream remote.** Determine which remote points to the
canonical repo (`shirl/shirl`). Prefer a remote named `upstream`; if
none exists, fall back to `origin`.
```
UPSTREAM_REMOTE=$(git remote -v | grep -q '[u]pstream' && echo upstream || echo origin)
```
All `gh` commands below use `--repo shirl/shirl` when `$UPSTREAM_REMOTE`
is not `origin`, so the command works for fork-based contributors and
maintainers alike.

**Parse the argument:**
- If `$ARGUMENTS` is a number or `#<number>`, use that as the PR number.
- If `$ARGUMENTS` is a URL (e.g. `https://github.com/shirl/shirl/pull/42`),
  extract the PR number from it.
- If `$ARGUMENTS` is empty, find the PR for the current branch:
  ```
  gh pr list --head $(git branch --show-current) --json number --jq '.[0].number'
  ```
  If that returns nothing, try with the upstream repo:
  ```
  gh pr list --repo shirl/shirl --head <owner>:$(git branch --show-current) --json number --jq '.[0].number'
  ```
  where `<owner>` is extracted from the `origin` remote URL. If that also
  returns nothing, tell the user to pass a PR number explicitly.

**Gather the PR:**
1. `gh pr view <number> --json title,body,headRefName,baseRefName,files,additions,deletions,changedFiles` — PR metadata (add `--repo shirl/shirl` if `$UPSTREAM_REMOTE` is not `origin`)
2. `gh pr diff <number>` — the full diff (same `--repo` logic)
3. If the PR is large (>20 files or >2000 lines), focus on the most significant files and sample the rest. Do not silently skip files — say what you're sampling and why.

**Review against AGENTS.md rules.** Check for:
1. **Correctness** — tests pass, no `unwrap()` in production paths, proper `?` propagation
2. **Simplicity** — no defensive complexity, no speculative abstractions
3. **DRY** — no copy-paste logic
4. **Cohesion** — one thing per module/struct/fn
5. **Test coverage** — new behavior and new error paths have tests
6. **Dependency direction** — no upward imports across crate boundaries
7. **Feature gating** — new providers/tools behind their own Cargo feature
8. **Docs** — public API changes update AGENTS.md and README.md where applicable
9. **Security** — no hardcoded credentials, no secrets in code

**Format the review as:**
- A one-paragraph summary of what the PR does
- A bulleted list of findings, each prefixed with `[critical]` (must fix), `[warning]` (suggestion), or `[good]` (positive observation)
- Be concise. Quote specific lines when referencing code.
