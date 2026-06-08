Run the mandatory pre-commit checklist in order:

1. `cargo fmt --all` — fix formatting, don't just check.
2. `cargo clippy --workspace --all-targets -- -D warnings` — zero warnings.
3. `cargo test --workspace`
4. `cargo doc --workspace --no-deps --all-features`

Fix any issues found and re-run failed steps until all pass cleanly.
Do NOT commit — just verify the tree is green.
