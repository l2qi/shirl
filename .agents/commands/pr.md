Create a pull request for the current branch using the `gh` CLI. Steps:

1. **Gather context.** First refresh the remote base so the comparison is
   against the *latest* master, not a stale local copy:
   - `git fetch origin master` — update `origin/master` (does not touch your
     working tree or local `master`)

   Then run these commands and study the output (note: compare against
   `origin/master`, never the local `master` branch, which may be behind):
   - `git branch --show-current` — the branch name
   - `git log origin/master..HEAD --oneline` — commits on this branch
   - `git diff origin/master...HEAD --stat` — file-level change summary
   - `git diff origin/master...HEAD` — the full diff

2. **Draft the PR.** Based on the commits and diff:
   - **Title:** imperative mood, ≤72 chars, summarizing the branch's purpose.
   - **Body:** group the changes into logical sections with Markdown headings. Explain *why*, not just what. Reference relevant issue numbers if present in commit messages.
   - **Base branch:** `master` unless the branch name or commits suggest otherwise. (The PR's base is the remote branch name `master`; `gh` resolves it against the remote regardless of your local `master`.)

3. **Create the PR.** Run:
   ```
   gh pr create --title "<title>" --body "<body>" --base <base>
   ```
   Use a heredoc or temp file for the body to avoid shell escaping issues.

4. **Report.** Print the PR URL from the output.

**Hard rules — do NOT:**
- Push the branch or any refs. If the branch is not pushed, tell the user to push first.
- Commit, stage, or amend anything.
- Modify any code or files. This command is read-only except for the `gh pr create` call.
