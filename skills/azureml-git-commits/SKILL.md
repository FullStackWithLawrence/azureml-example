---
name: azureml-git-commits
description: Use whenever the user asks you to "write a git commit" (or a commit message, or the git add) in the azureml-example repository. Covers the never-execute rule, authorship, the multiple -m command format, conventional-commit types and scopes as commitlint and semantic-release read them, the 72-character header limit, zsh quoting traps, the CI skip trap, and how thorough the body must be.
---

# Git Commits

## The rule: write, never execute

"Write a git commit", in any wording or tone, means **output the command as
text**. It never means run it. Emphatic or frustrated phrasing still asks for
the command text.

- Never execute `git commit`, `git add`, `git reset`, `git restore --staged`,
  `git push`, `git tag`, `git stash`, or any other git command that writes.
  The user stages, commits and pushes. (`.claude/settings.local.json` denies
  most of these too; don't look for a way around it.)
- Read-only commands (`git status`, `git diff`, `git log`) are fine.
- Don't volunteer commit messages. Write one only when asked.

## Authorship

You are the author; Lawrence McDaniel is the co-author.

- `--author="Claude <noreply@anthropic.com>"`
- The last `-m` is `Co-authored-by: Lawrence McDaniel <lpm0073@gmail.com>`.

## Command format

Exactly one `git commit` command in one code block:

```console
git commit --author="Claude <noreply@anthropic.com>" \
  -m "<type>(<scope>): <subject>" \
  -m "<body paragraph>" \
  -m "<body paragraph or bullet list>" \
  -m "Co-authored-by: Lawrence McDaniel <lpm0073@gmail.com>"
```

- One `-m` per paragraph; git joins them with blank lines.
- No heredoc, no `-F`, no `git add` in the same block. When the user asks for
  the `git add`, write it as a separate block that lists paths explicitly.
- No backticks, `$` or `!` inside the double-quoted messages. In zsh a `!` in
  double quotes is history expansion and aborts the command with
  `event not found`. Reword instead.
- **Never write a CI skip instruction** anywhere in the message, not even to
  describe one (`[skip ci]`, `[ci skip]`, `[no ci]`, `skip-checks: true`).
  GitHub then starts no workflows for the push. semantic-release's own release
  commits use one on purpose; yours never do. Write "a CI skip instruction".

## Subject line

`<type>(<scope>): <subject>`, checked by commitlint
(`@commitlint/config-conventional` + `@commitlint/config-angular`) and read by
semantic-release (`release.config.js`, branches `main`, `beta`, `alpha`),
which versions the release from it.

- **type:** `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `style`,
  `build`, `ci`, `chore`.
  - `feat` (minor release) is for a truly new capability: a new helper class,
    a new command, a new deployment example. When in doubt, it is `fix`.
  - `fix` (patch release) covers bugs and shortcomings of something that
    exists, however much code the fix takes.
  - Notebook and docs content is `docs`; dependency pins are `build` or
    `chore(deps)`; workflow changes are `ci`.
  - `chore(release): ...` belongs to semantic-release. Never write it.
- **scope:** the area, in lower case: `dataset`, `cluster`, `endpoint`,
  `automl`, `workspace`, `commands`, `tests`, `function`, `notebooks`,
  `docs`, `deps`, `skills`. Omit it for repository-wide changes.
- **subject:** imperative mood, lower case, no trailing period.
- **length:** the whole header must be **72 characters or fewer**
  (config-angular). Measure it, aim for 70:

  ```console
  printf %s "fix(dataset): fall back to Excel when a csv file is a workbook" | wc -c
  ```

## Body

Write for a reader with no context, such as a student reading the history six
months from now.

- Start with the problem or motivation: what was wrong or missing, and how it
  showed up.
- Then area by area, a short heading paragraph per area (e.g.
  `Dataset helper (azure_ai/ml_studio.py):`), then one bullet per change, each
  saying what changed and why.
- Name files, classes, methods and commands precisely.
- Cover the tests: what's new, what they prove, and whether they ran offline
  or live.
- End with the status: skipped tests, pre-existing failures, follow-ups.

## Before writing

1. Run `git status` and `git diff --stat` (and `git diff` where unclear), so the
   message describes the working tree, not your memory.
2. Ask which files belong to the commit if that's ambiguous.
3. Measure the header with `wc -c`.
4. Start the response with a one-line `Written for:` audience note.
