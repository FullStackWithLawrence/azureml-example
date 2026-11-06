---
name: azureml-pull-requests
description: Use when asked to create, open, write or update a GitHub pull request in FullStackWithLawrence/azureml-example, or to review or describe a contributor's pull request (often a notebook change from a fork). Covers the branches, the title, filling in every section of .github/PULL_REQUEST_TEMPLATE.md from the commits' own messages, creating it with gh pr create --body-file, the checks that run on pull requests, and what never to do (push, merge, leave placeholders in).
---

# Pull Requests

The repository is `FullStackWithLawrence/azureml-example`. Every pull request
uses [.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md).
`gh pr create --body-file` doesn't pre-fill it, so fill in every section from
the commits themselves.

## Rules

- **Create one only when asked.** A pull request is public and notifies
  people. Writing the description as text is fine when that's all the user
  asked for.
- **Never push, merge, or change a branch.** If the head branch isn't on
  GitHub, or is behind the local one, say so and stop.
- **Never leave a placeholder** (`Fixes #[Add issue number here.]`,
  `_Describe what this Pull Request does_`). Each section gets content or a
  plain "None.".
- **The repository is public.** No subscription ids, keys, or personal details.
- **No bare `@name`** unless you mean to notify that GitHub user.
- **No CI skip instruction** anywhere (see `azureml-git-commits`).

## Branches and title

- Work lands on `main` (the default). `alpha` and `beta` are semantic-release
  prerelease branches; `next` also exists. Ask which base the user wants if a
  branch isn't obviously meant for `main`.
- Students and course contributors open pull requests from forks (e.g. notebook
  updates). The title is the main commit's subject, a conventional commit.
- Check for an open one first: `gh pr list --repo FullStackWithLawrence/azureml-example --head <branch>`.
  If there is one, update it with `gh pr edit <n> --body-file ...`.

## Gather the material

```bash
git fetch origin
git status -sb
git log --oneline origin/main..<branch>
git log --format='### %h %s%n%n%b' origin/main..<branch>
git diff --stat origin/main...<branch> | tail -1
```

Leave out `chore(release)` and merge commits. Read every commit body.

## Fill in the template

Keep its headings and order, drop the `<!-- -->` hints.

- **Type of Change:** check every box a commit earns: `feat` = New feature,
  `fix` = Bug fix, `docs` (and notebook content) = Documentation, `refactor` =
  Refactor, `ci`/`build`/`chore`/`test` = Chore.
- **Resolves:** one `- Fixes #<n>` per issue a commit names, or `- None.`
- **Changes:** a sentence or two on the whole, then a `###` subsection per
  commit (subject and short hash) with the problem and bullets naming files.
- **Testing:** what ran and how: offline unit tests, live tests (and what they
  created and deleted), notebooks run by hand. Never claim a test nobody ran.
- **Screenshots:** for notebook plots or Azure portal changes; otherwise
  `None: no visible change.`
- **Dependencies:** changed pins in `requirements/` or `jupyter/requirements.txt`,
  new secrets (CI uses `PAT` and `KAGGLE_JSON`), new Azure resources a reader
  must create first. Otherwise `None.`
- **Breaking Changes:** changed class or command signatures, renamed
  notebooks, new required configuration. Otherwise `None.`

End with the attribution line from the conversation's system context, if any.

## What runs on a pull request

- `pullRequestController.yml` (`pull_request_target`, Python or requirements
  changes): a semantic-release dry run, then the offline Python tests.
- `test.yml` (`pull_request`, `*.py` changes): the offline Python tests.
- `auto-assign.yml`; Mergify approves and merges dependabot pull requests.

Because `pullRequestController.yml` uses `pull_request_target`, it runs with
the base repository's secrets. Point out to the user any fork pull request
that changes `.github/` files.

## Create it

```bash
gh pr create --repo FullStackWithLawrence/azureml-example --base main --head <branch> \
  --title "<type>(<scope>): <subject>" --body-file <scratchpad>/pr-body.md
```

Write the body to a file in your scratchpad first. Give the user the url, and
say what you couldn't verify.
