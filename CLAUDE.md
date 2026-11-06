# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repository.

azureml-example is a teaching repository from Full Stack With Lawrence. It
shows how to drive Azure Machine Learning's Automated ML service from Python:
connect to a workspace, register datasets (from a DataFrame, a local file or
Kaggle), create compute clusters, run AutoML experiments, register the best
model, and serve it from a batch endpoint or an Azure Function. It is used in
university courses and YouTube tutorials, so clarity matters as much as
correctness.

Detailed know-how lives in the Agent Skills under `skills/` (symlinked to
`.claude/skills`). **Load `azureml-development` first for any change.** Its
table says which narrower `azureml-*` skill covers each kind of work. This file
holds only what applies to nearly every task.

## Hard rules

1. **Never run git write commands.** "Write a git commit" means print the
   command as text. Never run `git commit`, `git add`, `git reset`, `git push`
   or anything else that changes the index or history. Format and authorship
   are in `azureml-git-commits`.
2. **Azure costs real money.** `config.json`, `az login` and
   `DefaultAzureCredential` reach a real, paid Azure subscription. Creating a
   compute cluster, a batch endpoint, a deployment, an AutoML run or a
   Function App bills the user's account. Don't run code, tests, commands or
   notebook cells that create Azure resources unless the user asks for that
   specific thing; hand them the command instead. See `azureml-azure-safety`.
3. **Run the tests yourself, offline by default.** After any change, run
   `GITHUB_ACTIONS=true python -m unittest discover -s azure_ai/`, which skips
   every test that talks to Azure. Run the live tests only when the user asks.
   Never ask the user to re-run tests. Report failures honestly, including
   pre-existing ones. See `azureml-python-testing`.
4. **Python 3.14 and the Azure ML SDK v2 (`azure-ai-ml`) only.** The v1 SDK
   (`azureml-*`) is retired and doesn't support 3.14; don't add it back. Keep
   the Python version in the Makefile, Dockerfile, workflows, pre-commit
   config and pyproject in step, and regenerate `requirements/*.txt` from
   `requirements/in/*.in` with `make python-requirements` rather than editing
   pins by hand. See `azureml-python-environment`.
5. **The pre-commit hooks are the lint standard** (black, isort, flake8,
   bandit, codespell, pyupgrade, Prettier, and the pre-commit-hooks checks).
   See `azureml-code-quality`.
6. **This repository is public.** No subscription ids, tenant ids, client
   secrets, Kaggle keys, storage keys or personal data in code, tests,
   notebooks, docs or skills. `config.json`, `.env` and `kaggle.json` are
   git-ignored and stay that way.
7. **Notebooks are course material.** Don't run notebook cells that reach
   Azure or Kaggle, don't strip or regenerate outputs that a lesson relies on,
   and edit cells rather than rewriting whole notebooks. See
   `azureml-notebooks`.

## Repository map

| Path                    | Contents                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `azure_ai/ml_studio/`   | The helper classes, one module each: workspace, dataset, compute cluster, batch endpoint, AutoML |
| `azure_ai/commands/`    | Small command-line entry points, run as `python3 -m azure_ai.commands.<name>`                    |
| `azure_ai/tests/`       | `unittest` tests, `AzureMLTestBase`, test data in `tests/data/`                                  |
| `titanic-survival-app/` | An Azure Function that serves a downloaded AutoML model over HTTP                                |
| `jupyter/`              | Course notebooks, with their own `requirements.txt`                                              |
| `docs/`                 | Markdown guides: account setup, workspace config, AutoML, deployment, Azure CLI                  |
| `requirements/`         | `base.txt` (shared pins), `local.txt` (dev tools), `prod.txt` (Docker and CI)                    |
| `.github/workflows/`    | Tests on pull requests, semantic-release on `main`, version bumps on alpha, beta                 |
| `skills/`               | Agent Skills for developing this repository                                                      |

## Common commands

```bash
make init            # build ./venv with Python 3.14 and requirements/local.txt
make test            # all tests, LIVE against Azure when config.json is present

# Offline tests (what CI runs; skips everything that calls Azure)
source venv/bin/activate
GITHUB_ACTIONS=true python -m unittest discover -s azure_ai/

# Lint the changed files (zsh doesn't word-split $FILES, so use xargs)
git diff --name-only | xargs pre-commit run --files
```

- `make test` runs the **live** suite whenever `GITHUB_ACTIONS` isn't `true`:
  it creates and deletes datasets, a compute cluster and batch endpoints. Read
  `azureml-azure-safety` before running it.
- `make init` and `make python-init` start with `make clean`, which deletes
  `venv/`, `node_modules/` and `package-lock.json`. Ask before running them.
- `make release` makes an empty commit and pushes it. Never run it.

## Conventions

- **Pattern first.** Before adding a helper, a command or a test, read the
  existing ones in full and copy their structure: the `get_or_create()` /
  `get_default_name()` / `delete()` shape of the helper classes, the
  `argparse` command modules, the `is_testable` guard in tests.
- **Logging** uses `logger = getLogger(__name__)` and lazy %-style arguments
  (`logger.info("Found cluster: %s", name)`), never f-strings.
- **Docstrings** use the Google style already in `azure_ai/ml_studio/` (`Args:`,
  `Returns:`, `Raises:`), and every public class and function has one.
- **Docs** are Markdown in `docs/`. Weave new material into the existing guide
  it belongs to rather than adding a "what's new" section.
- **Commits** are conventional commits, checked by commitlint and read by
  semantic-release, which versions `main`, `beta` and `alpha`.

## Definition of done

- The change follows an existing pattern.
- Tests cover it, and you ran the offline suite (and said whether you ran any
  live tests).
- Pre-commit hooks pass on the changed files.
- Nothing billable was created in Azure without the user's say-so, and
  anything a test created was deleted.
- Your summary says what changed, what you verified, and what you skipped.
