---
name: azureml-development
description: Start here for any change to the azureml-example repository (the azure_ai helper classes and commands, their tests, the titanic-survival-app Azure Function, the Jupyter notebooks, the Markdown docs, the GitHub workflows, commits and pull requests). Gives the non-negotiable rules, the pattern-first workflow, known oddities in the repository, and an index of the narrower azureml-* skills to load for each kind of work.
---

# azureml-example Development

azureml-example demonstrates Azure Machine Learning's Automated ML service from
Python. It is a teaching repository (university courses, the Full Stack With
Lawrence YouTube channel), so the code is meant to be read: small, explicit,
well commented, and consistent from one class to the next.

## Repository map

| Path                                        | What lives there                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `azure_ai/ml_studio.py`                     | Every helper class. One file, one class per Azure ML concept                                   |
| `azure_ai/commands/<name>.py`               | Entry points: `python3 -m azure_ai.commands.<name> ...`                                        |
| `azure_ai/tests/`                           | `unittest` tests; `test_base.py` holds `AzureMLTestBase`                                       |
| `azure_ai/tests/data/maths.csv`             | Test data. It is really an Excel workbook named `.csv`, on purpose                             |
| `azure_ai/__version__.py`                   | Written by CI (`semanticVersionBump.yml`). Never edit it                                       |
| `titanic-survival-app/`                     | Azure Function (Python 3.14) serving the AutoML model's ONNX export (`automl-model/`, ignored) |
| `jupyter/`, `docs/contrib/`                 | Course notebooks and contributed notebooks                                                     |
| `docs/*.md`                                 | Setup, configuration, AutoML, deployment and Azure CLI guides                                  |
| `requirements/`                             | `in/*.in` sources, compiled by pip-compile into `base/test/local/prod.txt`                     |
| `setup_config.py`                           | Interactive helper that installs `config.json`                                                 |
| `Dockerfile`                                | `python:3.14` image with the Azure CLI and `requirements/prod`                                 |
| `.pre-commit-config.yaml`                   | The lint and format standard                                                                   |
| `.github/workflows/`                        | `test.yml` (tests, coverage), `release.yml` (semantic-release), version bumps, auto-assign     |
| `codecov.yml`                               | Codecov settings for the coverage that `test.yml` uploads (the README badge)                   |
| `commitlint.config.js`, `release.config.js` | Commit message rules and semantic-release (branches main, beta, alpha)                         |

## Non-negotiable rules

1. **Never execute git write commands.** "Write a git commit" means output the
   command text. See `azureml-git-commits`.
2. **Never create billable Azure resources on your own.** Compute clusters,
   batch endpoints and deployments, AutoML runs and Function Apps cost the
   user money, and the local `config.json` and `az login` reach a real
   subscription. See `azureml-azure-safety`.
3. **Run the offline tests yourself** after every change:
   `GITHUB_ACTIONS=true python -m unittest discover -s azure_ai/`. Run live
   tests only when the user asks. See `azureml-python-testing`.
4. **Python 3.14 and the Azure ML SDK v2 (`azure-ai-ml`) only.** Never add an
   `azureml-*` (v1) package. See `azureml-python-environment`.
5. **The pre-commit hooks are the lint standard.** See `azureml-code-quality`.
6. **The repository is public.** No credentials, subscription or tenant ids,
   Kaggle keys or personal data anywhere, including notebook outputs.
7. **Notebooks are course material.** See `azureml-notebooks`.

## Pattern first

Before writing anything, find the closest existing example and copy its shape.

- **A helper class** in `ml_studio.py` subclasses `AzureAIMLWorkspace`, takes
  its resource name first and options as keyword arguments, keeps a class
  attribute for the Azure object (`cluster`, `endpoint`, `dataset`), and
  offers `get_default_name(ml_client)` (a classmethod), `get_or_create()` and,
  where it makes sense, `delete()`. See `azureml-ml-studio`.
- **A command** in `azure_ai/commands/` is a module with a docstring that
  includes a `usage:` line, `argparse` arguments, and an
  `if __name__ == "__main__":` block that prints a result. Add it to
  `commands/help.py` and to the README's Usage section.
- **A test** subclasses `AzureMLTestBase`, starts every Azure-touching test
  with the `is_testable` guard, and deletes what it creates.

## Known oddities (don't copy, don't "fix" unasked)

These came from the template the repository was made from. Leave them alone
unless the task is about them, and don't spread them to new files.

- `docs/CONTRIBUTING.md` and `docs/12-FACTOR.md` mention Smarter, Terraform,
  LangChain, AWS and docker-compose, none of which this repository uses.
- `.github/actions/merge-branch/action.yml` is a composite action that no
  workflow calls.
- Docstrings in `ml_studio.py` sometimes list `Args:` that a method no longer
  takes. When you change a method, correct its docstring.

When one of these bites the task at hand, fix it as its own change and say so.

## Which skill to load

| Task                                                                 | Skill                         |
| -------------------------------------------------------------------- | ----------------------------- |
| Anything that may reach Azure or Kaggle: code, tests, CLI, notebooks | `azureml-azure-safety`        |
| The virtual environment, Python version, requirements, Docker image  | `azureml-python-environment`  |
| Writing or running tests                                             | `azureml-python-testing`      |
| The helper classes in `ml_studio.py` and the command modules         | `azureml-ml-studio`           |
| Lint, format, docstrings, pre-commit                                 | `azureml-code-quality`        |
| Writing a commit message                                             | `azureml-git-commits`         |
| Creating or describing a pull request                                | `azureml-pull-requests`       |
| The Jupyter notebooks in `jupyter/` and `docs/contrib/`              | `azureml-notebooks`           |
| `titanic-survival-app`, model download, Azure Functions              | `azureml-function-deployment` |

## Definition of done

- The code follows an existing pattern, and reads well to a student: simple
  over compact, about one operation per line (see `azureml-code-quality`).
- The offline tests pass, you ran them, and new behavior has a test (offline
  where possible, live-guarded where it must call Azure).
- The pre-commit hooks pass on the changed files.
- Public classes and functions have Google-style docstrings; a user-facing
  change is reflected in the README or the relevant `docs/` guide.
- No Azure resource was created without the user's say-so, and none was left
  behind.
- Your summary says what changed, what you verified, and what you skipped.
