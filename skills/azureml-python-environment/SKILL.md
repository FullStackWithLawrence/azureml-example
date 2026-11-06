---
name: azureml-python-environment
description: Use when azureml-example code must run locally (tests, commands, setup_config.py), when the virtual environment is missing or broken (e.g. "bad CPU type in executable", ModuleNotFoundError azureml), or when changing Python versions, requirements/*.txt pins, the Dockerfile, or make targets. Covers the Python 3.9 pin and why, the venv and venv3.12 folders, rebuilding the venv on Apple silicon without deleting anything unasked, the three requirements files, the Docker image, and the Makefile's destructive targets.
---

# Python Environment

## Python 3.9, on purpose

The v1 SDK (`azureml-core`, `azureml-train-automl`, `azureml-widgets`,
`azureml-pipeline-core`, all pinned at 1.60.x) supports old Pythons only, and
AutoML pins old pandas and scikit-learn (`pandas==1.5.3`,
`scikit-learn==1.5.1`). The README warns students not to upgrade. So:

- Don't change `PYTHON = python3.9` in the Makefile, `python:3.9` in the
  Dockerfile, `python-version: "3.9"` in the workflows, or
  `default_language_version: python: python3.9` in `.pre-commit-config.yaml`.
- Write code that runs on 3.9: `Optional[...]` and `Union[...]` from `typing`
  rather than `X | None`, no `match`, no 3.10+ standard library functions.
  pyupgrade may suggest newer syntax only where 3.9 allows it.
- Don't upgrade a pinned package across a major version (pandas 1 to 2 breaks
  AutoML), and treat dependabot's major bumps of `azureml-*`, pandas,
  scikit-learn or scipy as suspicious.

## The folders on disk

| Folder          | What it is                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------ |
| `venv/`         | The project venv from `make init`, Python 3.9 + `requirements/local.txt`                   |
| `venv3.12/`     | A separate Jupyter environment for the notebooks (pandas 2.x, no `azureml`). Not for tests |
| `node_modules/` | semantic-release, Prettier, commitlint (from `npm install`)                                |

**`venv/bin/python` says "bad CPU type in executable"** on the user's Apple
silicon Mac: the venv was built from an Intel (x86_64) Python 3.9 in
`/usr/local/bin`, and Rosetta isn't installed. Don't delete or rebuild it on
your own. Tell the user, and offer one of these:

```bash
# A: an arm64 Python 3.9 through pyenv (installed at /opt/homebrew/bin/pyenv)
pyenv install 3.9
$(pyenv prefix 3.9)/bin/python -m venv venv39
source venv39/bin/activate && pip install -r requirements/local.txt

# B: run in the Docker image, which is linux/amd64 Python 3.9
make docker-build
docker run --rm -e GITHUB_ACTIONS=true -v "$PWD":/dist -w /dist \
  localhost/azureml-example python -m unittest discover -s azure_ai/
```

A new venv folder needs a `.gitignore` entry (only `venv`, `.venv` and
`venv3.12` are ignored today). Whether some `azureml-*` 1.60 wheels install on
macOS arm64 at all is not guaranteed; if pip fails, option B is the fallback.
Report which environment you actually ran in.

## Requirements files

| File                                    | Contents                                                                                                                             | Used by              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| `requirements/base.txt`                 | Runtime pins: python-dotenv, pandas, kaggle, azureml-\*, azure-ai-ml, ...                                                            | everything           |
| `requirements/local.txt`                | `-r base.txt` + black, flake8, isort, pylint, bandit, pre-commit, mypy, tox, codespell, `azure.functions` (for linting the Function) | `make init`          |
| `requirements/prod.txt`                 | `-r base.txt`                                                                                                                        | Dockerfile, CI tests |
| `jupyter/requirements.txt`              | The notebooks' own pins                                                                                                              | `venv3.12`           |
| `titanic-survival-app/requirements.txt` | The Function's runtime packages, unpinned                                                                                            | Azure Functions      |

- Pin exact versions (`==`) with a short trailing comment saying what the
  package is for, as `base.txt` does.
- A runtime dependency goes in `base.txt`; a dev tool in `local.txt`.
- `release.config.js` commits `requirements/**/*` on release; don't fight it.

## Makefile

| Target                                        | Effect                                                                                        |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `make` (any target)                           | Creates `.env` with defaults if it's missing                                                  |
| `make init`, `python-init`                    | **Runs `make clean` first**: deletes `venv/`, `node_modules/`, `package-lock.json`. Ask first |
| `make test`                                   | Activates `venv`, runs the tests, **live** unless `GITHUB_ACTIONS=true`                       |
| `make test-ci`                                | Same without activating a venv (CI)                                                           |
| `make lint`                                   | isort, all pre-commit hooks, black, flake8, pylint on every file                              |
| `make docker-build/run`                       | Builds and runs the image                                                                     |
| `make docker-push`, `release`, `docker-prune` | Pushes an image, pushes an empty commit, deletes all local Docker images. Never run them      |

The host shell is zsh. It doesn't word-split variables, so pass file lists with
`xargs` or an array, not `$FILES`.
