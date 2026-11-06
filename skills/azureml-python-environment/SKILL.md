---
name: azureml-python-environment
description: Use when azureml-example code must run locally (tests, commands, setup_config.py), when the virtual environment is missing or broken (e.g. ModuleNotFoundError), or when changing Python versions, requirements (requirements/in/*.in and the compiled requirements/*.txt), the Dockerfile, or make targets. Covers the Python 3.14 target and the Azure ML SDK v2 that makes it possible, the venv folder (which also runs the notebooks), compiling requirements with pip-compile, the Docker image, and the Makefile's destructive targets.
---

# Python Environment

## Python 3.14, on the Azure ML SDK v2

The repository runs on Python 3.14: `PYTHON := python3.14` in the Makefile,
`python:3.14` in the Dockerfile, `python-version: "3.14"` in the workflows,
`default_language_version: python: python3.14` in `.pre-commit-config.yaml`,
and `requires-python = ">=3.14"` (black `py314`) in `pyproject.toml`.

That is possible because the code uses only the Azure ML SDK v2
(`azure-ai-ml`, with `azure-identity` and `azure-storage-blob`). The v1 SDK
(`azureml-core`, `azureml-train-automl`, ...) was retired by Microsoft on
June 30, 2026, stops at Python 3.11 (`azureml-train-automl` requires `<3.12`),
and its data layer (`azureml-dataprep-rslex`) has no 3.14 or arm64 macOS
wheels. So:

- Never add an `azureml-*` package. Microsoft's `mltable` and `azureml-fsspec`
  are v2-era but depend on `azureml-dataprep` and cap at `<3.14`; that's why
  `dataset_to_dataframe()` reads data assets through `azure-storage-blob`.
- Before adding any package, check PyPI's `requires_python` and wheels for
  3.14 (`curl -s https://pypi.org/pypi/<name>/json`).
- Modern syntax is fine: `X | None`, `match`, 3.14 standard library.
- `kaggle` imports `six` without declaring it, so `base.in` lists `six`. Keep
  it until kaggle fixes its metadata.
- There are no version constraints in `constraints.in` today; the old pandas
  1.5 and scikit-learn 1.5 caps came from v1 AutoML and went with it.
- The same limit applies to AutoML models: a pickled AutoML model needs
  `azureml-automl-runtime` / `azureml-training-tabular` (`<3.12`) to load, so
  titanic-survival-app serves the model's ONNX export with `onnxruntime`
  (`azureml-function-deployment`).

## The folders on disk

| Folder          | What it is                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| `venv/`         | The project venv from `make init`: arm64 Python 3.14 (Homebrew) + `requirements/local.txt`                   |
| `venv3.12/`     | An old Python 3.12 Jupyter environment, no longer needed: `venv/` runs every notebook. Still in `.gitignore` |
| `node_modules/` | semantic-release, Prettier, commitlint (from `npm install`)                                                  |

If `venv/` is missing or broken, don't rebuild it with `make init` unasked
(see below). `pip install -r requirements/local.txt` into the existing venv is
fine. Report which environment you actually ran in.

To test in the Docker image instead (Linux, Python 3.14, the Azure CLI, no
credentials inside unless you mount them):

```bash
docker build -t localhost/azureml-example .
docker run --rm -e GITHUB_ACTIONS=true -w /dist localhost/azureml-example \
  python -m unittest discover -s azure_ai/
```

Don't mount the repository root into the container for tests: it holds
`config.json`.

## Requirements files

The `.txt` files are compiled from `requirements/in/*.in` with pip-compile
(from `pip-tools`, in `local.in`), on Python 3.14 so that only releases that
support it are picked.

| File                                      | Contents                                                                                                             | Used by                    |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `requirements/in/constraints.in`          | Global version constraints (none today), included by every `.in` file                                                | pip-compile                |
| `requirements/in/base.in`                 | Runtime packages: python-dotenv, kaggle, six, openpyxl, pandas, ..., azure-ai-ml, azure-identity, azure-storage-blob | pip-compile                |
| `requirements/in/test.in`                 | `base.in` + `../../titanic-survival-app/requirements.txt` + `onnx` (builds the Function tests' tiny model)           | pip-compile                |
| `requirements/in/local.in`                | `test.in` + pip-tools, `ipykernel`, `ipywidgets` (the notebooks), linters, mypy, tox                                 | pip-compile                |
| `requirements/in/prod.in`                 | `base.in`                                                                                                            | pip-compile                |
| `requirements/{base,test,local,prod}.txt` | The compiled, fully pinned output. Don't edit by hand. CI installs `test.txt`; the Dockerfile installs `prod.txt`    | everything                 |
| `jupyter/requirements.txt`                | The notebooks' data-science pins on 3.14, hand-kept in step with `local.txt`, for running them outside `venv/`       | other 3.14 envs            |
| `titanic-survival-app/requirements.txt`   | The Function's runtime packages (`azure-functions`, `numpy`, `onnxruntime`), pinned by hand, `#` comments only       | Azure Functions, `test.in` |

- Add a package to the right `.in` file, unpinned, with a short trailing
  comment saying what it is for, then recompile:
  `make python-requirements`, or by hand
  `pip-compile --strip-extras requirements/in/base.in -o requirements/base.txt`
  (and `test`, `local`, `prod`). Add `--upgrade` only when upgrading is the task.
- A runtime dependency goes in `base.in`; a test-only one in `test.in`; a dev
  tool in `local.in`. When you change a pin in
  `titanic-survival-app/requirements.txt` or the notebooks' packages,
  recompile and update `jupyter/requirements.txt` to match `local.txt`.
- `release.config.js` commits `requirements/**/*` on release; don't fight it.

## Makefile

| Target                                        | Effect                                                                                   |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `make` (any target)                           | Creates `.env` from `.env.example` if it's missing                                       |
| `make init`, `python-init`                    | **Runs `make clean` first**: deletes `venv/` and `__pycache__` directories. Ask first    |
| `make python-requirements`                    | Compiles `base`, `local`, `prod` and `test` `.in` files into `requirements/*.txt`        |
| `make test`                                   | Activates `venv`, runs the tests, **live** unless `GITHUB_ACTIONS=true`                  |
| `make test-ci`                                | Same without activating a venv (CI)                                                      |
| `make lint`                                   | isort, all pre-commit hooks, black, flake8, pylint on every file                         |
| `make docker-build/run`                       | Builds and runs the image                                                                |
| `make docker-push`, `release`, `docker-prune` | Pushes an image, pushes an empty commit, deletes all local Docker images. Never run them |

The host shell is zsh. It doesn't word-split variables, so pass file lists with
`xargs` or an array, not `$FILES`.
