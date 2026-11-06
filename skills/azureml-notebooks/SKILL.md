---
name: azureml-notebooks
description: Use when reading, editing, adding, reviewing or running the Jupyter notebooks in azureml-example (jupyter/*.ipynb such as anomaly_detection_creditcard, House_Prices_Regression_Demo, k-means; and contributed notebooks in docs/contrib/). Covers which notebooks reach Azure or Kaggle, their kernels and requirements, editing cell by cell, outputs and what may be committed, the pickled model artifact, codespell, and reviewing a student's notebook pull request.
---

# Notebooks

The notebooks are course material: students read them top to bottom, often in
front of a class. Accuracy of the prose matters as much as the code.

## What's there

| Notebook                                             | Kernel                            | Reaches                                                   |
| ---------------------------------------------------- | --------------------------------- | --------------------------------------------------------- |
| `jupyter/anomaly_detection_creditcard.ipynb`         | `venv (3.14.9)`, the project venv | Azure ML                                                  |
| `jupyter/House_Prices_Regression_Demo.ipynb`         | Google Colab (`Python 3`)         | Kaggle                                                    |
| `jupyter/k-means.ipynb`                              | `venv (3.14.9)`                   | local only                                                |
| `docs/contrib/anomaly_detection_creditcard-01.ipynb` | `venv (3.14.9)`                   | Azure ML (fallback when `data/creditcard.csv` is missing) |
| `docs/contrib/anomaly_detection_creditcard-02.ipynb` | `venv (3.14.9)`                   | none (markdown only)                                      |

- Every notebook that runs locally uses the Python 3.14 project venv
  (`azureml-python-environment`), which has the Azure ML SDK v2 plus
  `ipykernel`, `ipywidgets`, matplotlib, seaborn and shap. `venv3.12` is no
  longer needed. `jupyter/requirements.txt` holds the same data-science pins
  (kept in step with `requirements/local.txt`) for running the notebooks in
  some other 3.14 environment. `House_Prices_Regression_Demo.ipynb` runs in
  Google Colab (`google.colab`), which picks its own Python.
- The two credit-card notebooks use the v2 SDK through the repository's
  helpers: they put the repository root on `sys.path`, then
  `AzureAIMLAssetsDataset(dataset_name='creditcard_fraud').dataset_to_dataframe()`.
  The v2 SDK alone can't read a data asset into pandas on 3.14
  (`ml_client.data` has no download, and `mltable` / `azureml-fsspec` cap
  below 3.14). `jupyter/anomaly_detection_creditcard.ipynb` registers its model
  with `dataset.ml_client.models.create_or_update(Model(..., type=AssetTypes.CUSTOM_MODEL))`.
  `MLClient.from_config()` finds `config.json` in the notebook's folder or a
  parent (the repository root, locally).
- `jupyter/isolation_forest.pkl` is a trained model; the anomaly notebook
  writes it with `joblib.dump` before registering it. Don't regenerate or
  replace it unasked. Loading a pickle runs code; only load ones from this
  repository.
- `docs/contrib/` holds notebooks contributed by students, kept as they wrote
  them. Fix only what's asked.

## Rules

1. **Don't run cells that reach Azure or Kaggle** (anything constructing an
   `azure_ai.ml_studio` class, `MLClient`, `ml_client.*`, `kaggle.api`)
   without the user's OK. See `azureml-azure-safety`. Local
   pandas and scikit-learn cells are fine to run in the matching kernel.
2. **Edit cell by cell** with the notebook editing tool. Don't rewrite or
   re-serialize a whole notebook: it churns metadata and makes the diff
   unreadable. The tool itself causes churn: it saves an edited cell's
   `source` as one string instead of a list of lines, and clears an edited
   code cell's `outputs` and `execution_count`. After editing, normalize every
   cell's `source` back to a list of lines (`str.splitlines(keepends=True)`)
   and write the notebook with `json.dumps(nb, indent=1, ensure_ascii=False)`
   plus a trailing newline. For contributed notebooks, restore an edited
   cell's outputs from `git show HEAD:<path>` when they're still valid for
   the new code. Then check that `git diff` shows only the real changes. Edit
   kernel metadata as a text replacement in the file.
3. **Commit notebooks without outputs**, as the `jupyter/` notebooks are now,
   unless the user wants outputs kept for a lesson. Outputs can leak a
   subscription id, workspace details, file paths with a user name, or Kaggle
   data. Check outputs for those before any notebook is committed, contributed
   ones included.
4. **Keep the teaching voice.** Markdown cells explain why, for the audience
   the notebook states (some are written for executives, some for students).
   Correct wrong statements (metrics, precision and recall, what a plot
   shows), but don't flatten the explanations into terse comments.
5. **No secrets in cells.** Credentials come from `config.json`, `.env` and
   `~/.kaggle/kaggle.json`, never literals.
6. A code change to `azure_ai/` that changes a signature a notebook calls
   means updating that notebook in the same change. Search with
   `grep -l "<name>" jupyter/*.ipynb docs/contrib/*.ipynb`.

## Lint

pre-commit's codespell checks notebooks (`.codespellrc` skips
`jupyter/anomaly_detection_creditcard.ipynb`), and `check-json` checks that
they parse. black and isort don't touch `.ipynb` here.

## Reviewing a student's notebook pull request

Check that it runs in its stated kernel (by reading it, or running local-only
cells), that the prose matches the code and the results, that no outputs leak
anything (rule 3), that generated HTML or large data files weren't added, and
that it's in the right folder. Report findings; don't push changes to their
branch.
