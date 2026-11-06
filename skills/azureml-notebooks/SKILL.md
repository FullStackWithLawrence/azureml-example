---
name: azureml-notebooks
description: Use when reading, editing, adding, reviewing or running the Jupyter notebooks in azureml-example (jupyter/*.ipynb such as anomaly_detection_creditcard, House_Prices_Regression_Demo, k-means; and contributed notebooks in docs/contrib/). Covers which notebooks reach Azure or Kaggle, their kernels and requirements, editing cell by cell, outputs and what may be committed, the pickled model artifact, codespell, and reviewing a student's notebook pull request.
---

# Notebooks

The notebooks are course material: students read them top to bottom, often in
front of a class. Accuracy of the prose matters as much as the code.

## What's there

| Notebook                                                          | Kernel                   | Reaches          |
| ----------------------------------------------------------------- | ------------------------ | ---------------- |
| `jupyter/anomaly_detection_creditcard.ipynb`                      | `venv` (Python 3.9)      | Azure ML, Kaggle |
| `jupyter/House_Prices_Regression_Demo.ipynb`                      | Python 3                 | Kaggle           |
| `jupyter/k-means.ipynb`                                           | `venv3.12` (Python 3.12) | local only       |
| `docs/contrib/anomaly_detection_creditcard-01.ipynb`, `-02.ipynb` | contributors' kernels    | Azure ML, Kaggle |

- Notebooks that use `azure_ai` or `azureml` need the Python 3.9 project venv
  (`azureml-python-environment`). Pure data-science notebooks use
  `jupyter/requirements.txt` (pandas 2.x, scikit-learn 1.7, shap) in
  `venv3.12`. Don't merge the two requirement sets: their pandas versions
  conflict.
- `jupyter/isolation_forest.pkl` is a trained model the anomaly notebook loads.
  Don't regenerate or replace it unasked. Loading a pickle runs code; only
  load ones from this repository.
- `docs/contrib/` holds notebooks contributed by students, kept as they wrote
  them. Fix only what's asked.

## Rules

1. **Don't run cells that reach Azure or Kaggle** (anything constructing an
   `azure_ai.ml_studio` class, `Workspace`, `MLClient`, `AutoMLConfig`,
   `kaggle.api`) without the user's OK. See `azureml-azure-safety`. Local
   pandas and scikit-learn cells are fine to run in the matching kernel.
2. **Edit cell by cell** with the notebook editing tool. Don't rewrite or
   re-serialize a whole notebook: it churns metadata and makes the diff
   unreadable.
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
