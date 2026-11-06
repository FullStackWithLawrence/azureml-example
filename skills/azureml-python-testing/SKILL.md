---
name: azureml-python-testing
description: Use when writing, fixing or running azureml-example's Python unit tests (azure_ai/tests), or measuring coverage. Covers the offline and live modes (GITHUB_ACTIONS and AzureMLTestBase.is_testable), how to run one module or test, the base class and why its patches don't reach ml_studio, writing offline tests with mocks patched at the point of use, live-test naming and cleanup, the test data, CI's test workflows, and how to report results.
---

# Python Testing

Tests use the standard library's `unittest`, discovered from `azure_ai/`. Read
`azureml-azure-safety` before running anything live.

## Two modes

`AzureMLTestBase` (`azure_ai/tests/test_base.py`) sets

```python
is_github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
is_testable = not is_github_actions
```

and every Azure-touching test begins with

```python
if not self.is_testable:
    logger.warning("Skipping test_xxx() as the environment is not testable.")
    return
```

| Mode    | How                                  | What runs                                                                          |
| ------- | ------------------------------------ | ---------------------------------------------------------------------------------- |
| Offline | `GITHUB_ACTIONS=true`                | Import checks (`test_config.test_azureml_imports`) and offline tests. CI runs this |
| Live    | `GITHUB_ACTIONS` unset (`make test`) | Everything, against the subscription in `config.json`. Costs money                 |

**Run offline by default**, after every change, yourself:

```bash
source venv/bin/activate     # see azureml-python-environment if this venv is broken
GITHUB_ACTIONS=true python -m unittest discover -s azure_ai/
GITHUB_ACTIONS=true python -m unittest azure_ai.tests.test_dataset
GITHUB_ACTIONS=true python -m unittest azure_ai.tests.test_dataset.AzureMLTestDataset.test_file_dataset
```

Run live tests only when the user asks, naming the modules, and say what they
create (the table in `azureml-azure-safety`).

Note that a skipped test returns early and so **counts as passed**: "OK" in
offline mode proves the imports and offline tests only. Say so when reporting.

## Writing tests

- One test module per source area: `test_<area>.py`, class
  `AzureMLTest<Area>(AzureMLTestBase)`, a docstring on every test that says
  what it proves (several existing ones are copy-pasted; don't copy those).
- **Prefer offline tests.** Most logic in `ml_studio.py` (choosing a data
  source, `from_file()`'s CSV-then-Excel fallback, default names, building a
  `ModelBatchDeployment`, argument handling in commands) can be tested with
  mocks and no network. Such a test has no `is_testable` guard.
- **Patch at the point of use**, in `azure_ai.ml_studio`:

  ```python
  from unittest.mock import MagicMock, patch

  @patch("azure_ai.ml_studio.Workspace")
  @patch("azure_ai.ml_studio.Dataset")
  def test_get_or_create_existing(self, mock_dataset, mock_workspace):
      mock_workspace.from_config.return_value = MagicMock()
      ...
  ```

  `AzureMLTestBase._setup_azure_mocks()` patches `azureml.core.Workspace`,
  `azure.ai.ml.MLClient` and friends at their source. `ml_studio` has already
  imported those names, so those patches don't affect it: they don't make a
  test safe. Patch `azure_ai.ml_studio.<Name>` yourself.

- Also patch `azure_ai.ml_studio.load_dotenv` when a test must not read the
  user's `.env`, and `azure_ai.ml_studio.kaggle` for Kaggle downloads.
- **Live tests** keep the `is_testable` guard, use unique names (a hash or
  timestamp suffix, as `test_batch_endpoint.py` does), and delete what they
  create in the test and again in `tearDownClass` inside `try/except`.
- Test data goes in `azure_ai/tests/data/`. `maths.csv` is deliberately an
  Excel file with a `.csv` name, to test the Excel fallback; don't "fix" it.

## Coverage

No coverage tool is configured. If the user asks, use it ad hoc without adding
it to the requirements unasked:

```bash
pip install coverage
GITHUB_ACTIONS=true coverage run -m unittest discover -s azure_ai/ && coverage report -m --include='azure_ai/*'
```

## CI

- `.github/workflows/test.yml` (pushes to `main` and `next`, PRs touching
  `*.py`) and `pullRequestController.yml` (PRs from forks too, via
  `pull_request_target`) call `.github/actions/tests/python`, which writes a
  dummy `config.json`, installs `requirements/prod.txt` on Python 3.9 and runs
  `make test-ci` with `GITHUB_ACTIONS=true`.
- A CI failure that you can't reproduce offline locally is usually an
  environment difference (package versions, Python 3.9 vs local). Run
  `gh run view <id> --log-failed` to read it.

## Reporting

Say which mode you ran, in which environment (venv, Docker), the `Ran N tests`
line, how many were skipped by the guard, and every failure, including
pre-existing ones.
