---
name: azureml-python-testing
description: Use when writing, fixing or running azureml-example's Python unit tests (azure_ai/tests), or measuring coverage. Covers the offline and live modes (GITHUB_ACTIONS and AzureMLTestBase.is_testable), how to run one module or test, the base class's point-of-use patches and its mock_azure() helper, writing offline tests, live-test naming and cleanup, the test data, CI's test workflows, and how to report results.
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

| Mode    | How                                  | What runs                                                             |
| ------- | ------------------------------------ | --------------------------------------------------------------------- |
| Offline | `GITHUB_ACTIONS=true`                | Import checks (`test_config`) and the `*_offline` tests. CI runs this |
| Live    | `GITHUB_ACTIONS` unset (`make test`) | Everything, against the subscription in `config.json`. Costs money    |

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

In offline mode `AzureMLTestBase.setUpClass()` also patches
`azure_ai.ml_studio.MLClient` and `azure_ai.ml_studio.DefaultAzureCredential`
(where `ml_studio` uses them), so a test that forgets its guard still can't
reach Azure. The patches stop in `tearDownClass()`, which runs **before** a
subclass's own cleanup code: a `tearDownClass` that would touch Azure must
return early when `not cls.is_testable`, as `test_batch_endpoint.py` does.
For extra safety in a local offline run, move `config.json` aside first.

## Writing tests

- One test module per source area: `test_<area>.py`, class
  `AzureMLTest<Area>(AzureMLTestBase)`, a docstring on every test that says
  what it proves (several existing ones are copy-pasted; don't copy those).
- **Prefer offline tests.** Most logic in `ml_studio.py` (choosing a data
  source, `from_file()`'s CSV-then-Excel fallback, default names, building a
  `ModelBatchDeployment`, argument handling in commands) can be tested with
  mocks and no network. Such a test has no `is_testable` guard.
- **Fake the workspace with `self.mock_azure()`**, which patches `MLClient`,
  `DefaultAzureCredential` and `load_dotenv` in `azure_ai.ml_studio` for one
  test, in any mode, and returns the mock `ml_client` that every helper class
  gets. Name such tests `test_<what>_offline`:

  ```python
  def test_existing_cluster_offline(self):
      ml_client = self.mock_azure()
      ml_client.compute.get.return_value = AmlCompute(name="tiny-cluster")

      cluster = AzureAIMLStudioComputeCluster(cluster_name="tiny-cluster")

      ml_client.compute.get.assert_called_once_with("tiny-cluster")
  ```

- Return real SDK entities (`AmlCompute`, `ComputeInstance`,
  `AzureBlobDatastore`, `AccountKeyConfiguration`) where `ml_studio` checks
  types with `isinstance`; a `MagicMock` fails those checks. For a `Job`,
  `test_automated_ml.py`'s `fake_job(**fields)` casts a `SimpleNamespace`.
- Any other patch goes at the point of use too: `azure_ai.ml_studio.<Name>`
  (e.g. `ContainerClient`, `kaggle`), never the name's source module, which
  `ml_studio` has already imported from.
- **Live tests** keep the `is_testable` guard, use unique names (a hash or
  timestamp suffix, as `test_batch_endpoint.py` does), and delete what they
  create in the test and again in `tearDownClass` inside `try/except`.
- `test_titanic_survival_app.py` tests the Azure Function in
  `titanic-survival-app/`: it builds a tiny ONNX model with the `onnx` package
  and imports `function_app.py` with `importlib`. See
  `azureml-function-deployment`.
- Test data goes in `azure_ai/tests/data/`. `maths.csv` is deliberately an
  Excel file with a `.csv` name, to test the Excel fallback; don't "fix" it.

## Coverage

The offline tests cover about two thirds of `ml_studio.py`; batch endpoint
creation and Kaggle downloads are the main gaps. `test_automated_ml.py` is
offline only, on purpose: a live AutoML job costs hours of cluster time.

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
  dummy `config.json`, installs `requirements/test.txt` (the runtime packages
  plus the Function's `azure-functions`, `numpy`, `onnxruntime`, and `onnx`) on
  Python 3.14 and runs `make test-ci` with `GITHUB_ACTIONS=true`. A test that
  needs a new package needs it in `requirements/in/test.in`, or CI fails on
  the import.
- A CI failure that you can't reproduce offline locally is usually an
  environment difference (package versions, Linux vs macOS). Run
  `gh run view <id> --log-failed` to read it.

## Reporting

Say which mode you ran, in which environment (venv, Docker), the `Ran N tests`
line, how many were skipped by the guard, and every failure, including
pre-existing ones.
