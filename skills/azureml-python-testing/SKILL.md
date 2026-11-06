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

In offline mode `AzureMLTestBase.setUpClass()` also patches `MLClient` and
`DefaultAzureCredential` in `azure_ai.ml_studio.workspace`, and
`DefaultAzureCredential` in `azure_ai.ml_studio.assets_dataset` (where they
are used), so a test that forgets its guard still can't reach Azure. The patches stop in `tearDownClass()`, which runs **before** a
subclass's own cleanup code: a `tearDownClass` that would touch Azure must
return early when `not cls.is_testable`, as `test_batch_endpoint.py` does.
For extra safety in a local offline run, move `config.json` aside first.

## Writing tests

- One test module per source area: `test_<area>.py`, class
  `AzureMLTest<Area>(AzureMLTestBase)`, a docstring on every test that says
  what it proves (several existing ones are copy-pasted; don't copy those).
- **Prefer offline tests.** Most logic in `azure_ai/ml_studio/` (choosing a data
  source, `from_file()`'s CSV-then-Excel fallback, default names, building a
  `ModelBatchDeployment`, argument handling in commands) can be tested with
  mocks and no network. Such a test has no `is_testable` guard.
- **Fake the workspace with `self.mock_azure()`**, which patches `MLClient`,
  `DefaultAzureCredential` and `load_dotenv` in `azure_ai.ml_studio.workspace`
  (and `DefaultAzureCredential` in `assets_dataset`) for one test, in any
  mode, and returns the mock `ml_client` that every helper class gets. Name such tests `test_<what>_offline`:

  ```python
  def test_existing_cluster_offline(self):
      ml_client = self.mock_azure()
      ml_client.compute.get.return_value = AmlCompute(name="tiny-cluster")

      cluster = AzureAIMLStudioComputeCluster(cluster_name="tiny-cluster")

      ml_client.compute.get.assert_called_once_with("tiny-cluster")
  ```

- Return real SDK entities (`AmlCompute`, `ComputeInstance`,
  `AzureBlobDatastore`, `AccountKeyConfiguration`) where the helpers check
  types with `isinstance`; a `MagicMock` fails those checks. For a `Job`,
  `test_automated_ml.py`'s `fake_job(**fields)` casts a `SimpleNamespace`.
- Any other patch goes at the point of use too: the module that uses the
  name, e.g. `azure_ai.ml_studio.assets_dataset.ContainerClient` or
  `azure_ai.ml_studio.assets_dataset.kaggle`. Never patch the name's source
  module or the package's re-export: the module has already imported it.
  A command imports its class from the package at run time, so
  `test_commands.py` patches the class there (`azure_ai.ml_studio.<Class>`).
- Pure helpers in `azure_ai/ml_studio/utils.py` need no mocks at all; see
  `test_utils.py`.
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

**Target:** at least 90% for each module, measured by the offline suite
(`make coverage`). Higher is better; the suite is at 100% for every module
in `azure_ai/` and `titanic-survival-app/`, so a change should keep it there
or say why not. Large blocks of misses in a module, or one module far below
the target, are not acceptable even when the total looks fine.

To raise it efficiently:

1. Work from the `coverage report -m` output, and rank modules by missed
   lines, not percentage.
2. Test behavior through the public interface: the helper class's
   constructor and methods, the command run as `python3 -m` (see
   `test_commands.py`, which uses `runpy`), the Function's `predict()`. Don't
   call private helpers just to touch lines.
3. An error path counts: give the mock a `side_effect` and assert the error
   is raised, or the warning logged (`assertLogs("azure_ai.ml_studio")`).
4. A line no input can reach is dead code. Remove it rather than test it.

`test_automated_ml.py` is offline only, on purpose: a live AutoML job costs
cluster time.

`coverage` is in `requirements/in/test.in`, and its options are in
`pyproject.toml`'s `[tool.coverage]` sections: it measures `azure_ai/` (not
`azure_ai/tests/`) and `titanic-survival-app/`. `make coverage` runs the
offline suite under it, with `GITHUB_ACTIONS=true` set, and prints the
per-file report with missing lines. It's safe to run: nothing reaches Azure.

CI uploads `coverage.xml` to Codecov under the `python` flag (`codecov.yml`),
which draws the README's coverage badge from `main`. Coverage is
informational: a pull request that lowers it still passes.

## CI

- `.github/workflows/test.yml` runs on every pull request (forks included,
  via `pull_request`, without secrets) and on pushes to `main` and `next`. It
  installs `requirements/test.txt` (the runtime packages plus the Function's
  `azure-functions`, `numpy`, `onnxruntime`, and `onnx`, and `coverage`) on
  Python 3.14 and runs `make test-ci` with `GITHUB_ACTIONS=true`: the tests
  under coverage, the report, and `coverage.xml`. The job summary shows the
  per-file report. It needs no `config.json` or Kaggle key (kaggle 2.x
  tolerates a missing key on import). A test that needs a new package needs
  it in `requirements/in/test.in`, or CI fails on the import.
- A CI failure that you can't reproduce offline locally is usually an
  environment difference (package versions, Linux vs macOS). Run
  `gh run view <id> --log-failed` to read it.

## Reporting

Say which mode you ran, in which environment (venv, Docker), the `Ran N tests`
line, how many were skipped by the guard, and every failure, including
pre-existing ones.
