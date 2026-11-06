---
name: azureml-ml-studio
description: Use when adding or changing a helper class in the azure_ai/ml_studio package (AzureAIMLWorkspace, AzureAIMLAssetsDataset, AzureAIMLStudioComputeCluster, AzureAIMLStudioAssetsBatchEndpoint, AzureAIMLStudioAuthoringAutomatedML) or a command module in azure_ai/commands/. Covers the class hierarchy and naming, the get_default_name / get_or_create / delete pattern, the Azure ML SDK v2 (azure-ai-ml) objects each class wraps, logging and error handling, and the checklist for a new class or command.
---

# The ml_studio Helpers and Commands

## The classes

They live in the `azure_ai/ml_studio/` package, one module per class. Names
follow the Azure ML Studio menu: `AzureAIMLStudio<Menu><Item>` or
`AzureAIML<Menu><Item>`.

| Module                         | Class                                 | Studio area        | Main methods                                                                                 |
| ------------------------------ | ------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------- |
| `workspace.py`                 | `AzureAIMLWorkspace`                  | Workspace          | `connect_workspace()` via `MLClient.from_config()`; base of all the others                   |
| `assets_dataset.py`            | `AzureAIMLAssetsDataset`              | Assets > Data      | `get_or_create()`, `from_file()`, `from_kaggle()`, `dataset_to_dataframe()`, `delete()`      |
| `ml_studio_compute_cluster.py` | `AzureAIMLStudioComputeCluster`       | Compute > Clusters | `get_default_name()`, `get_or_create()`, `create()`                                          |
| `assets_batch_endpoint.py`     | `AzureAIMLStudioAssetsBatchEndpoint`  | Assets > Endpoints | `get_or_create()`, `_create_batch_deployment()`, `invoke_batch_job()`, `delete()`            |
| `authoring_automatedml.py`     | `AzureAIMLStudioAuthoringAutomatedML` | Authoring > AutoML | `run_automl_experiment()`, `get_best_model()`, `register_best_model()`, `show_run_details()` |

What the classes share lives beside them:

- `const.py`: `MLTABLE_TEMPLATE`, `AUTOML_TASKS`, `MODEL_FORMATS`.
- `exceptions.py`: `is_not_found(error)`, which recognizes each way the SDK
  reports a missing asset (`ResourceNotFoundError`, or a
  `ValidationException` of type `RESOURCE_NOT_FOUND`, possibly wrapped in an
  `MlException`). Use it in every `get_or_create()` lookup.
- `utils.py`: pure helpers with no Azure calls, `write_mltable_folder()` and
  `parse_datastore_path()`.
- `__init__.py` re-exports every public name, so code imports from the
  package: `from azure_ai.ml_studio import AzureAIMLAssetsDataset`. Modules
  inside the package import each other relatively (`from .workspace import
AzureAIMLWorkspace`).

**One SDK: v2 (`azure-ai-ml`).** `AzureAIMLWorkspace.__init__()` creates
`self.ml_client` with `MLClient.from_config(credential=DefaultAzureCredential())`
(reads `config.json`) and `self.workspace`, the v2 `Workspace` entity (may be
`None` by type). Every subclass uses the inherited `self.ml_client`; don't
build a second `MLClient`. Never import the retired v1 SDK (`azureml.*`): it
doesn't install on Python 3.14 (`azureml-python-environment`).

How each Azure concept maps to v2:

- **Data** is a v2 `Data` asset of type `AssetTypes.MLTABLE`: a folder with
  the CSV file plus an `MLTable` file (`MLTABLE_TEMPLATE`), uploaded by
  `ml_client.data.create_or_update()`. AutoML v2 only accepts MLTable for
  tabular training data. `get_or_create()` looks up `label="latest"`.
  `dataset_to_dataframe()` parses the asset's `azureml://.../datastores/<name>/paths/<path>`
  and downloads the CSV with `azure-storage-blob`, via
  `get_datastore_container()` (datastore account key or SAS, else
  `DefaultAzureCredential`), because `mltable` and `azureml-fsspec` don't
  support 3.14. `delete()` **archives** all versions
  (`ml_client.data.archive()`): v2 can't delete a data asset.
- **Compute** is `AmlCompute` via `ml_client.compute`. `get_or_create()`
  gets the cluster, calls `create()` (which waits on
  `begin_create_or_update().result()`) when it's not found, and raises
  `TypeError` if the result isn't an `AmlCompute` (e.g. the name belongs to a
  compute instance).
- **AutoML** jobs come from `AUTOML_TASKS` (`automl.classification`,
  `regression`, `forecasting`; forecasting takes `time_column_name`), with
  `set_limits()` and
  `set_training(enable_onnx_compatible_models=...)`, on by default so that
  every trial also saves `outputs/model.onnx` for titanic-survival-app
  (`azureml-function-deployment`). Submitted by `ml_client.jobs.create_or_update()`, which
  returns at once. `get_best_model()` waits with `ml_client.jobs.stream()`,
  then reads the best trial's name from the parent job's
  `automl_best_child_run_id` tag (or property) and returns that child `Job`.
  That lookup has not yet been checked against a live job.
  `register_best_model(model_format=...)` looks the format up in
  `MODEL_FORMATS`: `"mlflow"` (default) registers
  `azureml://jobs/<best>/outputs/artifacts/outputs/mlflow-model/` as an
  `MLFLOW_MODEL`, which batch endpoints serve without a scoring script;
  `"onnx"` registers `.../outputs/model.onnx` as a `CUSTOM_MODEL`, which the
  Function serves on Python 3.14. The ONNX artifact path, like the best-trial
  lookup, hasn't been checked against a live job.
  `show_run_details()` returns the job's `studio_url` (the v1 Jupyter widget
  has no v2 equivalent).

## The pattern

```python
class AzureAIMLStudio<Thing>(AzureAIMLWorkspace):
    """
    Azure AI ML Studio <Menu> -> <Item> class.
    Class to encapsulate Azure ML <thing> functionality.
    """

    <thing>: <AzureType> | None = None

    def __init__(self, <thing>_name: str | None = None, *args, <options with defaults>, **kwargs):
        super().__init__(*args, **kwargs)
        self.<thing>_name = <thing>_name or AzureAIMLStudio<Thing>.get_default_name(self.ml_client)
        ...
        self.<thing> = self.get_or_create()

    @classmethod
    def get_default_name(cls, ml_client: MLClient) -> str:
        """Name of the first <thing> in the workspace. Raises ValueError if none."""

    def get_or_create(self) -> <AzureType>:
        """Return the cached object, else the existing one by name, else create it."""

    def delete(self):
        """Delete (or archive) it; log a warning, don't raise, when it doesn't exist."""
```

- Constructing an instance **creates the resource if it doesn't exist**. That
  is the teaching point, and the reason tests and commands cost money. Keep
  creation in `get_or_create()`, never in a getter or a property.
- Options are keyword arguments with sensible, cheap defaults (small VM sizes,
  `min_nodes=0`, short timeouts).
- Logging: `logger.info("Found existing cluster: %s", self.cluster_name)`,
  lazy %-style arguments, no f-strings in log calls, no secrets.
- Error handling: fall through to creation only when the lookup says the
  resource doesn't exist, which `is_not_found(e)` (`exceptions.py`) decides;
  a lookup by `label="latest"` doesn't raise `ResourceNotFoundError`. For
  anything else log `logger.error("Failed to ... %s: %s", name, e)` and
  re-raise, so an auth or network error never creates a second resource. A
  deliberate broad `except Exception` carries `# pylint: disable=W0718`.
- Long-running Azure calls wait on a poller's `.result()` (or
  `ml_client.jobs.stream()` for jobs), so a student sees progress.
- Type-check with the SDK's own types: pyright (what Pylance runs in the
  editor) and mypy should report no errors on `azure_ai/ml_studio/`. Narrow
  `X | None` values and base types (`Compute`, `Datastore`) with a check that
  raises a clear error, rather than `cast()` or `# type: ignore`.
- Google-style docstrings (`Args:`, `Returns:`, `Raises:`) that match the
  actual signature.

## Adding a class

1. Add it as its own module in `azure_ai/ml_studio/`, named after its studio
   area (e.g. `assets_models.py`), following the pattern, and re-export it
   from `__init__.py` (the import and `__all__`).
2. Add `azure_ai/tests/test_<thing>.py`: offline tests that fake the
   workspace with `self.mock_azure()`, plus guarded live tests that clean up
   (`azureml-python-testing`).
3. Add a command if a student would run it from the shell (below).
4. List it in the README's bullet list of classes.
5. If it creates something billable, add it to the cost table in
   `azureml-azure-safety`.

## Adding a command

```python
"""
Lawrence McDaniel
https://lawrencemcdaniel.com

<One line: what it does.>
usage:
    python3 -m azure_ai.commands.<name> <args>
"""

import argparse

from azure_ai.ml_studio import <Class>


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="<what it does>")
    parser.add_argument("<arg>", type=str, help="<help>, e.g., <example>")
    args = parser.parse_args()
    ...
    print(<result>)
```

Then add its usage line to `azure_ai/commands/help.py` and to the README's
Usage block. Use example paths relative to the repository
(`azure_ai/tests/data/maths.csv`), not a personal home directory, in new text.
Running a command that constructs a helper class creates resources: don't run
it to "try it out" without the user's OK.
