---
name: azureml-ml-studio
description: Use when adding or changing a helper class in azure_ai/ml_studio.py (AzureAIMLWorkspace, AzureAIMLAssetsDataset, AzureAIMLStudioComputeCluster, AzureAIMLStudioAssetsBatchEndpoint, AzureAIMLStudioAuthoringAutomatedML) or a command module in azure_ai/commands/. Covers the class hierarchy and naming, the get_default_name / get_or_create / delete pattern, the split between the v1 azureml-core SDK and the v2 azure-ai-ml SDK, logging and error handling, and the checklist for a new class or command.
---

# The ml_studio Helpers and Commands

## The classes

All live in `azure_ai/ml_studio.py`. Names follow the Azure ML Studio menu:
`AzureAIMLStudio<Menu><Item>` or `AzureAIML<Menu><Item>`.

| Class                                 | Studio area        | SDK | Main methods                                                                                 |
| ------------------------------------- | ------------------ | --- | -------------------------------------------------------------------------------------------- |
| `AzureAIMLWorkspace`                  | Workspace          | v1  | `connect_workspace()` via `Workspace.from_config()`; base of all the others                  |
| `AzureAIMLAssetsDataset`              | Assets > Data      | v1  | `get_or_create()`, `from_file()`, `from_kaggle()`, `dataset_to_dataframe()`, `delete()`      |
| `AzureAIMLStudioComputeCluster`       | Compute > Clusters | v1  | `get_default_name()`, `get_or_create()`                                                      |
| `AzureAIMLStudioAssetsBatchEndpoint`  | Assets > Endpoints | v2  | `get_or_create()`, `_create_batch_deployment()`, `invoke_batch_job()`, `delete()`            |
| `AzureAIMLStudioAuthoringAutomatedML` | Authoring > AutoML | v1  | `run_automl_experiment()`, `get_best_model()`, `register_best_model()`, `show_run_details()` |

**Two SDKs.** AutoML, datasets and compute use the v1 `azureml-core` /
`azureml-train-automl` SDK (`Workspace`, `Dataset`, `ComputeTarget`,
`Experiment`, `AutoMLConfig`). Batch endpoints and models use the v2
`azure-ai-ml` SDK (`MLClient`, `BatchEndpoint`, `ModelBatchDeployment`),
authenticated with `DefaultAzureCredential` and built from the v1
workspace's `subscription_id`, `resource_group` and `name`. Keep each class on
the SDK it uses now; don't mix in the other inside one method without a reason
written in a comment. v1 is deprecated by Microsoft but is what the course
teaches.

## The pattern

```python
class AzureAIMLStudio<Thing>(AzureAIMLWorkspace):
    """
    Azure AI ML Studio <Menu> -> <Item> class.
    Class to encapsulate Azure ML <thing> functionality.
    """

    <thing>: Optional[<AzureType>] = None

    def __init__(self, <thing>_name: Optional[str] = None, *args, <options with defaults>, **kwargs):
        super().__init__(*args, **kwargs)
        self.<thing>_name = <thing>_name or AzureAIMLStudio<Thing>.get_default_name(self.workspace)
        ...
        self.<thing> = self.get_or_create()

    @classmethod
    def get_default_name(cls, workspace: Workspace) -> str:
        """Name of the first <thing> in the workspace. Raises ValueError if none."""

    def get_or_create(self) -> <AzureType>:
        """Return the cached object, else the existing one by name, else create it."""

    def delete(self):
        """Delete it; log a warning, don't raise, when it doesn't exist."""
```

- Constructing an instance **creates the resource if it doesn't exist**. That
  is the teaching point, and the reason tests and commands cost money. Keep
  creation in `get_or_create()`, never in a getter or a property.
- Options are keyword arguments with sensible, cheap defaults (small VM sizes,
  `min_nodes=0`, short timeouts).
- Logging: `logger.info("Found existing cluster: %s", self.cluster_name)`,
  lazy %-style arguments, no f-strings in log calls, no secrets.
- Error handling: catch the specific "not found" exception
  (`UserErrorException`, `ComputeTargetException`, `ResourceNotFoundError`)
  to fall through to creation; for anything else log
  `logger.error("Failed to ... %s: %s", name, e)` and re-raise. A deliberate
  broad `except Exception` carries `# pylint: disable=W0718`.
- Long-running Azure calls wait with `wait_for_completion(show_output=True)`
  or a poller's `.result()`, so a student sees progress.
- Google-style docstrings (`Args:`, `Returns:`, `Raises:`) that match the
  actual signature.

## Adding a class

1. Add it to `ml_studio.py` following the pattern, after the class it depends on.
2. Add `azure_ai/tests/test_<thing>.py`: offline tests with mocks patched at
   `azure_ai.ml_studio.<Name>`, plus guarded live tests that clean up
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
