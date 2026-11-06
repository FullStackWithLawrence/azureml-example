---
name: azureml-azure-safety
description: Use before writing, testing or running any azureml-example code, test, command, notebook cell or CLI command that reaches Azure (azure-ai-ml MLClient: data assets, compute, AutoML jobs, models, batch endpoints and deployments; azure-storage-blob; az ml, az functionapp, func azure functionapp publish) or Kaggle. Explains which credentials are live on this machine, what each operation costs, what the live tests create and delete, the offline test mode, patching at the point of use, and what needs the user's OK first.
---

# Azure Safety

## Why this matters

On the user's machine, these reach a **real, paid Azure subscription**:

- `config.json` in the repository root (git-ignored), read by
  `MLClient.from_config()` in `AzureAIMLWorkspace`.
- The Azure CLI login (`az login`, in `~/.azure`), which
  `DefaultAzureCredential` and every `az` command use.
- `~/.kaggle/kaggle.json`, the user's Kaggle API key.

Nothing in the code stops a call from creating a resource. Treat every call
into `azure.ai.ml`, `azure.storage.blob`, `az` or `func` as real unless you have faked
it.

## What costs money

| Operation                                                                     | Cost and risk                                                                                                                         |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `AzureAIMLStudioComputeCluster(...)` for a name that doesn't exist            | Creates an AmlCompute cluster (default `STANDARD_DS3_V2`, up to 4 nodes). Scales to 0 after 300 s idle, but nodes bill while they run |
| `AzureAIMLStudioAssetsBatchEndpoint(...)` for a name that doesn't exist       | Creates an endpoint and a deployment (default `instance_count=2`), and may create a cluster                                           |
| `invoke_batch_job()`                                                          | Starts nodes and runs a job                                                                                                           |
| `run_automl_experiment()`                                                     | Hours of cluster time (`experiment_timeout_hours`, default 1.0) across many child runs. The most expensive call in the repository     |
| `AzureAIMLAssetsDataset(...)` for a new name                                  | Uploads to the workspace blob store and registers an MLTable data asset version. Cheap, but it is a write and it lingers              |
| `func azure functionapp publish`, `az functionapp create`, `az ml ... create` | Creates or replaces live, public services                                                                                             |
| `register_best_model()`, `delete()` methods, `az ml ... delete`               | Changes or removes the user's assets; may destroy someone's work                                                                      |

Read-only calls (`MLClient.from_config()`, `get_default_name()`,
`ml_client.data.get()`, `dataset_to_dataframe()`, `az ml compute show`,
`az account show`) are fine
when a task needs them, but they still need a network and a login. Never print
or log the subscription id, tenant id or keys they return
(`get_datastore_container()` reads the datastore's account key).

## Rules

1. **Ask before creating, changing or deleting anything in Azure**, for the
   specific resource. "Fix the batch endpoint code" is not permission to
   create an endpoint. When a step needs a real resource, give the user the
   command and let them run it.
2. **Run tests offline by default** (`GITHUB_ACTIONS=true`, see
   `azureml-python-testing`). Run the live suite only when the user asks, and
   then say what it will create.
3. **Fake before the first run.** When a new test must exercise code that calls
   Azure, add the patches before running it the first time.
4. **Patch at the point of use.** `ml_studio.py` imports names at module load
   (`from azure.ai.ml import MLClient`), so patch `azure_ai.ml_studio.MLClient`,
   `azure_ai.ml_studio.DefaultAzureCredential`, `azure_ai.ml_studio.ContainerClient`
   and so on; `self.mock_azure()` in `AzureMLTestBase` does the first two.
   Patching `azure.ai.ml.MLClient` has no effect on `ml_studio`'s
   already-imported name.
5. **Live tests clean up.** Name what they create with a unique suffix (see
   `test_batch_endpoint.py`'s hash suffix), and delete it in the test or in
   `tearDownClass`, even when the test fails. A `tearDownClass` returns early
   offline, since the base class's mocks are already stopped by then.
6. **Never run `make release`, `make docker-push`, `func azure functionapp
publish`, or `az ... create|update|delete`** unless the user asks for that
   command.
7. **Never commit credentials.** `config.json`, `.env` and `kaggle.json` are
   git-ignored. Don't copy their values into code, tests, docs, notebooks,
   skills or your summary. `config.example.json` and `.env.example` hold
   placeholders only.

## Names the live tests use

The live suite assumes things already exist in the user's workspace, and
creates others:

| Test file                                     | Uses or creates                                                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `test_cluster.py`                             | Gets, or creates, `tiny-cluster`                                                                                 |
| `test_dataset.py`                             | Reads `student-performance-base`; creates and archives `whenamancodes-student-performance`, `test-maths-dataset` |
| `test_batch_endpoint.py`                      | Needs model `best-01` and cluster `tiny-cluster`; creates and deletes `test-batch-endpoint-<hash>` endpoints     |
| `test_azureml_workspace.py`, `test_config.py` | Read-only workspace connection                                                                                   |

`AzureAIMLAssetsDataset.delete()` archives a data asset (v2 can't delete
one); `ml_client.data.restore()` brings it back, and archived assets still
show in the studio under "Include archived". Data assets made by the old v1
code are not MLTable and won't train an AutoML v2 job; re-create them under a
new name.

If a live test fails half way, check for leftovers and tell the user, with the
command to delete them, e.g. `az ml batch-endpoint list -o table` and
`az ml batch-endpoint delete --name <name> --yes` (the user runs the delete).

## Kaggle

`AzureAIMLAssetsDataset.from_kaggle()` downloads with the user's Kaggle key.
Downloads are free but rate limited, and some datasets need the user to accept
terms on kaggle.com first. In CI the key comes from the `KAGGLE_JSON` secret.
