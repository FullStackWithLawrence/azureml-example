# Azure AutoML Real-Time Deployment

Microsoft's recommended approach for deploying AutoML models to production for real-time endpoints is to download your AutoML model and deploy it using Azure Functions.

This repository serves the model from an Azure Function running Python 3.14. A pickled AutoML model (`model.pkl`) can't be loaded there: unpickling it needs Microsoft's AutoML runtime packages (`azureml-automl-runtime`, `azureml-training-tabular`), which don't support Python later than 3.11. Instead, AutoML exports the best model to [ONNX](https://onnx.ai/), an open model format, and the Function runs it with [onnxruntime](https://onnxruntime.ai/), which supports Python 3.14 and needs no AutoML code.

## Prerequisites

- Azure CLI installed and logged in
- Azure Functions Core Tools installed
- Python 3.14 installed locally, and this repository's virtual environment (`make init`)
- Access to the Azure ML workspace containing your training data

```console
# Install Azure Functions Core Tools (Mac)
brew tap azure/functions
brew install azure-functions-core-tools@4

# Verify installation
func --version
```

## Deployment

### Step 1: Train and Register an ONNX Model

`AzureAIMLStudioAuthoringAutomatedML.run_automl_experiment()` asks AutoML for ONNX-compatible models by default (`enable_onnx_compatible_models=True`). AutoML then tries only the algorithms it can export to ONNX, and saves each trial's model as `outputs/model.onnx` as well as in MLflow format. Register the best trial's ONNX model:

```python
from azure_ai.ml_studio import AzureAIMLStudioAuthoringAutomatedML

automl = AzureAIMLStudioAuthoringAutomatedML(dataset_name="titanic", cluster_name="tiny-cluster")
job = automl.run_automl_experiment(experiment_name="titanic-survival", target_column="Survived")
automl.register_best_model(job, model_name="titanic-survival-onnx", model_format="onnx")
```

AutoML jobs bill compute time for up to `experiment_timeout_hours` (one hour by default).

### Step 2: Download Your ONNX Model

```console
az ml model download \
  --name titanic-survival-onnx \
  --version 1 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name> \
  --download-path ./titanic-survival-app/automl-model/
```

This writes `titanic-survival-app/automl-model/titanic-survival-onnx/model.onnx`. The Function loads the first `model.onnx` it finds under `automl-model/`. The folder is in `.gitignore`.

To see the column names the model expects, which are the training data's columns without the target column:

```console
python -c "import glob, onnxruntime; m = glob.glob('titanic-survival-app/automl-model/**/model.onnx', recursive=True)[0]; print([i.name for i in onnxruntime.InferenceSession(m).get_inputs()])"
```

### Step 3: Create the Azure Function App

Python 3.14 runs on the [Flex Consumption](https://learn.microsoft.com/en-us/azure/azure-functions/flex-consumption-plan) plan, which bills per execution like the older Consumption plan. The Linux Consumption plan gets no new Python versions.

```console
# Create a new storage account (name must be globally unique, 3-24 chars, lowercase/numbers only)
az storage account create \
  --name <storage-account-name> \
  --resource-group <resource-group> \
  --location eastus2 \
  --sku Standard_LRS

# Create Function App
az functionapp create \
  --resource-group <resource-group> \
  --flexconsumption-location eastus2 \
  --runtime python \
  --runtime-version 3.14 \
  --functions-version 4 \
  --name <function-app-name> \
  --storage-account <storage-account-name>
```

### Step 4: Review the Function Code

The Function uses the Azure Functions [Python v2 programming model](https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python), in which a decorator in the Python code declares the HTTP trigger:

- [function_app.py](../titanic-survival-app/function_app.py): the `predict` function, at `/api/predict`
- [requirements.txt](../titanic-survival-app/requirements.txt): `azure-functions`, `numpy` and `onnxruntime`, which Azure installs when you publish
- [host.json](../titanic-survival-app/host.json): the Functions host configuration

### Step 5: Deploy Function

```console
cd titanic-survival-app
func azure functionapp publish <function-app-name>
```

### Step 6: Test Your Deployment

The function requires a function key. Get it from the Azure portal (your Function App -> Functions -> predict -> Function Keys), and keep it out of code and documentation.

Send one object per passenger, keyed by the model's column names (Step 2):

```console
curl -X POST "https://<function-app-name>.azurewebsites.net/api/predict" \
  -H "Content-Type: application/json" \
  -H "x-functions-key: <function-key>" \
  -d '{"data": [{"Pclass": 3, "Sex": 1, "Age": 22.0, "Fare": 7.25, "...": "..."}]}'
```

A list of values in the model's column order also works: `{"data": [[3, 1, 22.0, 7.25, ...]]}`.

Expected response, one prediction per passenger:

```json
{
  "prediction": [0]
}
```

To test locally before you publish, run `func start` in `titanic-survival-app/` and send the same request to `http://localhost:7071/api/predict`, without the key.

## 🔧 Troubleshooting

### Check Function Logs

```console
# Check function logs
az functionapp log tail --name <function-app-name> --resource-group <resource-group>
```

### Common Issues

1. **`{"error": "The model is not available"}` (status 500)**: there's no `model.onnx` under `titanic-survival-app/automl-model/`. Repeat Step 2, then publish again.
2. **`{"error": "Row 0 has no value for '...'"}` (status 400)**: the request is missing one of the model's columns. The error names it; Step 2 shows how to list them all.
3. **No `outputs/model.onnx` in the AutoML job**: the job ran with `enable_onnx_compatible_models=False`. Run it again with the default.
4. **Python 3.14 isn't offered**: create the Function App on the Flex Consumption plan (`--flexconsumption-location`), not the Linux Consumption plan.
5. **Dependency Issues**: pin all package versions in requirements.txt, and keep them in step with `requirements/test.txt`, which the Function's unit tests use.
