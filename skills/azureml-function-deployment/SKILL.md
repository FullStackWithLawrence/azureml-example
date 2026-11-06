---
name: azureml-function-deployment
description: Use when working on titanic-survival-app (the Azure Function that serves an AutoML model, exported to ONNX, over HTTP on Python 3.14), its function_app.py, host.json, requirements.txt and automl-model folder, its tests, or docs/AZURE_DEPLOYMENT.md, or when asked about registering or downloading the ONNX model, creating a Flex Consumption Function App, publishing with func azure functionapp publish, testing the function locally, or calling its endpoint. Covers why the app serves ONNX rather than a pickle, what is safe to do locally and what must be handed to the user.
---

# Function Deployment (titanic-survival-app)

Microsoft's suggested route for real-time inference from an AutoML model is to
download the model and serve it from an Azure Function. `titanic-survival-app/`
is that example; `docs/AZURE_DEPLOYMENT.md` is the guide.

## Why ONNX

The Function runs on Python 3.14. A pickled AutoML model (`model.pkl`) can't be
loaded there: unpickling it needs `azureml-automl-runtime` and
`azureml-training-tabular`, which require Python below 3.12, and every AutoML
model pickles their classes (v2 AutoML jobs too: the training runs on
Microsoft's own image). So the Function serves the ONNX copy of the model,
which AutoML saves as `outputs/model.onnx` when the job runs with
`enable_onnx_compatible_models` (the default in
`AzureAIMLStudioAuthoringAutomatedML.run_automl_experiment()`), and runs it
with `onnxruntime`, which has 3.14 wheels.

The model in the user's local `automl-model/titanic-survival/mlflow-model/`
(v1 AutoML, Python 3.9, scikit-learn 1.5.1) can't be served on 3.14. Serving
the app needs a new AutoML run with ONNX enabled, registered with
`register_best_model(..., model_format="onnx")`. That run costs money: the
user starts it (`azureml-azure-safety`).

## The pieces

| File                                   | Role                                                                                                                         |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `function_app.py`                      | Python v2 programming model: `app = func.FunctionApp(http_auth_level=FUNCTION)`; `predict` is a POST route at `/api/predict` |
| `host.json`                            | Functions host configuration (version 2.0, extension bundle 4.x)                                                             |
| `.funcignore`                          | What `func azure functionapp publish` leaves out (venvs, `__pycache__`, `local.settings.json`, README)                       |
| `requirements.txt`                     | The Function runtime's packages, pinned: `azure-functions`, `numpy`, `onnxruntime`. `#` comments only                        |
| `automl-model/<model-name>/model.onnx` | The downloaded ONNX model. `load_model()` uses the first `automl-model/**/model.onnx`                                        |

How `function_app.py` works:

- `load_model()` is `@cache`d: the model loads on the first request, not at
  import, so a missing model doesn't stop the app from starting. A missing
  model returns status 500, `{"error": "The model is not available"}`, without
  file paths.
- AutoML's ONNX model has one input per column, shaped `[None, 1]`.
  `to_model_inputs()` builds one numpy column per input, typed from
  `ONNX_DTYPES` (string inputs are converted with `str()`), and
  `column_values()` reads each column from the rows.
- A request is `{"data": [...]}`, each row either a dict keyed by column name
  or a list of values in the model's input order. The first output is the
  label: `{"prediction": [...]}`.
- Any bad request (invalid JSON, no `data`, empty rows, a missing column, a
  wrong row length, a type onnxruntime rejects) returns status 400 with
  `{"error": "..."}`.

`automl-model` is in `.gitignore`. A downloaded model is the user's artifact;
never commit a new one or overwrite the local copy unasked.

## Tests

`azure_ai/tests/test_titanic_survival_app.py` runs offline. It builds a tiny
ONNX model with the `onnx` package (`Age` float, `Sex` int64, `Name` string;
label is `Age + Sex > 30`) in a temporary `automl-model/` folder, imports
`function_app.py` with `importlib` (the folder name has a hyphen), patches
`MODEL_GLOB` and calls `load_model.cache_clear()` in `setUp`. The decorated
`predict` is callable directly with an `azure.functions.HttpRequest`.

The test needs `onnx`, `onnxruntime` and `azure-functions`, which come from
`requirements/test.txt` (`test.in` includes
`../../titanic-survival-app/requirements.txt`). Keep the Function's pins and
`test.txt` in step (`make python-requirements`).

## What needs the user

These create, replace or bill real, public resources. Never run them yourself;
give the user the exact command:

- The AutoML run and `register_best_model(..., model_format="onnx")`
- `az ml model download --name <model-name> --version <n> --download-path ./titanic-survival-app/automl-model/ ...`
  (writes over the local model folder)
- `az storage account create`, and
  `az functionapp create --flexconsumption-location <region> --runtime python --runtime-version 3.14 --functions-version 4 ...`.
  Python 3.14 runs on the Flex Consumption plan; the Linux Consumption plan
  gets no new Python versions.
- `func azure functionapp publish <function-app-name>` (replaces the live code)
- `az functionapp keys list` and anything that prints a function key

Safe for you to do:

- Read and edit the code, `host.json`, `requirements.txt` and docs.
- Run the offline tests.
- Run `func start` in `titanic-survival-app/` and POST to
  `localhost:7071/api/predict` with `curl` when a `model.onnx` is present
  locally and the user wants a local test. This needs `azure-functions-core-tools` (installed
  at `/opt/homebrew/bin/func`).
- List a local model's input names with
  `onnxruntime.InferenceSession(path).get_inputs()`.

## Rules for the code

- The model's feature columns and order come from the trained model; keep the
  README's example payload and `docs/AZURE_DEPLOYMENT.md` in step with them.
- Never load a pickle in the Function, and never load a model from a request.
- Errors return a JSON body; status 400 for the caller's mistakes, 500 when the
  model is missing. Never return a traceback or a file path.
- Pin `requirements.txt` to versions with Python 3.14 wheels, in step with
  `requirements/test.txt`, and keep it free of `"""` docstrings: pip reads
  every non-`#` line as a requirement.
- Function keys and the app's url with `?code=` are secrets; keep them out of
  code, docs and your replies. Docs show the `x-functions-key: <function-key>`
  header.
- The docs give placeholders (`<resource-group>`, `<function-app-name>`) for
  resource names.
