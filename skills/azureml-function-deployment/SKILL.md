---
name: azureml-function-deployment
description: Use when working on titanic-survival-app (the Azure Function that serves a downloaded AutoML model over HTTP), its function.json, requirements.txt and automl-model folder, or docs/AZURE_DEPLOYMENT.md, or when asked about downloading a model with az ml model download, creating a Function App, publishing with func azure functionapp publish, testing the function locally, or calling its endpoint. Covers what is safe to do locally and what must be handed to the user.
---

# Function Deployment (titanic-survival-app)

Microsoft's suggested route for real-time inference from an AutoML model is to
download the model and serve it from an Azure Function. `titanic-survival-app/`
is that example; `docs/AZURE_DEPLOYMENT.md` is the guide.

## The pieces

| File                                          | Role                                                                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`                                 | `main(req)`: loads `automl-model/model.pkl` once at cold start, predicts on `{"data": [[...]]}`, returns `{"prediction": [...]}` |
| `function.json`                               | HTTP trigger, `authLevel: function`, POST only                                                                                   |
| `requirements.txt`                            | The Function runtime's packages                                                                                                  |
| `automl-model/titanic-survival/mlflow-model/` | The downloaded model: `MLmodel`, `model.pkl`, `conda.yaml`, `python_env.yaml`                                                    |

`automl-model` is in `.gitignore`. A downloaded model is the user's artifact;
never commit a new one or overwrite the local copy unasked.

## What needs the user

These create, replace or bill real, public resources. Never run them yourself;
give the user the exact command:

- `az ml model download ...` (writes over the local model folder)
- `az storage account create`, `az functionapp create`
- `func azure functionapp publish titanic-survival-app` (replaces the live code)
- `az functionapp keys list` and anything that prints a function key

Safe for you to do:

- Read and edit the code, `function.json` and docs.
- Run `func start` in `titanic-survival-app/` and `curl` `localhost:7071`
  when the model is present locally and the user wants a local test. This
  needs `azure-functions-core-tools` (installed at `/opt/homebrew/bin/func`).
- Unit-test `main()` with a mocked `azure.functions.HttpRequest` and a stub
  model, without loading the real pickle (patch `pickle.load` or `open` before
  importing). The folder name has a hyphen, so import it with `importlib`.

## Rules for the code

- The model's feature columns and order come from the trained model; keep the
  README's example payload and `docs/AZURE_DEPLOYMENT.md` in step with them.
- `pickle.load` keeps its `# nosec B301` comment: the model is our own
  artifact. Never load a pickle from a request.
- Errors return a JSON body and status 400, as `main()` does now; never return
  a traceback.
- Keep `requirements.txt` compatible with the model's `conda.yaml` /
  `python_env.yaml` (the scikit-learn version that trained it). Version skew
  shows up only at prediction time.
- Note that `requirements.txt` starts with a `"""` docstring, which pip reads
  as a requirement line. If the user reports a failed deploy, check that first.
- Function keys and the app's url with `?code=` are secrets; keep them out of
  code, docs and your replies.
- The docs give real resource names (`ubc-cdl10`, `titanic-survival-app`) as
  the course's examples. In new text, show placeholders (`<resource-group>`).
