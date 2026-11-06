# Azure function deployment resources

This directory contains the deployment resources for a machine learning model created using Azure AutoML (Automated Machine Learning).
See [Azure AutoML Real-Time Deployment](../docs/AZURE_DEPLOYMENT.md)

## Overview

These files deploy an AutoML-trained model as a serverless Azure Function for real-time inference, on Python 3.14. AutoML exports the model to ONNX, and the function runs it with onnxruntime.

## Files

- **`function_app.py`** - The Azure Function: the `predict` HTTP trigger, which loads the ONNX model and returns predictions (Python v2 programming model)
- **`requirements.txt`** - Python dependencies required for the function runtime
- **`host.json`** - Azure Functions host configuration
- **`.funcignore`** - Files that `func azure functionapp publish` leaves out
- **`automl-model/`** - Directory containing the downloaded ONNX model (`automl-model/<model-name>/model.onnx`, not committed)

The unit tests are in [azure_ai/tests/test_titanic_survival_app.py](../azure_ai/tests/test_titanic_survival_app.py). They build a tiny ONNX model, so they run without Azure or a downloaded model.

## Usage

This function is designed to be deployed to Azure Functions and provides a REST API endpoint for real-time model predictions.

### API Endpoint

```bash
curl -X POST "https://<function-app-name>.azurewebsites.net/api/predict" \
  -H "Content-Type: application/json" \
  -H "x-functions-key: <function-key>" \
  -d '{"data": [{"Pclass": 3, "Sex": 1, "Age": 22.0, "Fare": 7.25, "...": "..."}]}'
```

Each passenger is an object keyed by the model's column names. A list of values in the model's column order also works.

### Response

```json
{
  "prediction": [0]
}
```

Errors return a JSON body, `{"error": "..."}`: status 400 for a bad request, 500 when the model isn't there.

## Deployment

For complete deployment instructions, see the [Azure Deployment Guide](../docs/AZURE_DEPLOYMENT.md).

## Model Information

- **Source**: Azure AutoML, with ONNX-compatible models enabled
- **Format**: ONNX, run with onnxruntime
- **Task**: Binary Classification (Titanic Survival Prediction)
- **Input Features**: The training data's columns, such as passenger class, sex, age, siblings/spouses, parents/children, fare and embarked port
- **Output**: Survival prediction (0 = did not survive, 1 = survived)
