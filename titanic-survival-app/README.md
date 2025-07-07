# Azure function deployment resources

This directory contains the deployment resources for a machine learning model created using Azure AutoML (Automated Machine Learning).
See [Azure AutoML Real-Time Deployment](../docs/AZURE_DEPLOYMENT.md)

## Overview

These files enable deployment of an AutoML-trained model as a serverless Azure Function for real-time inference. The model was trained using Azure's AutoML service and is now packaged for production deployment.

## Files

- **`__init__.py`** - Main Azure Function handler that loads the AutoML model and processes prediction requests
- **`requirements.txt`** - Python dependencies required for the function runtime
- **`function.json`** - Azure Functions configuration defining the HTTP trigger and bindings
- **`automl-model/`** - Directory containing the downloaded AutoML model artifacts

## Usage

This function is designed to be deployed to Azure Functions and provides a REST API endpoint for real-time model predictions.

### API Endpoint

```bash
curl -X POST https://titanic-survival-app.azurewebsites.net/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[3, 1, 22, 1, 0, 7.25, 2]]}'
```

### Response

```json
{
  "prediction": [0]
}
```

## Deployment

For complete deployment instructions, see the [Azure Deployment Guide](../docs/AZURE_DEPLOYMENT.md).

## Model Information

- **Source**: Azure AutoML
- **Task**: Binary Classification (Titanic Survival Prediction)
- **Input Features**: Passenger class, sex, age, siblings/spouses, parents/children, fare, embarked port
- **Output**: Survival prediction (0 = did not survive, 1 = survived)
