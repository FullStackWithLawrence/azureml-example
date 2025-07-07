# Azure AutoML Real-Time Deployment

Microsoft's recommended approach for deploying AutoML models to production for real-time endpoints is to download your AutoML model and deploy it using Azure Functions.

## Prerequisites

- Azure CLI installed and logged in
- Azure Functions Core Tools installed
- Python 3.8+ installed locally
- Access to the Azure ML workspace containing your model

```console
# Install Azure Functions Core Tools (Mac)
brew tap azure/functions
brew install azure-functions-core-tools@4

# Verify installation
func --version
```

## Deployment

### Step 1: Download Your AutoML Model

```console
# Download your trained AutoML model
az ml model download \
  --name titanic-survival \
  --version 1 \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --download-path ./titanic-survival-app/automl-model/
```

### Step 2: Create Azure Function App

```console
# Create a new storage account (name must be globally unique, 3-24 chars, lowercase/numbers only)
az storage account create \
  --name ubccdl10titanic \
  --resource-group ubc-cdl10 \
  --location eastus2 \
  --sku Standard_LRS

# Create Function App
az functionapp create \
  --resource-group ubc-cdl10 \
  --consumption-plan-location eastus2 \
  --runtime python \
  --runtime-version 3.8 \
  --functions-version 4 \
  --name titanic-survival-app \
  --storage-account ubccdl10titanic
```

### Step 3: Create Function Code

- [**init**.py](../titanic-survival-app/__init__.py)
- [requirements.txt](../titanic-survival-app/requirements.txt)
- [function.json](../titanic-survival-app/function.json)

### Step 4: Deploy Function

```console
cd titanic-survival-app
func azure functionapp publish titanic-survival-app
```

### Step 5: Test Your Deployment

```console
# Test the function
curl -X POST https://titanic-survival-app.azurewebsites.net/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[3, 1, 22, 1, 0, 7.25, 2]]}'
```

Expected response:

```json
{
  "prediction": [0]
}
```

## 🔧 Troubleshooting

### Check Function Logs

```console
# Check function logs
az functionapp log tail --name titanic-survival-app --resource-group ubc-cdl10
```

### Common Issues

1. **Model Loading Errors**: Ensure model.pkl is in the same directory as **init**.py
2. **Dependency Issues**: Pin all package versions in requirements.txt
3. **Memory Issues**: Use Premium plan for large models
4. **JSON Issues**: Ensure input data format matches training data
