# Azure ML CLI Commands Reference

## Overview

This document provides common Azure ML CLI commands with example responses to help you manage your Azure ML workspace resources from the command line.

## 🔧 Prerequisites

Make sure you have the Azure ML CLI extension installed:

```console
az extension add --name ml
```

Login to Azure:

```console
az login
```

## 📊 Compute Management

### Show Compute Cluster Information

Query specific properties of a compute cluster:

```console
$ az ml compute show --name tiny-cluster --resource-group ubc-cdl10 --workspace-name UBC-CDL10 --query "{size:size, location:location, maxInstances:scaleSettings.maxNodeCount}"
```

**Response:**

```json
{
  "location": "eastus2",
  "maxInstances": null,
  "size": "Standard_D4s_v3"
}
```

### Update Compute Cluster Scaling

Update the minimum and maximum instance counts for auto-scaling:

```console
$ az ml compute update \
  --name tiny-cluster \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --max-instances 2 \
  --min-instances 0
```

**Response:**

```json
{
  "enable_node_public_ip": true,
  "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/ubc-cdl10/computes/tiny-cluster",
  "idle_time_before_scale_down": 120,
  "location": "eastus2",
  "max_instances": 2,
  "min_instances": 0,
  "name": "tiny-cluster",
  "provisioning_state": "Succeeded",
  "resourceGroup": "ubc-cdl10",
  "size": "Standard_D4s_v3",
  "ssh_public_access_enabled": false,
  "tier": "dedicated",
  "type": "amlcompute"
}
```

### List All Compute Resources

Show all compute resources in your workspace:

```console
$ az ml compute list --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
[
  {
    "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/ubc-cdl10/computes/tiny-cluster",
    "location": "eastus2",
    "max_instances": 2,
    "min_instances": 0,
    "name": "tiny-cluster",
    "provisioning_state": "Succeeded",
    "size": "Standard_D4s_v3",
    "type": "amlcompute"
  }
]
```

### Create New Compute Cluster

Create a new compute cluster:

```console
$ az ml compute create \
  --name new-cluster \
  --type amlcompute \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --size Standard_D2s_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --idle-time-before-scale-down 300
```

**Response:**

```json
{
  "enable_node_public_ip": true,
  "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/ubc-cdl10/computes/new-cluster",
  "idle_time_before_scale_down": 300,
  "location": "eastus2",
  "max_instances": 4,
  "min_instances": 0,
  "name": "new-cluster",
  "provisioning_state": "Creating",
  "resourceGroup": "ubc-cdl10",
  "size": "Standard_D2s_v3",
  "type": "amlcompute"
}
```

## 🏗️ Workspace Management

### Show Workspace Information

Display workspace details:

```console
$ az ml workspace show --name UBC-CDL10 --resource-group ubc-cdl10
```

**Response:**

```json
{
  "discovery_url": "https://eastus2.api.azureml.ms/discovery",
  "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/UBC-CDL10",
  "location": "eastus2",
  "name": "UBC-CDL10",
  "resourceGroup": "ubc-cdl10",
  "sku": "Basic",
  "storage_account": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.Storage/storageAccounts/ubccdl10storage",
  "tags": {},
  "type": "Microsoft.MachineLearningServices/workspaces"
}
```

### List Workspaces

List all workspaces in a resource group:

```console
$ az ml workspace list --resource-group ubc-cdl10
```

**Response:**

```json
[
  {
    "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/UBC-CDL10",
    "location": "eastus2",
    "name": "UBC-CDL10",
    "resourceGroup": "ubc-cdl10",
    "sku": "Basic",
    "type": "Microsoft.MachineLearningServices/workspaces"
  }
]
```

## 📊 Data Management

### List Datastores

Show all datastores in your workspace:

```console
$ az ml datastore list --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
[
  {
    "name": "workspaceblobstore",
    "type": "azure_blob",
    "account_name": "ubccdl10storage",
    "container_name": "azureml-blobstore-12345678-1234-1234-1234-123456789012",
    "is_default": true
  },
  {
    "name": "workspacefilestore",
    "type": "azure_file",
    "account_name": "ubccdl10storage",
    "file_share_name": "azureml-filestore-12345678-1234-1234-1234-123456789012",
    "is_default": false
  }
]
```

### Show Datastore Details

Get detailed information about a specific datastore:

```console
$ az ml datastore show --name workspaceblobstore --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
{
  "account_name": "ubccdl10storage",
  "container_name": "azureml-blobstore-12345678-1234-1234-1234-123456789012",
  "credential_type": "account_key",
  "is_default": true,
  "name": "workspaceblobstore",
  "protocol": "https",
  "type": "azure_blob"
}
```

## 🧪 Experiment and Job Management

### List Experiments

Show all experiments in your workspace:

```console
$ az ml experiment list --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
[
  {
    "name": "Default",
    "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/UBC-CDL10/experiments/Default"
  },
  {
    "name": "training-experiment",
    "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/UBC-CDL10/experiments/training-experiment"
  }
]
```

### List Jobs

Show recent jobs:

```console
$ az ml job list --resource-group ubc-cdl10 --workspace-name UBC-CDL10 --max-results 5
```

**Response:**

```json
[
  {
    "name": "brave_needle_12345",
    "display_name": "Training Job",
    "status": "Completed",
    "creation_context": {
      "created_at": "2025-07-07T10:30:00Z",
      "created_by": "user@example.com"
    },
    "experiment_name": "training-experiment",
    "type": "command"
  }
]
```

## 🚀 Model Management

### List Registered Models

Show all registered models:

```console
$ az ml model list --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
[
  {
    "name": "my-classification-model",
    "version": "1",
    "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/UBC-CDL10/models/my-classification-model/versions/1",
    "creation_context": {
      "created_at": "2025-07-07T09:15:00Z"
    },
    "tags": {
      "accuracy": "0.95",
      "framework": "scikit-learn"
    }
  }
]
```

### Show Model Details

Get detailed information about a specific model:

```console
$ az ml model show --name my-classification-model --version 1 --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
{
  "name": "my-classification-model",
  "version": "1",
  "id": "/subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourceGroups/ubc-cdl10/providers/Microsoft.MachineLearningServices/workspaces/UBC-CDL10/models/my-classification-model/versions/1",
  "path": "azureml://subscriptions/1b74cc43-233e-43a5-9564-d1d3f3e4f39c/resourcegroups/ubc-cdl10/workspaces/UBC-CDL10/datastores/workspaceblobstore/paths/models/my-classification-model/1/",
  "description": "Classification model trained on customer data",
  "tags": {
    "accuracy": "0.95",
    "framework": "scikit-learn"
  },
  "creation_context": {
    "created_at": "2025-07-07T09:15:00Z",
    "created_by": "user@example.com"
  }
}
```

## 🔧 Environment Management

### List Environments

Show all custom environments:

```console
$ az ml environment list --resource-group ubc-cdl10 --workspace-name UBC-CDL10
```

**Response:**

```json
[
  {
    "name": "sklearn-env",
    "version": "1",
    "description": "Environment for scikit-learn training",
    "creation_context": {
      "created_at": "2025-07-06T14:20:00Z"
    }
  }
]
```

## 💡 Useful Query Examples

### Get Compute Costs Summary

```console
$ az ml compute list --resource-group ubc-cdl10 --workspace-name UBC-CDL10 --query "[].{name:name, size:size, maxInstances:max_instances, state:provisioning_state}"
```

**Response:**

```json
[
  {
    "maxInstances": 2,
    "name": "tiny-cluster",
    "size": "Standard_D4s_v3",
    "state": "Succeeded"
  }
]
```

### Get Failed Jobs

```console
$ az ml job list --resource-group ubc-cdl10 --workspace-name UBC-CDL10 --query "[?status=='Failed'].{name:name, experiment:experiment_name, status:status}"
```

**Response:**

```json
[
  {
    "experiment": "training-experiment",
    "name": "sad_mountain_67890",
    "status": "Failed"
  }
]
```

## 🔗 Helpful Aliases

Add these to your shell profile for convenience:

```console
# Add to ~/.bashrc or ~/.zshrc
alias azml="az ml"
alias azmlws="az ml workspace"
alias azmlcompute="az ml compute"
alias azmlmodel="az ml model"

# With your specific workspace details
alias myws="--resource-group ubc-cdl10 --workspace-name UBC-CDL10"
```

Usage examples:

```console
azmlcompute list myws
azmlmodel list myws
```

## 📚 Additional Resources

- [Azure ML CLI Reference](https://docs.microsoft.com/en-us/cli/azure/ml)
- [Azure CLI Installation Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Azure ML CLI Extension](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)
