# Azure Account Setup for AutoML

## Overview

This guide walks you through setting up a complete Azure environment from scratch to run AutomatedML jobs successfully. Follow these steps in order to avoid common quota and configuration issues.

**Note: This guide uses a combination of both web console steps, and also command-line interface instructions.**

## 🚀 Step 1: Create Azure Account

### New Users

1. Go to [Azure Portal](https://portal.azure.com)
2. Click **"Start free"** or **"Create account"**
3. Sign up with Microsoft account or create new one
4. Complete identity verification (phone/credit card)
5. **Free tier includes**: $200 credit for 30 days

### Existing Users

1. Sign in to [Azure Portal](https://portal.azure.com)
2. Ensure you have an active subscription

## 💳 Step 2: Create or Verify Subscription

### Check Current Subscriptions Using CLI

```console
az login
az account list --output table
```

### Create New Subscription in Azure Portal

1. In Azure Portal, go to **Subscriptions**
2. Click **+ Add**
3. Choose subscription type:
   - **Pay-As-You-Go** (recommended for AutoML)
   - **Free Trial** (limited but good for testing)

### Set Active Subscription Using CLI

```console
az account set --subscription "your-subscription-name"
```

## 🏗️ Step 3: Create Resource Group

### Create Resource Group in Azure Portal

1. Go to **Resource groups**
2. Click **+ Create**
3. Choose:
   - **Subscription**: Your subscription
   - **Resource group name**: `automl-resources` (or your preference)
   - **Region**: `East US 2` (recommended for ML workloads)

### Create Resource Group Using CLI

```console
az group create --name automl-resources --location eastus2
```

## 🧠 Step 4: Create Azure ML Workspace

### Create Workspace in Azure Portal

1. Search for **"Machine Learning"**
2. Click **+ Create**
3. Configure:
   - **Subscription**: Your subscription
   - **Resource group**: `automl-resources`
   - **Workspace name**: `automl-workspace`
   - **Region**: `East US 2`
   - **Storage account**: Create new (default)
   - **Key vault**: Create new (default)
   - **Application insights**: Create new (default)
   - **Container registry**: None (will create when needed)

### Create Workspace Using CLI

```console
az ml workspace create \
  --name automl-workspace \
  --resource-group automl-resources \
  --location eastus2
```

## 🖥️ Step 5: Install Azure CLI and ML Extension

### Install on macOS

```console
# Install Azure CLI
brew install azure-cli

# Install ML extension
az extension add --name ml

# Allow preview features (required for some ML commands)
az config set extension.dynamic_install_allow_preview=true
```

### Install on Linux

```console
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az extension add --name ml
```

### Install on Windows

```powershell
# Download and run Azure CLI installer from:
# https://aka.ms/installazurecliwindows
az extension add --name ml
```

### Verify Installation

```console
az --version
az ml --help
```

## ⚙️ Step 6: Register Required Resource Providers

Azure ML requires several resource providers to be registered:

```console
# Core providers for Azure ML
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.ContainerInstance

# Check registration status
az provider list --query "[?namespace=='Microsoft.MachineLearningServices' || namespace=='Microsoft.Compute' || namespace=='Microsoft.Storage'].{Namespace:namespace, State:registrationState}" --output table
```

**Note**: Registration can take 5-10 minutes. Wait for all to show "Registered" before proceeding.

## 💻 Step 7: Create Compute Cluster

### Create Compute Cluster Using CLI

```console
az ml compute create \
  --name tiny-cluster \
  --type amlcompute \
  --size Standard_D4s_v3 \
  --max-instances 1 \
  --min-instances 0 \
  --idle-time-before-scale-down 300 \
  --resource-group automl-resources \
  --workspace-name automl-workspace
```

### Verify Cluster Creation

```console
az ml compute show \
  --name tiny-cluster \
  --resource-group automl-resources \
  --workspace-name automl-workspace \
  --query "{size:size, location:location, maxInstances:scaleSettings.maxNodeCount}"
```

**Expected Output:**

```json
{
  "location": "eastus2",
  "maxInstances": 4,
  "size": "Standard_D4s_v3"
}
```

## 📊 Step 8: Configure Compute Quotas

### Check Current Quotas

```console
# Check your region's compute quotas
az vm list-usage --location eastus2 --query "[?contains(name.value, 'Standard')]" --output table
```

### Common Quota Requirements for AutoML

| VM Family                  | Recommended Quota | Use Case               |
| -------------------------- | ----------------- | ---------------------- |
| Standard DSv3 Family vCPUs | 2-10              | General AutoML jobs    |
| Standard Dv3 Family vCPUs  | 16-32             | CPU-intensive training |
| Standard FSv2 Family vCPUs | 16-32             | Fast compute jobs      |
| Total Regional vCPUs       | 50-100            | Overall regional limit |

### Request Quota Increase in Azure Portal

1. Go to **Subscriptions** → Your subscription
2. Click **Usage + quotas**
3. Filter:
   - **Provider**: Compute
   - **Location**: East US 2
4. Find **"Standard DSv3 Family vCPUs"**
5. Click pencil icon → **New limit**: 32
6. Submit request

### Request Quota Increase Using Support Ticket

```console
# This creates a support ticket for quota increase
az support tickets create \
  --ticket-name "AutoML Compute Quota Increase" \
  --description "Request to increase Standard DSv3 Family vCPUs quota from current limit to 32 for AutoML workloads" \
  --contact-first-name "Your Name" \
  --contact-last-name "Last Name" \
  --contact-method "email" \
  --contact-email "your-email@domain.com" \
  --severity "minimal"
```

**Processing Time**: Quota increases typically take 1-2 business days.

## 🔧 Step 9: Update Compute Cluster After Quota Increase

### Update Cluster Maximum Instances

Once your quota is approved, update your cluster:

```console
# Update cluster to use new quota limits
az ml compute update \
  --name tiny-cluster \
  --resource-group automl-resources \
  --workspace-name automl-workspace \
  --max-instances 2 \
  --min-instances 0
```

### Verify Cluster Update

```console
az ml compute show \
  --name tiny-cluster \
  --resource-group automl-resources \
  --workspace-name automl-workspace \
  --query "scaleSettings"
```

## 🧪 Step 10: Test AutoML Setup

### Download Workspace Configuration File

1. Go to [Azure ML Studio](https://ml.azure.com)
2. Select your workspace
3. Click download icon (⬇️) next to workspace name
4. Save as `config.json` in your project root

### Test Workspace Connection

```python
from azureml.core import Workspace, Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment

# Test workspace connection
ws = Workspace.from_config()
print(f"✅ Connected to workspace: {ws.name}")

# Test compute target
compute_target = ws.compute_targets["tiny-cluster"]
print(f"✅ Compute cluster ready: {compute_target.provisioning_state}")
```

## 🚨 Troubleshooting Common Issues

### Issue 1: Quota Exceeded Error

```console
Requested 2 nodes but AzureMLCompute cluster only has 1 maximum nodes.
```

**Solution:**

1. Check your VM family quota (Step 8)
2. Request quota increase
3. Update cluster max instances

### Issue 2: Resource Provider Not Registered

```console
The selected provider is not registered for some of the selected subscriptions.
```

**Solution:**

```console
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.Compute
```

### Issue 3: Workspace Not Found

```console
ParentResourceNotFound: Failed to perform 'read' on resource
```

**Solution:**

- Verify workspace name (check for spaces/special characters)
- Ensure you're in the correct subscription
- Check resource group name

### Issue 4: Authentication Errors

```console
AuthenticationException: Authentication failed
```

**Solution:**

```console
az login
az account set --subscription "your-subscription-name"
```

## 💰 Cost Management

### Estimated Costs for AutoML

| Resource                | Estimated Cost      | Notes                |
| ----------------------- | ------------------- | -------------------- |
| Standard_D4s_v3 compute | $0.19/hour per node | Only when running    |
| Storage account         | $0.05/month         | Minimal for datasets |
| Key Vault               | $0.03/month         | For secrets          |

### Configure Cost Optimization

1. **Set idle scale-down**: Clusters auto-scale to 0 when idle
2. **Use appropriate VM sizes**: Don't over-provision
3. **Monitor usage**: Set up billing alerts
4. **Delete resources**: Remove test resources when done

```console
# Set up auto-scale down after 5 minutes
az ml compute update \
  --name tiny-cluster \
  --idle-time-before-scale-down 300 \
  --resource-group automl-resources \
  --workspace-name automl-workspace
```

## ✅ Validation Checklist

Before running AutoML jobs, verify:

- [ ] Azure account created and verified
- [ ] Active subscription with sufficient credit/billing
- [ ] Resource group created in appropriate region
- [ ] Azure ML workspace created and accessible
- [ ] Azure CLI installed with ML extension
- [ ] All resource providers registered and active
- [ ] Compute cluster created with appropriate VM size
- [ ] Compute quotas increased (16+ vCPUs recommended)
- [ ] Cluster max instances updated after quota approval
- [ ] Workspace config.json downloaded and placed
- [ ] Test connection successful

## 🔗 Next Steps

After completing this setup:

1. **Follow the [Azure ML Configuration Guide](./AZURE_ML_CONFIG.md)** for workspace connection
2. **Run your first AutoML experiment**
3. **Set up data storage and datasets**
4. **Configure model deployment endpoints**

## 📚 Additional Resources

- [Azure Free Account](https://azure.microsoft.com/free/)
- [Azure ML Pricing](https://azure.microsoft.com/pricing/details/machine-learning/)
- [AutoML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/ml)
- [Quota Management](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/azure-subscription-service-limits)
