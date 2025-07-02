# Azure ML Configuration Guide

## Overview

This guide explains how to securely configure your Azure ML workspace connection using `config.json` and test that it works properly.

## 📁 Where to Place config.json

### Option 1: Project Root (Recommended for Development)

```
/Users/mcdaniel/Desktop/gh/fswl/azureml-example/config.json
```

**Pros:**

- Automatically detected by `Workspace.from_config()`
- Project-specific configuration
- Already in `.gitignore` (secure)

**Cons:**

- Need separate config for each project

### Option 2: User Home Directory (Recommended for Global Use)

```
~/.azureml_example/config.json
```

**Pros:**

- Shared across all Azure ML projects
- Standard Azure ML location
- Outside project directory

**Cons:**

- Global configuration (may not work if you have multiple workspaces)

## 🔒 Security Considerations

### ✅ Already Secured in This Project:

- `config.json` is in `.gitignore` ✅
- Won't be committed to version control ✅

### 🔐 Additional Security Steps:

#### 1. File Permissions (macOS/Linux)

```bash
chmod 600 config.json  # Read/write for owner only
```

#### 2. Environment-Specific Configurations

For production/CI environments, use environment variables instead:

```python
from azureml_example.core import Workspace

# Using environment variables
ws = Workspace(
    subscription_id=os.environ['AZUREML_SUBSCRIPTION_ID'],
    resource_group=os.environ['AZUREML_RESOURCE_GROUP'],
    workspace_name=os.environ['AZUREML_WORKSPACE_NAME']
)
```

#### 3. Service Principal Authentication (Production)

For automated deployments, use service principal authentication:

```python
from azureml_example.core.authentication import ServicePrincipalAuthentication

svc_pr = ServicePrincipalAuthentication(
    tenant_id=os.environ['AZURE_TENANT_ID'],
    service_principal_id=os.environ['AZURE_CLIENT_ID'],
    service_principal_password=os.environ['AZURE_CLIENT_SECRET']
)

ws = Workspace(
    subscription_id=os.environ['AZUREML_SUBSCRIPTION_ID'],
    resource_group=os.environ['AZUREML_RESOURCE_GROUP'],
    workspace_name=os.environ['AZUREML_WORKSPACE_NAME'],
    auth=svc_pr
)
```

## 📥 How to Download config.json

1. **Go to Azure ML Studio**: https://ml.azure.com
2. **Select your workspace** from the workspace dropdown
3. **Click the download icon** (⬇️) next to your workspace name in the top navigation
4. **Save the file** as `config.json`

The downloaded file should look like this:

```json
{
  "subscription_id": "your-subscription-id",
  "resource_group": "your-resource-group-name",
  "workspace_name": "your-workspace-name"
}
```

## 🧪 Testing Your Configuration

### Automated Setup and Test

```bash
# Run the setup script
python setup_config.py

# Run the test
python azureml_example/tests/test_azureml.py
```

### Manual Testing

```bash
# From project root
python azureml_example/tests/test_azureml.py
```

### Test Components

The test script will check:

1. **✅ Azure ML SDK Installation**

   - Core modules import correctly
   - Training estimators available
   - Pipeline components accessible

2. **✅ Configuration File Detection**

   - Searches multiple locations for config.json
   - Validates file structure
   - Reports found location

3. **✅ Workspace Connection**
   - Connects to your Azure ML workspace
   - Displays workspace information
   - Verifies authentication

### Expected Output

**Successful Test:**

```
Testing Azure ML Installation
========================================
✅ Azure ML Core version: 1.48.0
✅ Azure ML Core classes imported successfully
✅ Azure ML Training estimators imported successfully
✅ Azure ML Pipeline components imported successfully
⚠️  AutoML components skipped (has dependency conflicts)
✅ Found config.json at: ./config.json
🔄 Attempting to connect to Azure ML workspace...
✅ Successfully connected to workspace: my-workspace
   Subscription: 12345678-1234-1234-1234-123456789012
   Resource Group: my-resource-group
   Location: eastus

========================================
✅ Azure ML installation and configuration are working!

Next steps:
1. Create your first experiment
2. Upload data to your workspace
3. Train your first model
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. FileNotFoundError

```
❌ config.json not found
```

**Solution:** Download config.json from Azure ML Studio and place it in project root

#### 2. Authentication Error

```
❌ Workspace connection error: AuthenticationException
```

**Solutions:**

- Check if you're logged into Azure CLI: `az login`
- Verify config.json has correct subscription ID
- Ensure you have access to the workspace

#### 3. Permission Error

```
❌ Workspace connection error: Forbidden
```

**Solutions:**

- Check your Azure ML workspace permissions
- Ensure you're using the correct subscription
- Contact your Azure administrator

#### 4. Network Error

```
❌ Workspace connection error: ConnectionError
```

**Solutions:**

- Check internet connectivity
- Verify corporate firewall settings
- Try using VPN if on restricted network

## 🌍 Different Environments

### Development

```
./config.json  # Local development workspace
```

### Staging

```python
# Use environment variables
ws = Workspace.from_config(path="./config-staging.json")
```

### Production

```python
# Use service principal authentication
ws = Workspace(
    subscription_id=os.environ['AZUREML_SUBSCRIPTION_ID'],
    resource_group=os.environ['AZUREML_RESOURCE_GROUP'],
    workspace_name=os.environ['AZUREML_WORKSPACE_NAME'],
    auth=svc_pr
)
```

## 📝 Best Practices

1. **Never commit config.json** to version control
2. **Use different workspaces** for dev/staging/prod
3. **Use service principal auth** for automated deployments
4. **Set restrictive file permissions** on config files
5. **Regularly rotate** service principal secrets
6. **Monitor workspace access** and usage
7. **Use Azure Key Vault** for production secrets

## 🔗 Next Steps

After successful configuration:

1. **Create your first experiment**
2. **Upload training data**
3. **Set up compute targets**
4. **Train your first model**
5. **Deploy models to endpoints**

## 📚 Additional Resources

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)
- [Authentication Guide](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication)
