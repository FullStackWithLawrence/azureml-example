# Azure AutoML Setup Guide: Titanic Survival Prediction

## Overview

This guide walks you through setting up an Azure AutoML regression job to predict survival on the Titanic dataset. We'll cover data preparation, compute setup, job configuration, and timing expectations for a 1,300-row dataset.

## 🎯 Project Goals

- **Dataset**: Titanic passenger data (1,309 passengers)
- **Task**: Binary classification (survival prediction)
- **Method**: Azure AutoML with multiple algorithm comparison
- **Compute**: Small cluster optimized for learning/experimentation

## 📊 Dataset Preparation

### Download Titanic Dataset from Kaggle

1. **Create Kaggle Account** (if you don't have one):

   - Go to [kaggle.com](https://www.kaggle.com)
   - Sign up for a free account

2. **Get Kaggle API Credentials**:

   ```console
   # Install Kaggle CLI
   pip install kaggle

   # Go to kaggle.com/settings and download kaggle.json
   # Place it in ~/.kaggle/kaggle.json
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the Dataset**:

   ```console
   # Create data directory
   mkdir -p data/titanic
   cd data/titanic

   # Download Titanic competition data
   kaggle competitions download -c titanic

   # Extract files
   unzip titanic.zip
   ls -la
   # Should see: train.csv, test.csv, gender_submission.csv
   ```

4. **Dataset Overview**:

   ```console
   # Quick look at the data
   head -5 train.csv
   wc -l *.csv
   ```

   **Expected output**:

   ```console
   train.csv: 892 rows (including header)
   test.csv: 419 rows (including header)
   ```

### Upload to Azure ML Datastore

```console
# Upload training data to Azure ML
az ml data create \
  --name titanic-train \
  --version 1 \
  --type uri_file \
  --path ./data/titanic/train.csv \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10
```

## 🖥️ Compute Setup

### Create Tiny Cluster for Small Datasets

```console
# Create cost-effective cluster for experimentation
az ml compute create \
  --name tiny-cluster \
  --type amlcompute \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --size Standard_D2s_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --idle-time-before-scale-down 120
```

```console
# Create a GPU cluster. Warning: you might encounter quota availability problems depending on the region.
az ml compute create \
  --name gpu-cluster \
  --type AmlCompute \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --size STANDARD_NC4AS_T4_V3 \
  --min-instances 0 \
  --max-instances 1 \
  --idle-time-before-scale-down 120
```

**Why Standard_D2s_v3?**

- ✅ 2 vCPUs, 8GB RAM - perfect for small datasets
- ✅ Fits within common quota limits (6 vCPU default)
- ✅ Cost-effective for learning/experimentation
- ✅ Sufficient compute power for 891 training samples

### Verify Cluster Status

```console
# Check cluster creation
az ml compute show \
  --name tiny-cluster \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --query "{name:name, size:size, state:provisioning_state, maxInstances:scale_settings.maximum_node_count}"
```

## 🤖 AutoML Job Configuration

### Create AutoML Job YAML

Create `jobs/titanic-automl.yml`:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/autoMLJob.schema.json
type: automl

experiment_name: titanic-survival-prediction
description: "AutoML classification to predict Titanic passenger survival"

compute: azureml:tiny-cluster

task: classification
primary_metric: accuracy

target_column_name: Survived
training_data:
  type: uri_file
  path: azureml:titanic-train:1

limits:
  timeout_minutes: 30
  trial_timeout_minutes: 5
  max_trials: 15
  max_concurrent_trials: 1
  enable_early_termination: true

training:
  blocked_training_algorithms:
    - ExtremeRandomTrees # Can be slow on small datasets
  enable_onnx_compatible_models: true
  enable_vote_ensemble: true
  enable_stack_ensemble: true

featurization:
  mode: auto
  drop_columns:
    - PassengerId
    - Name
    - Ticket
```

### Submit the Job

```console
# Submit AutoML job
az ml job create \
  --file jobs/titanic-automl.yml \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10
```

## ⏱️ Timing Expectations

### Expected Duration: **20-40 minutes total**

| Phase                 | Duration  | Description                                |
| --------------------- | --------- | ------------------------------------------ |
| **Cluster Startup**   | 3-5 min   | Cold start, VM provisioning                |
| **Data Preparation**  | 2-3 min   | Data validation, featurization             |
| **Model Training**    | 15-25 min | Multiple algorithms, hyperparameter tuning |
| **Ensemble Creation** | 2-5 min   | Voting/stacking ensembles                  |
| **Finalization**      | 1-2 min   | Model registration, cleanup                |

### Performance Factors

**Why it takes this long:**

- 🔄 **Algorithm Diversity**: AutoML tries 10-15 different algorithms
- 🎛️ **Hyperparameter Tuning**: Each algorithm gets multiple parameter combinations
- 🏗️ **Feature Engineering**: Automatic feature creation and selection
- 📊 **Cross-Validation**: 5-fold CV for robust evaluation
- 🎯 **Ensemble Methods**: Combines best models for final predictor

**Single Node Impact:**

- ⚠️ Models train sequentially (not parallel)
- ⚠️ 1 concurrent trial maximum
- ✅ Still efficient for small datasets
- ✅ Cost-effective for experimentation

### Speed Optimization Tips

**For faster experimentation:**

```yaml
# Quick AutoML for development
limits:
  timeout_minutes: 15 # Reduce total time
  trial_timeout_minutes: 3 # Shorter per-model limit
  max_trials: 8 # Fewer algorithms
  max_concurrent_trials: 1 # Single node anyway

training:
  blocked_training_algorithms:
    - ExtremeRandomTrees
    - AutoArima
    - Prophet # Block slower algorithms
```

**For production-quality results:**

```yaml
# Comprehensive AutoML for best model
limits:
  timeout_minutes: 60 # More time for thorough search
  trial_timeout_minutes: 10 # Allow complex models
  max_trials: 25 # Try more algorithms

training:
  enable_vote_ensemble: true
  enable_stack_ensemble: true
  ensemble_iterations: 5 # More ensemble combinations
```

## 📊 Monitoring Progress

### Azure ML Studio Dashboard

1. Navigate to [Azure ML Studio](https://ml.azure.com)
2. Select your workspace: **UBC-CDL10**
3. Go to **Jobs** → **titanic-survival-prediction**
4. Monitor real-time progress:
   - Models completed
   - Current best accuracy
   - Remaining time

### CLI Monitoring

```console
# Check job status
az ml job show \
  --name <job-name> \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10 \
  --query "{status:status, duration:duration}"

# Stream job logs
az ml job stream \
  --name <job-name> \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10
```

## 🎯 Expected Results

### Typical Performance on Titanic Dataset

- **Best Algorithm**: Usually XGBoost or LightGBM
- **Expected Accuracy**: 82-85%
- **Cross-Validation Score**: ~0.83
- **Feature Importance**: Age, Fare, Sex, Pclass as top predictors

### Model Artifacts

After completion, you'll have:

- ✅ Best performing model registered
- ✅ ONNX model for deployment
- ✅ Feature importance rankings
- ✅ Model explanations and visualizations
- ✅ Ensemble models for robust predictions

## 🚀 Next Steps

### Download Best Model

```console
# List registered models
az ml model list \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10

# Download the best model
az ml model download \
  --name <model-name> \
  --version 1 \
  --download-path ./models/ \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10
```

### Deploy for Testing

```console
# Create endpoint for model testing
az ml online-endpoint create \
  --name titanic-endpoint \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10
```

## 💡 Troubleshooting

### Common Issues

**"Quota Exceeded" Error:**

```console
# Scale down cluster if needed
az ml compute update \
  --name tiny-cluster \
  --max-instances 1 \
  --resource-group ubc-cdl10 \
  --workspace-name UBC-CDL10
```

**Job Taking Too Long:**

- Check if cluster is starting up (first 5 minutes)
- Reduce `max_trials` in YAML config
- Use `enable_early_termination: true`

**Poor Model Performance:**

- Check target column name matches exactly
- Verify no data leakage in features
- Review feature importance for insights

## 📚 Additional Resources

- [AutoML Classification Guide](https://docs.microsoft.com/azure/machine-learning/how-to-configure-auto-train)
- [Compute Instance Types](https://docs.microsoft.com/azure/virtual-machines/sizes)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Azure ML CLI Reference](https://docs.microsoft.com/cli/azure/ml)
