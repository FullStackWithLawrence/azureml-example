"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Constants shared by the Azure ML helper classes.
"""

from azure.ai.ml import automl
from azure.ai.ml.constants import AssetTypes

# An MLTable file tells Azure ML how to read the CSV file next to it. AutoML
# jobs in the v2 SDK only accept tabular training data in this format.
MLTABLE_TEMPLATE = """\
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable
paths:
  - file: ./{csv_filename}
transformations:
  - read_delimited:
      delimiter: ","
      encoding: utf8
      header: all_files_same_headers
"""

# AutoML task name -> the v2 SDK function that builds that kind of job
AUTOML_TASKS = {
    "classification": automl.classification,
    "regression": automl.regression,
    "forecasting": automl.forecasting,
}

# Model format -> (where an AutoML trial saves the model, the model asset type).
# "mlflow" serves from Azure ML endpoints. "onnx" runs anywhere onnxruntime
# does, including Python 3.14, where the AutoML runtime packages that a
# pickled AutoML model needs don't install. titanic-survival-app uses it.
MODEL_FORMATS = {
    "mlflow": ("outputs/mlflow-model/", AssetTypes.MLFLOW_MODEL),
    "onnx": ("outputs/model.onnx", AssetTypes.CUSTOM_MODEL),
}
