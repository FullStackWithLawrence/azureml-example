"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Basic Azure ML example, built on the Azure Machine Learning SDK v2 (azure-ai-ml),
showing how to
- Connect to Azure ML workspace
- Create a data asset from a pandas DataFrame, a local file or a Kaggle dataset
- Create a compute cluster
- Create a batch endpoint for model inference
- Run an AutoML job
- Register the best model from AutoML
- Invoke batch inference job on the endpoint
- Archive the data asset, and delete the batch endpoint

One module per Azure ML Studio area, named after the studio's menus:

- workspace: AzureAIMLWorkspace, the connection every other class inherits
- assets_dataset: AzureAIMLAssetsDataset (Assets -> Data)
- ml_studio_compute_cluster: AzureAIMLStudioComputeCluster (Compute -> Clusters)
- assets_batch_endpoint: AzureAIMLStudioAssetsBatchEndpoint (Assets -> Endpoints)
- authoring_automatedml: AzureAIMLStudioAuthoringAutomatedML (Authoring -> Automated ML)
- const, exceptions, utils: what the classes share

Import the classes from here: from azure_ai.ml_studio import AzureAIMLAssetsDataset
"""

from .assets_batch_endpoint import AzureAIMLStudioAssetsBatchEndpoint
from .assets_dataset import AzureAIMLAssetsDataset
from .authoring_automatedml import AzureAIMLStudioAuthoringAutomatedML
from .const import AUTOML_TASKS, MLTABLE_TEMPLATE, MODEL_FORMATS
from .exceptions import is_not_found
from .ml_studio_compute_cluster import AzureAIMLStudioComputeCluster
from .utils import parse_datastore_path, write_mltable_folder
from .workspace import AzureAIMLWorkspace

__all__ = [
    "AUTOML_TASKS",
    "MLTABLE_TEMPLATE",
    "MODEL_FORMATS",
    "AzureAIMLAssetsDataset",
    "AzureAIMLStudioAssetsBatchEndpoint",
    "AzureAIMLStudioAuthoringAutomatedML",
    "AzureAIMLStudioComputeCluster",
    "AzureAIMLWorkspace",
    "is_not_found",
    "parse_datastore_path",
    "write_mltable_folder",
]
