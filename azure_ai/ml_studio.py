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
"""

import glob
import io
import os
import re
import tempfile
from logging import getLogger

import kaggle
import pandas as pd
from azure.ai.ml import Input, MLClient, Output, automl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    AccountKeyConfiguration,
    AmlCompute,
    AzureBlobDatastore,
    BatchEndpoint,
    BatchRetrySettings,
    Compute,
    Data,
    Job,
    Model,
    ModelBatchDeployment,
    SasTokenConfiguration,
    Workspace,
)
from azure.ai.ml.exceptions import ValidationException
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv

logger = getLogger(__name__)

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


class AzureAIMLWorkspace:
    """
    Azure AI ML Studio Workspace class.

    Class to encapsulate Azure ML example functionality.

    Every helper class below inherits ml_client, the v2 SDK's entry point for
    all workspace operations (data, compute, jobs, models, endpoints).
    """

    ml_client: MLClient
    workspace: Workspace | None

    def __init__(self):
        """Initialize the AzureMLExample class."""
        load_dotenv()
        self.ml_client = self.connect_workspace()
        self.workspace = self.ml_client.workspaces.get(self.ml_client.workspace_name)

    def connect_workspace(self) -> MLClient:
        """
        Connect to Azure ML workspace.

        Reads config.json from the current directory or one of its parents, and
        authenticates with DefaultAzureCredential (for example, your `az login`).

        Returns:
            MLClient: A client for the workspace in config.json
        """
        try:
            return MLClient.from_config(credential=DefaultAzureCredential())
        except ValidationException as e:
            logger.error("Failed to connect to Azure ML workspace: %s", e)
            raise e


class AzureAIMLAssetsDataset(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Assets -> Data class.

    Class to encapsulate Azure ML data asset functionality.

    args:
        dataset_name: Name of the data asset in Azure ML workspace
        source_data: pandas DataFrame to register as data asset (optional)
        kaggle_dataset: Kaggle dataset name to download if data asset doesn't exist (optional)
                        example: whenamancodes/student-performance
        file_name: Pandas readable local file to register as data asset (optional)
                   example: student-mat.csv
        description: Description for the data asset (optional)
    """

    dataset: Data | None = None

    def __init__(
        self,
        dataset_name: str,
        *args,
        source_data: pd.DataFrame | None = None,
        kaggle_dataset: str | None = None,
        file_name: str | None = None,
        description: str | None = "Dataset created from DataFrame or Kaggle dataset",
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.source_data = source_data
        self.kaggle_dataset = kaggle_dataset
        self.file_name = file_name
        self.description = description

        if dataset_name is not None or source_data is not None or kaggle_dataset is not None or file_name is not None:
            self.dataset = self.get_or_create()

    @classmethod
    def get_default_name(cls, ml_client: MLClient) -> str:
        """
        Get the name of the first data asset in the workspace.

        Args:
            ml_client: Azure ML workspace client

        Returns:
            str: Name of the first data asset

        Raises:
            ValueError: If no data assets are found in the workspace
        """
        try:
            for dataset in ml_client.data.list():
                if dataset.name:
                    logger.info("Found first dataset: %s", dataset.name)
                    return dataset.name

            raise ValueError("No datasets found in the workspace")

        except Exception as e:
            logger.error("Failed to get first dataset name: %s", e)
            raise e

    def get_or_create(self) -> Data:
        """
        Get or create an Azure ML data asset from a pandas DataFrame, a local file or a Kaggle dataset.

        A new data asset is an MLTable: a folder holding the data as a CSV file
        plus an MLTable file that describes how to read it. The SDK uploads the
        folder to the workspace's default datastore.

        Returns:
            Data: The latest version of the Azure ML data asset

        Raises:
            ValueError: If none of source_data, kaggle_dataset or file_name is provided
            Exception: If data asset creation fails
        """
        if self.dataset is not None:
            logger.info("Using existing dataset: %s", self.dataset_name)
            return self.dataset

        try:
            dataset = self.ml_client.data.get(name=self.dataset_name, label="latest")
            logger.info("Found existing dataset: %s", self.dataset_name)
            return dataset
        except ResourceNotFoundError:
            logger.info("Dataset not found, creating new dataset: %s", self.dataset_name)
        except Exception as e:
            logger.error("Failed to get or create dataset %s: %s", self.dataset_name, e)
            raise e

        # ----------------------------------------------------------------------
        # eval the source data
        # ----------------------------------------------------------------------
        if self.source_data is not None:
            df = self.source_data
            logger.info("Using provided DataFrame with shape: %s", df.shape)
        elif self.kaggle_dataset:
            df = self.from_kaggle()
        elif self.file_name:
            df = self.from_file(self.file_name)
        else:
            raise ValueError("Either source_data or kaggle_dataset or file_name must be provided")
        if df is None or df.empty:
            raise ValueError(f"File {self.file_name} does not contain valid data")

        # ----------------------------------------------------------------------
        # write the CSV file and its MLTable file to a folder, then register
        # the folder as a data asset. create_or_update() uploads it.
        # ----------------------------------------------------------------------
        description = self.description
        if not description and self.source_data is not None:
            description = "Dataset created from DataFrame"
        if not description:
            description = f"Dataset created from {self.kaggle_dataset or self.file_name}"

        with tempfile.TemporaryDirectory() as upload_dir:
            csv_filename = f"{os.path.basename(self.dataset_name)}.csv"
            csv_path = os.path.join(upload_dir, csv_filename)
            df.to_csv(csv_path, index=False)

            mltable = MLTABLE_TEMPLATE.format(csv_filename=csv_filename)
            mltable_path = os.path.join(upload_dir, "MLTable")
            with open(mltable_path, "w", encoding="utf-8") as mltable_file:
                mltable_file.write(mltable)
            logger.info("Saved DataFrame and MLTable file to temporary folder: %s", upload_dir)

            data_asset = Data(
                name=self.dataset_name,
                path=upload_dir,
                type=AssetTypes.MLTABLE,
                description=description,
            )
            dataset = self.ml_client.data.create_or_update(data_asset)

        logger.info("Successfully created and registered dataset: %s version %s", dataset.name, dataset.version)
        return dataset

    def dataset_to_dataframe(self) -> pd.DataFrame:
        """
        Convert the latest version of the Azure ML data asset to a pandas DataFrame.

        Downloads the data asset's CSV file from its datastore with the Azure
        Blob Storage client. (Microsoft's mltable and azureml-fsspec packages
        do this too, but they don't support Python 3.14.)

        Returns:
            pd.DataFrame: The data asset as a pandas DataFrame

        Raises:
            ValueError: If the data asset is not stored in a datastore, or holds no CSV file
        """
        dataset = self.ml_client.data.get(name=self.dataset_name, label="latest")
        dataset_path = str(dataset.path or "")

        # e.g. azureml://subscriptions/.../datastores/workspaceblobstore/paths/LocalUpload/<hash>/<folder>/
        match = re.search(r"datastores/(?P<datastore>[^/]+)/paths/(?P<path>.*)$", dataset_path)
        if not match:
            raise ValueError(f"Dataset {self.dataset_name} is not stored in a datastore: {dataset_path}")
        datastore_name = match.group("datastore")
        folder = match.group("path")

        container = self.get_datastore_container(datastore_name)
        for blob in container.list_blobs(name_starts_with=folder):
            if blob.name.endswith(".csv"):
                logger.info("Downloading dataset file: %s", blob.name)
                csv_bytes = container.download_blob(blob.name).readall()
                return pd.read_csv(io.BytesIO(csv_bytes))

        raise ValueError(f"Dataset {self.dataset_name} does not contain a CSV file")

    def get_datastore_container(self, datastore_name: str) -> ContainerClient:
        """
        Get a Blob Storage client for the container behind an Azure ML datastore.

        Uses the datastore's own account key or SAS token when it has one (the
        workspace's default datastore does), and your Azure login otherwise.

        Args:
            datastore_name: Name of the datastore, e.g. workspaceblobstore

        Returns:
            ContainerClient: A client for the datastore's blob container

        Raises:
            TypeError: If the datastore is not Azure Blob Storage
        """
        datastore = self.ml_client.datastores.get(datastore_name, include_secrets=True)
        if not isinstance(datastore, AzureBlobDatastore):
            raise TypeError(f"Datastore {datastore_name} is not of type AzureBlobDatastore")
        credential: str | DefaultAzureCredential | None = None
        if isinstance(datastore.credentials, AccountKeyConfiguration):
            credential = datastore.credentials.account_key
        if isinstance(datastore.credentials, SasTokenConfiguration):
            credential = datastore.credentials.sas_token
        if not credential:
            credential = DefaultAzureCredential()

        account_url = f"{datastore.protocol}://{datastore.account_name}.blob.{datastore.endpoint}"
        return ContainerClient(
            account_url=account_url,
            container_name=datastore.container_name,
            credential=credential,
        )

    def from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a dataset from a local file path as a pandas DataFrame.

        Args:
            file_path: Path to the local CSV file

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame

        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If loading fails
        """
        logger.info("Loading dataset from file: %s", file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError as e:
            logger.error("Failed to decode file %s: %s", file_path, e)
            logger.info("Trying to load as Excel file")

            try:
                df = pd.read_excel(file_path)
            except Exception as e2:
                logger.error("Failed to load file %s as Excel: %s", file_path, e2)
                raise e2

        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", file_path, e)
            raise e

        logger.info("Successfully loaded dataset from %s with shape: %s", file_path, df.shape)
        return df

    def from_kaggle(self, dataset: str | None = None) -> pd.DataFrame:
        """
        Download and load a Kaggle dataset as a pandas DataFrame.

        Args:
            dataset: Kaggle dataset name, e.g. 'heptapod/titanic' (optional,
                     defaults to the kaggle_dataset given to the constructor)

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame

        Raises:
            Exception: If dataset download or loading fails
        """
        self.kaggle_dataset = dataset or self.kaggle_dataset
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info("Downloading Kaggle dataset: %s", self.kaggle_dataset)

                kaggle.api.dataset_download_files(dataset=self.kaggle_dataset, path=temp_dir, unzip=True)

                csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))

                if not csv_files:
                    raise ValueError(f"No CSV files found in dataset: {self.kaggle_dataset}")

                for csv_file in csv_files:
                    logger.info("Found CSV file: %s", csv_file)
                    return self.from_file(csv_file)

                raise ValueError(f"No CSV files found in dataset {self.kaggle_dataset} download: {csv_files}")

        except Exception as e:
            logger.error("Failed to load Kaggle dataset %s: %s", self.dataset_name, e)
            raise e

    def delete(self):
        """
        Archive every version of the Azure ML data asset.

        The v2 SDK cannot delete a data asset. Archiving hides it from the
        studio and from list(), and ml_client.data.restore() brings it back.
        """
        try:
            self.ml_client.data.archive(name=self.dataset_name)
            logger.info("Successfully archived dataset: %s", self.dataset_name)
        except ResourceNotFoundError as e:
            logger.warning("Did not find dataset %s: %s", self.dataset_name, e)


class AzureAIMLStudioComputeCluster(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Compute -> Cluster class.

    Class to encapsulate Azure ML example functionality.
    """

    cluster: AmlCompute | None = None

    def __init__(
        self,
        *args,
        cluster_name: str | None = None,
        vm_size: str = "STANDARD_DS3_V2",
        min_nodes: int = 0,
        max_nodes: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cluster_name = cluster_name or AzureAIMLStudioComputeCluster.get_default_name(self.ml_client)
        self.vm_size = vm_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.cluster = self.get_or_create()

    @classmethod
    def get_default_name(cls, ml_client: MLClient) -> str:
        """
        Get the name of the first compute cluster in the workspace.

        Args:
            ml_client: Azure ML workspace client

        Returns:
            str: Name of the first compute cluster

        Raises:
            ValueError: If no compute clusters are found in the workspace
        """
        try:
            for compute in ml_client.compute.list():
                # Skip compute that isn't a cluster, e.g. a compute instance
                if compute.type != "amlcompute":
                    continue
                if compute.name:
                    logger.info("Found first compute cluster: %s", compute.name)
                    return compute.name

            raise ValueError("No compute clusters found in the workspace")

        except Exception as e:
            logger.error("Failed to get first cluster name: %s", e)
            raise e

    def get_or_create(self) -> AmlCompute:
        """
        Get or create an Azure ML compute cluster by name.

        Uses the cluster_name, vm_size, min_nodes and max_nodes given to the constructor.

        Returns:
            AmlCompute: The compute cluster

        Raises:
            TypeError: If the workspace's compute with this name is not a cluster, e.g. a compute instance
        """
        if self.cluster is not None:
            logger.info("Using existing cluster: %s", self.cluster_name)
            return self.cluster

        try:
            cluster = self.ml_client.compute.get(self.cluster_name)
            logger.info("Found existing cluster: %s", self.cluster_name)
        except ResourceNotFoundError:
            cluster = self.create()
        except Exception as e:
            logger.error("Failed to get cluster %s: %s", self.cluster_name, e)
            raise e

        if not isinstance(cluster, AmlCompute):
            raise TypeError(f"Compute {self.cluster_name} is a {cluster.type}, not a compute cluster")
        return cluster

    def create(self) -> Compute:
        """
        Create an Azure ML compute cluster, and wait until it's ready.

        Uses the cluster_name, vm_size, min_nodes and max_nodes given to the constructor.

        Returns:
            Compute: The new compute cluster
        """
        logger.info("Creating new cluster: %s", self.cluster_name)
        compute_config = AmlCompute(
            name=self.cluster_name,
            size=self.vm_size,
            min_instances=self.min_nodes,
            max_instances=self.max_nodes,
            idle_time_before_scale_down=300,
        )

        try:
            poller = self.ml_client.compute.begin_create_or_update(compute_config)
            cluster = poller.result()
        except Exception as e:
            logger.error("Failed to create cluster %s: %s", self.cluster_name, e)
            raise e

        logger.info("Successfully created cluster: %s", self.cluster_name)
        return cluster


class AzureAIMLStudioAssetsBatchEndpoint(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Assets -> Endpoint class.

    Class to encapsulate Azure ML assets batch endpoint functionality.
    """

    endpoint: BatchEndpoint | None = None

    def __init__(
        self,
        endpoint_name: str,
        model_name: str,
        *args,
        compute_target_name: str | None = None,
        deployment_name: str | None = None,
        instance_count: int = 2,
        max_concurrency_per_instance: int = 1,
        mini_batch_size: int = 10,
        retry_settings_max_retries: int = 3,
        retry_settings_timeout: int = 300,
        description: str | None = "Batch endpoint for model inference",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.compute_target_name = compute_target_name or AzureAIMLStudioComputeCluster.get_default_name(self.ml_client)
        self.deployment_name = deployment_name or f"{model_name}-deployment"
        self.instance_count = instance_count
        self.max_concurrency_per_instance = max_concurrency_per_instance
        self.mini_batch_size = mini_batch_size
        self.retry_settings_max_retries = retry_settings_max_retries
        self.retry_settings_timeout = retry_settings_timeout
        self.description = description

        self.endpoint = self.get_or_create()

    @classmethod
    def get_default_name(cls, ml_client: MLClient) -> str | None:
        """
        Get the name of the first batch endpoint in the workspace.

        Args:
            ml_client: Azure ML workspace client

        Returns:
            str: Name of the first batch endpoint

        Raises:
            ValueError: If no batch endpoints are found in the workspace
        """
        try:
            batch_endpoints = list(ml_client.batch_endpoints.list())

            if not batch_endpoints:
                raise ValueError("No batch endpoints found in the workspace")

            first_endpoint_name = batch_endpoints[0].name
            logger.info("Found first batch endpoint: %s", first_endpoint_name)
            return first_endpoint_name

        except Exception as e:
            logger.error("Failed to get first batch endpoint name: %s", e)
            raise e

    def get_or_create(self) -> BatchEndpoint:
        """
        Get or create an Azure ML batch endpoint by name.

        Returns:
            BatchEndpoint: The batch endpoint

        Raises:
            Exception: If endpoint creation fails
        """
        if self.endpoint is not None:
            logger.info("Using existing batch endpoint: %s", self.endpoint_name)
            return self.endpoint

        if not self.endpoint_name:
            raise ValueError("Endpoint name must be provided")

        try:
            logger.info("Looking for existing batch endpoint: %s", self.endpoint_name)
            endpoint = self.ml_client.batch_endpoints.get(name=self.endpoint_name)
            logger.info("Found existing batch endpoint: %s", self.endpoint_name)
            return endpoint
        # pylint: disable=W0718
        except Exception:
            logger.warning("Batch endpoint not found, creating new batch endpoint: %s", self.endpoint_name)

        if not self.description:
            self.description = f"Batch endpoint for {self.model_name} model"
        if not self.endpoint_name:
            raise ValueError("Endpoint name must be provided")

        try:
            logger.info("Creating batch endpoint: %s", self.endpoint_name)
            endpoint = BatchEndpoint(
                name=self.endpoint_name,
                description=self.description,
            )
            endpoint_poller = self.ml_client.batch_endpoints.begin_create_or_update(endpoint)
            endpoint = endpoint_poller.result()
            logger.info("Successfully created batch endpoint: %s", self.endpoint_name)

            # Create batch deployment
            self._create_batch_deployment(endpoint)

            return endpoint

        except Exception as e:
            logger.error("Failed to create batch endpoint %s: %s", self.endpoint_name, e)
            raise e

    def _create_batch_deployment(self, endpoint: BatchEndpoint):
        """
        Create a batch deployment for the endpoint.

        Args:
            endpoint: The batch endpoint to deploy to
        """
        try:
            logger.info("Looking for model: %s", self.model_name)
            model = self.ml_client.models.get(name=self.model_name, label="latest")
            logger.info("Retrieved model: %s", self.model_name)

            logger.info("Verifying compute target: %s", self.compute_target_name)
            AzureAIMLStudioComputeCluster(
                cluster_name=self.compute_target_name, vm_size="STANDARD_DS3_V2", min_nodes=0, max_nodes=4
            ).get_or_create()

            # Create batch deployment
            logger.info("Creating batch deployment: %s", self.deployment_name)
            model_batch_deployment = ModelBatchDeployment(
                name=self.deployment_name,
                description=f"Batch deployment for {self.model_name}",
                endpoint_name=self.endpoint_name,
                model=model,
                compute=self.compute_target_name,
                instance_count=self.instance_count,
                max_concurrency_per_instance=self.max_concurrency_per_instance,
                mini_batch_size=self.mini_batch_size,
                retry_settings=BatchRetrySettings(
                    max_retries=self.retry_settings_max_retries, timeout=self.retry_settings_timeout
                ),
                environment_variables={"AZUREML_MODEL_DIR": "$AZUREML_MODEL_DIR"},
            )
            logger.info("Created model batch deployment object: %s", self.deployment_name)

            try:
                logger.info("Creating batch deployment: %s", self.deployment_name)
                model_batch_deployment = self.ml_client.batch_deployments.begin_create_or_update(
                    model_batch_deployment
                ).result()
                logger.info("Successfully created batch deployment: %s", self.deployment_name)
            except ResourceNotFoundError as e:
                logger.error("Failed to create batch deployment %s: %s", self.deployment_name, e)
                raise e

            # Set as default deployment
            logger.info("Setting %s as default deployment for endpoint %s", self.deployment_name, self.endpoint_name)
            endpoint.defaults = {"deployment_name": self.deployment_name}
            self.ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
            logger.info("Set %s as default deployment for endpoint %s", self.deployment_name, self.endpoint_name)

        except Exception as e:
            logger.error("Failed to create batch deployment %s: %s", self.deployment_name, e)
            raise e

    def invoke_batch_job(self, input_data_path: str, output_data_path: str | None = None):
        """
        Invoke a batch inference job on the endpoint.

        Args:
            input_data_path: Path to input data for batch inference
            output_data_path: Path for output data (optional)

        Returns:
            BatchJob: The batch job object
        """
        try:
            logger.info("Invoking batch job on endpoint: %s", self.endpoint_name)

            inputs = {"input_data": Input(type="uri_folder", path=input_data_path)}

            # Create outputs configuration if provided
            outputs = None
            if output_data_path:
                outputs = {"output_data": Output(type="uri_file", path=output_data_path)}

            # Submit the batch job using the invoke method
            job = self.ml_client.batch_endpoints.invoke(
                endpoint_name=self.endpoint_name, deployment_name=self.deployment_name, inputs=inputs, outputs=outputs
            )

            logger.info("Successfully invoked batch job: %s", job.name)
            return job

        except Exception as e:
            logger.error("Failed to invoke batch job on endpoint %s: %s", self.endpoint_name, e)
            raise e

    def delete(self):
        """
        Delete the Azure ML batch endpoint.

        Raises:
            Exception: If endpoint deletion fails
        """
        try:
            self.ml_client.batch_endpoints.begin_delete(name=self.endpoint_name).result()
            logger.info("Successfully deleted batch endpoint: %s", self.endpoint_name)
        # pylint: disable=W0718
        except Exception as e:
            logger.warning("Could not delete batch endpoint %s: %s", self.endpoint_name, e)


class AzureAIMLStudioAuthoringAutomatedML(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Authoring -> AutomatedML class.

    Class to encapsulate Azure AutoML functionality.
    """

    def __init__(self, dataset_name: str, cluster_name: str, **kwargs):
        """
        Azure AI ML Studio Authoring -> AutomatedML class.

        Initialize the AzureAIMLStudioAuthoringAutomatedML class.
        """
        super().__init__()
        assets_dataset = AzureAIMLAssetsDataset(dataset_name=dataset_name, **kwargs)
        compute_cluster = AzureAIMLStudioComputeCluster(cluster_name=cluster_name, **kwargs)

        # get_or_create() returns what each constructor already fetched or created
        self.dataset: Data = assets_dataset.get_or_create()
        self.cluster: AmlCompute = compute_cluster.get_or_create()

    def run_automl_experiment(
        self,
        experiment_name: str,
        target_column: str,
        task: str = "classification",
        primary_metric: str = "accuracy",
        max_concurrent_iterations: int = 4,
        max_cores_per_iteration: int = -1,
        test_size: float = 0.2,
        cv_folds: int = 5,
        enable_early_stopping: bool = True,
        iteration_timeout_minutes: int = 20,
        experiment_timeout_hours: float = 1.0,
        time_column_name: str | None = None,
        enable_onnx_compatible_models: bool = True,
    ) -> Job:
        """
        Submit an AutoML job, and return without waiting for it to finish.

        Args:
            experiment_name: Name of the experiment that groups the job's runs
            target_column: Name of the target column for prediction
            task: ML task type ('classification', 'regression', 'forecasting')
            primary_metric: Primary metric to optimize
            max_concurrent_iterations: Maximum concurrent AutoML trials
            max_cores_per_iteration: Cores per trial (-1 for all available)
            test_size: Proportion of data held out for testing the best model
            cv_folds: Number of cross-validation folds
            enable_early_stopping: Whether to end the job early when scores stop improving
            iteration_timeout_minutes: Timeout per trial in minutes
            experiment_timeout_hours: Total job timeout in hours
            time_column_name: Name of the time column (required for 'forecasting')
            enable_onnx_compatible_models: Whether to try only algorithms that AutoML can
                export to ONNX, and save each trial's model as outputs/model.onnx too.
                The ONNX model is what titanic-survival-app serves on Python 3.14.

        Returns:
            Job: The submitted AutoML job

        Raises:
            ValueError: If the task is not supported
        """
        if task not in AUTOML_TASKS:
            raise ValueError(f"Unsupported AutoML task: {task}. Use one of {', '.join(AUTOML_TASKS)}")

        create_automl_job = AUTOML_TASKS[task]
        training_data_path = f"azureml:{self.dataset.name}:{self.dataset.version}"
        training_data = Input(type=AssetTypes.MLTABLE, path=training_data_path)
        timeout_minutes = int(experiment_timeout_hours * 60)

        try:
            # Configure the AutoML job. Featurization is automatic by default.
            automl_job = create_automl_job(
                compute=self.cluster.name,
                experiment_name=experiment_name,
                training_data=training_data,
                target_column_name=target_column,
                primary_metric=primary_metric,
                n_cross_validations=cv_folds,
                test_data_size=test_size,
            )
            automl_job.set_limits(
                timeout_minutes=timeout_minutes,
                trial_timeout_minutes=iteration_timeout_minutes,
                max_concurrent_trials=max_concurrent_iterations,
                max_cores_per_trial=max_cores_per_iteration,
                enable_early_termination=enable_early_stopping,
            )
            automl_job.set_training(enable_onnx_compatible_models=enable_onnx_compatible_models)
            if task == "forecasting":
                automl_job.set_forecast_settings(time_column_name=time_column_name)

            # Submit the job
            logger.info("Submitting AutoML job for experiment: %s", experiment_name)
            job = self.ml_client.jobs.create_or_update(automl_job)

            logger.info("AutoML job submitted successfully. Job name: %s", job.name)
            return job

        except Exception as e:
            logger.error("Failed to run AutoML experiment %s: %s", experiment_name, e)
            raise e

    def get_best_model(self, job: Job) -> Job:
        """
        Wait for an AutoML job to finish, then get its best trial.

        Args:
            job: The AutoML job returned by run_automl_experiment()

        Returns:
            Job: The child job (trial) that trained the best model

        Raises:
            ValueError: If the job was never submitted, or did not report a best trial
        """
        if not job.name:
            raise ValueError("The AutoML job has no name. Submit it with run_automl_experiment() first.")

        try:
            # Wait for job completion, printing its log as it runs
            self.ml_client.jobs.stream(job.name)
            job = self.ml_client.jobs.get(job.name)

            # AutoML records the best trial on the parent job, as a tag or a property
            tags = job.tags or {}
            properties = job.properties or {}
            best_child_name = tags.get("automl_best_child_run_id")
            if not best_child_name:
                best_child_name = properties.get("automl_best_child_run_id")
            if not best_child_name:
                raise ValueError(f"AutoML job {job.name} finished with status {job.status} and no best trial")

            best_job = self.ml_client.jobs.get(best_child_name)
            best_job_properties = best_job.properties or {}
            algorithm = best_job_properties.get("run_algorithm", "Unknown")
            logger.info("Best trial: %s", best_job.name)
            logger.info("Best model algorithm: %s", algorithm)

            return best_job

        except Exception as e:
            logger.error("Failed to get best model from job %s: %s", job.name, e)
            raise e

    def register_best_model(
        self,
        job: Job,
        model_name: str,
        description: str | None = None,
        tags: dict | None = None,
        model_format: str = "mlflow",
    ) -> Model:
        """
        Register the best model from an AutoML job.

        AutoML saves each trial's model in MLflow format, which batch
        endpoints can serve without a scoring script, and, when the job ran with
        enable_onnx_compatible_models, in ONNX format, which titanic-survival-app
        serves from an Azure Function.

        Args:
            job: The AutoML job returned by run_automl_experiment()
            model_name: Name to register the model with
            description: Description for the model
            tags: Tags to associate with the model
            model_format: 'mlflow' or 'onnx'

        Returns:
            Model: The registered model

        Raises:
            ValueError: If the model format is not supported
        """
        if model_format not in MODEL_FORMATS:
            raise ValueError(f"Unsupported model format: {model_format}. Use one of {', '.join(MODEL_FORMATS)}")
        artifact_path, model_type = MODEL_FORMATS[model_format]
        description = description or "Best model from AutoML experiment"
        tags = tags or {}

        try:
            best_job = self.get_best_model(job)
            logger.info(
                "Registering best %s model from job %s with name: %s, best trial: %s",
                model_format,
                job.name,
                model_name,
                best_job.name,
            )

            # Register the model
            model_path = f"azureml://jobs/{best_job.name}/outputs/artifacts/{artifact_path}"
            model = Model(
                name=model_name,
                path=model_path,
                type=model_type,
                description=description,
                tags=tags,
            )
            model = self.ml_client.models.create_or_update(model)

            logger.info("Successfully registered model: %s version %s", model.name, model.version)
            return model

        except Exception as e:
            logger.error("Failed to register model %s: %s", model_name, e)
            raise e

    def show_run_details(self, job: Job) -> str | None:
        """
        Get the link to an AutoML job in Azure ML studio.

        The studio shows the job's trials and metrics as it runs. (The v1 SDK's
        Jupyter widget has no v2 equivalent.)

        Args:
            job: The AutoML job to display

        Returns:
            str: The job's Azure ML studio url, or None if the job was never submitted
        """
        logger.info("AutoML job %s: %s", job.name, job.studio_url)
        return job.studio_url
