"""
Basic Azure ML example showing how to connect to workspace and create experiments.
"""

import glob
import os
import shutil
import tempfile
from logging import getLogger
from typing import Optional, Union

import kaggle
import pandas as pd
from azureml.core import Dataset, Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.run import Run
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_factory import FileDatasetFactory
from azureml.data.file_dataset import FileDataset
from azureml.data.tabular_dataset import TabularDataset
from azureml.exceptions import UserErrorException
from azureml.train.automl import AutoMLConfig
from azureml.widgets import RunDetails
from dotenv import load_dotenv


logger = getLogger(__name__)


class AzureMLWorkspace:
    """Class to encapsulate Azure ML example functionality."""

    workspace: Workspace

    def __init__(self):
        """Initialize the AzureMLExample class."""
        load_dotenv()
        self.workspace = self.connect_workspace()

    def connect_workspace(self) -> Workspace:
        """Connect to Azure ML workspace."""
        try:
            return Workspace.from_config()
        except UserErrorException as e:
            logger.error("Failed to connect to Azure ML workspace: %s", e)
            raise e


class AzureMLDataset(AzureMLWorkspace):
    """
    Class to encapsulate Azure ML dataset functionality.

    args:
        dataset_name: Name of the dataset in Azure ML workspace
        source_data: pandas DataFrame to register as dataset (optional)
        kaggle_dataset: Kaggle dataset name to download if dataset doesn't exist (optional)
                        example: whenamancodes/student-performance
        file_name: Pandas readable local file to register as dataset (optional)
                   example: student-mat.csv
        description: Description for the dataset (optional)
    """

    dataset: Union[FileDataset, TabularDataset]

    def __init__(
        self,
        dataset_name: str,
        *args,
        source_data: Optional[pd.DataFrame] = None,
        kaggle_dataset: Optional[str] = None,
        file_name: Optional[str] = None,
        description: Optional[str] = "Dataset created from DataFrame or Kaggle dataset",
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.source_data = source_data
        self.kaggle_dataset = kaggle_dataset
        self.file_name = file_name
        self.description = description
        self.dataset = self.get_or_create()

    def get_or_create(self) -> Union[FileDataset, TabularDataset]:
        """
        Get or create an Azure ML dataset from a pandas DataFrame or Kaggle source.

        Args:
            dataset_name: Name of the dataset in Azure ML workspace
            source_data: pandas DataFrame to register as dataset (optional)
            kaggle_dataset: Kaggle dataset name to download if dataset doesn't exist (optional)
            file_name: Specific file from Kaggle dataset (optional)
            description: Description for the dataset (optional)

        Returns:
            Dataset: The Azure ML dataset

        Raises:
            ValueError: If neither source_data nor kaggle_dataset is provided
            Exception: If dataset creation fails
        """
        try:
            dataset = Dataset.get_by_name(workspace=self.workspace, name=self.dataset_name)
            logger.info("Found existing dataset: %s", self.dataset_name)
            return dataset
        except UserErrorException:
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
            logger.info("Downloaded Kaggle dataset: %s", self.kaggle_dataset)
        elif self.file_name:
            df = self.from_file(self.file_name)
            logger.info("Loaded dataset from file: %s", self.file_name)
        else:
            raise ValueError("Either source_data or kaggle_dataset or file_name must be provided")

        # Get default datastore
        datastore = self.workspace.get_default_datastore()
        if not isinstance(datastore, AzureBlobDatastore):
            raise TypeError("Default datastore is not of type AzureBlobDatastore")

        # Save DataFrame to temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
            logger.info("Saved DataFrame to temporary file: %s", temp_file_path)

        try:
            logger.info("Uploading temporary file to datastore: %s", temp_file_path)
            target_path = f"datasets/{self.dataset_name}/"

            # Create a temporary directory with only our CSV file
            with tempfile.TemporaryDirectory() as upload_dir:
                # Copy our CSV file to the clean upload directory
                csv_filename = f"{self.dataset_name}.csv"
                upload_file_path = os.path.join(upload_dir, csv_filename)

                # Copy the CSV content to the new file
                shutil.copy2(temp_file_path, upload_file_path)

                # Now upload only this clean directory
                file_dataset = FileDatasetFactory.upload_directory(
                    src_dir=upload_dir, target=(datastore, target_path), overwrite=True, show_progress=True
                )

                datastore_path = f"{target_path}{csv_filename}"
                logger.info("Creating dataset from uploaded file path: %s", datastore_path)

            # Add validate=False to skip validation that's causing the stream error
            dataset = Dataset.Tabular.from_delimited_files(path=(datastore, datastore_path), validate=False)

            # Register the dataset
            dataset = dataset.register(
                workspace=self.workspace,
                name=self.dataset_name,
                description=self.description
                or f"Dataset created from {'DataFrame' if self.source_data is not None else self.kaggle_dataset}",
                create_new_version=True,
            )

            logger.info("Successfully created and registered dataset: %s", self.dataset_name)
            return dataset

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    def dataset_to_dataframe(self) -> pd.DataFrame:
        """
        Convert an Azure ML dataset to pandas DataFrame.

        Args:
            dataset_name: Name of the dataset in Azure ML workspace

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame
        """
        try:
            dataset = Dataset.get_by_name(workspace=self.workspace, name=self.dataset_name)
            return dataset.to_pandas_dataframe()
        except UserErrorException as e:
            logger.error("Failed to convert dataset %s to DataFrame: %s", self.dataset_name, e)
            raise e

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

    def from_kaggle(self) -> pd.DataFrame:
        """
        Download and load a Kaggle dataset as a pandas DataFrame.

        Args:
            dataset_name: Kaggle dataset name (e.g., 'titanic', 'house-prices-advanced-regression-techniques')
            file_name: Specific file to load from the dataset (optional)

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame

        Raises:
            Exception: If dataset download or loading fails
        """
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
        Delete the Azure ML dataset.

        Raises:
            Exception: If dataset deletion fails
        """
        try:
            dataset: TabularDataset = Dataset.get_by_name(workspace=self.workspace, name=self.dataset_name)
            dataset.unregister_all_versions()
            logger.info("Successfully deleted dataset: %s", self.dataset_name)
        except UserErrorException as e:
            logger.warning("Dis not find dataset %s: %s", self.dataset_name, e)


class AzureMLCluster(AzureMLWorkspace):
    """Class to encapsulate Azure ML example functionality."""

    cluster: ComputeTarget

    def __init__(
        self,
        cluster_name: str,
        *args,
        vm_size: str = "STANDARD_DS3_V2",
        min_nodes: int = 0,
        max_nodes: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cluster_name = cluster_name
        self.vm_size = vm_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.cluster = self.get_or_create()

    def get_or_create(self) -> ComputeTarget:
        """
        Get or create an Azure ML compute cluster by name.

        Args:
            cluster_name: Name of the compute cluster
            vm_size: VM size for the cluster nodes
            min_nodes: Minimum number of nodes (0 for auto-scaling)
            max_nodes: Maximum number of nodes

        Returns:
            ComputeTarget: The compute cluster
        """
        try:
            cluster = self.workspace.compute_targets[self.cluster_name]
            logger.info("Found existing cluster: %s", self.cluster_name)
            return cluster

        except ComputeTargetException:
            logger.info("Creating new cluster: %s", self.cluster_name)

            compute_config = AmlCompute.provisioning_configuration(
                vm_size=self.vm_size,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                idle_seconds_before_scaledown=300,
            )

            cluster = ComputeTarget.create(
                workspace=self.workspace, name=self.cluster_name, provisioning_configuration=compute_config
            )

            # Wait for cluster creation to complete
            cluster.wait_for_completion(show_output=True)
            logger.info("Successfully created cluster: %s", self.cluster_name)

            return cluster

        except Exception as e:
            logger.error("Failed to get or create cluster %s: %s", self.cluster_name, e)
            raise e


class AzureAutoML(AzureMLWorkspace):
    """Class to encapsulate Azure AutoML functionality."""

    def __init__(self, dataset_name: str, cluster_name: str, *args, **kwargs):
        """Initialize the AzureAutoML class."""
        super().__init__()
        self.dataset: Union[FileDataset, TabularDataset] = AzureMLDataset(
            dataset_name=dataset_name, *args, **kwargs
        ).dataset
        self.cluster: ComputeTarget = AzureMLCluster(cluster_name=cluster_name, *args, **kwargs).cluster

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
    ) -> Run:
        """
        Run an AutoML experiment.

        Args:
            experiment_name: Name of the experiment
            dataset_name: Name of the dataset to use
            target_column: Name of the target column for prediction
            task: ML task type ('classification', 'regression', 'forecasting')
            compute_target_name: Name of compute target to use
            primary_metric: Primary metric to optimize
            max_concurrent_iterations: Maximum concurrent AutoML iterations
            max_cores_per_iteration: Cores per iteration (-1 for all available)
            training_data: Optional DataFrame to create dataset from
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            enable_early_stopping: Whether to enable early stopping
            iteration_timeout_minutes: Timeout per iteration in minutes
            experiment_timeout_hours: Total experiment timeout in hours

        Returns:
            Run: The AutoML run object
        """
        try:

            # Create experiment
            experiment = Experiment(workspace=self.workspace, name=experiment_name)
            logger.info("Created experiment: %s", experiment_name)

            # Configure AutoML settings
            automl_config = AutoMLConfig(
                task=task,
                primary_metric=primary_metric,
                training_data=self.dataset,
                label_column_name=target_column,
                compute_target=self.cluster,
                enable_early_stopping=enable_early_stopping,
                featurization="auto",
                debug_log="automl_errors.log",
                verbosity=logger.level,
                n_cross_validations=cv_folds,
                test_size=test_size,
                max_concurrent_iterations=max_concurrent_iterations,
                max_cores_per_iteration=max_cores_per_iteration,
                iteration_timeout_minutes=iteration_timeout_minutes,
                experiment_timeout_hours=experiment_timeout_hours,
            )

            # Submit the experiment
            logger.info("Submitting AutoML experiment: %s", experiment_name)
            run = experiment.submit(automl_config, show_output=True)

            logger.info("AutoML experiment submitted successfully. Run ID: %s", run.id)
            return run

        except Exception as e:
            logger.error("Failed to run AutoML experiment %s: %s", experiment_name, e)
            raise e

    def get_best_model(self, run: Run):
        """
        Get the best model from an AutoML run.

        Args:
            run: The completed AutoML run

        Returns:
            tuple: (best_run, fitted_model)
        """
        try:
            # Wait for run completion
            run.wait_for_completion(show_output=True)

            # Get the best run and model
            best_run, fitted_model = run.get()

            logger.info("Best run ID: %s", best_run.id)
            logger.info("Best model algorithm: %s", best_run.properties.get("algorithm", "Unknown"))

            return best_run, fitted_model

        except Exception as e:
            logger.error("Failed to get best model from run %s: %s", run.id, e)
            raise e

    def register_best_model(
        self, run: Run, model_name: str, description: Optional[str] = None, tags: Optional[dict] = None
    ):
        """
        Register the best model from an AutoML run.

        Args:
            run: The completed AutoML run
            model_name: Name to register the model with
            description: Description for the model
            tags: Tags to associate with the model

        Returns:
            Model: The registered model
        """
        try:
            best_run, fitted_model = self.get_best_model(run)

            # Register the model
            model = best_run.register_model(
                model_name=model_name, description=description or "Best model from AutoML experiment", tags=tags or {}
            )

            logger.info("Successfully registered model: %s", model_name)
            return model

        except Exception as e:
            logger.error("Failed to register model %s: %s", model_name, e)
            raise e

    def show_run_details(self, run: Run):
        """
        Display run details in Jupyter notebook.

        Args:
            run: The AutoML run to display
        """
        try:
            return RunDetails(run)
        except Exception as e:
            logger.error("Failed to show run details: %s", e)
            raise e
