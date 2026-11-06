"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Azure AI ML Studio Assets -> Data: data assets from a DataFrame, a local file or a Kaggle dataset.
"""

import glob
import io
import os
import tempfile
from logging import getLogger

import kaggle
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    AccountKeyConfiguration,
    AzureBlobDatastore,
    Data,
    SasTokenConfiguration,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

from .exceptions import is_not_found
from .utils import parse_datastore_path, write_mltable_folder
from .workspace import AzureAIMLWorkspace

logger = getLogger(__name__)


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
        # pylint: disable=W0718
        except Exception as e:
            if not is_not_found(e):
                logger.error("Failed to get or create dataset %s: %s", self.dataset_name, e)
                raise e
            logger.info("Dataset not found, creating new dataset: %s", self.dataset_name)

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
            write_mltable_folder(df, upload_dir, csv_filename)
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

        datastore_path = parse_datastore_path(dataset_path)
        if datastore_path is None:
            raise ValueError(f"Dataset {self.dataset_name} is not stored in a datastore: {dataset_path}")
        datastore_name, folder = datastore_path

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

                # Use the first CSV file, while the download folder still exists
                csv_file = csv_files[0]
                logger.info("Found CSV file: %s", csv_file)
                return self.from_file(csv_file)

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
