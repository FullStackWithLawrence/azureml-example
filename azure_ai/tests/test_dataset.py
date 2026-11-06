"""Test script to verify Azure ML installation is working properly."""

import os
from logging import getLogger
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AccountKeyConfiguration, AzureBlobDatastore
from azure.core.exceptions import ResourceNotFoundError

from azure_ai.ml_studio import AzureAIMLAssetsDataset

from .test_base import AzureMLTestBase

HERE = os.path.abspath(os.path.dirname(__file__))
logger = getLogger(__name__)

DATA_PATH = (
    "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
    "datastores/workspaceblobstore/paths/LocalUpload/abc123/maths/"
)


class AzureMLTestDataset(AzureMLTestBase):
    """Tests of AzureAIMLAssetsDataset."""

    def test_existing_dataset_offline(self):
        """Test that an existing data asset is fetched by name, latest version, and not re-created."""
        ml_client = self.mock_azure()

        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        ml_client.data.get.assert_called_once_with(name="maths", label="latest")
        ml_client.data.create_or_update.assert_not_called()
        self.assertIs(dataset.dataset, ml_client.data.get.return_value)

    def test_create_mltable_dataset_offline(self):
        """
        Test that a new data asset is registered as an MLTable folder.

        The folder holds the data as a CSV file plus an MLTable file that reads it.
        """
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = ResourceNotFoundError("not found")
        uploaded = {}

        def capture_upload(data_asset):
            """Read the folder while it still exists: create_or_update() is where the SDK uploads it."""
            uploaded["data_asset"] = data_asset
            uploaded["files"] = sorted(os.listdir(data_asset.path))
            with open(os.path.join(data_asset.path, "MLTable"), encoding="utf-8") as mltable_file:
                uploaded["mltable"] = mltable_file.read()
            uploaded["df"] = pd.read_csv(os.path.join(data_asset.path, "maths.csv"))
            return data_asset

        ml_client.data.create_or_update.side_effect = capture_upload
        source_data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

        AzureAIMLAssetsDataset(dataset_name="maths", source_data=source_data, description="test data")

        data_asset = uploaded["data_asset"]
        self.assertEqual(data_asset.name, "maths")
        self.assertEqual(data_asset.type, AssetTypes.MLTABLE)
        self.assertEqual(data_asset.description, "test data")
        self.assertEqual(uploaded["files"], ["MLTable", "maths.csv"])
        self.assertIn("- file: ./maths.csv", uploaded["mltable"])
        self.assertIn("read_delimited", uploaded["mltable"])
        pd.testing.assert_frame_equal(uploaded["df"], source_data)

    def test_create_without_source_offline(self):
        """Test that creating a data asset with no source data raises ValueError."""
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = ResourceNotFoundError("not found")

        with self.assertRaises(ValueError):
            AzureAIMLAssetsDataset(dataset_name="maths")

    def test_from_file_excel_fallback_offline(self):
        """Test that from_file() reads maths.csv, an Excel workbook with a .csv name, as Excel."""
        self.mock_azure()
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        df = dataset.from_file(os.path.join(HERE, "data", "maths.csv"))

        self.assertFalse(df.empty)

    def test_from_file_missing_offline(self):
        """Test that from_file() raises FileNotFoundError for a missing file."""
        self.mock_azure()
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with self.assertRaises(FileNotFoundError):
            dataset.from_file(os.path.join(HERE, "data", "no-such-file.csv"))

    def test_dataset_to_dataframe_offline(self):
        """Test that dataset_to_dataframe() downloads the asset's CSV file from its datastore with the account key."""
        ml_client = self.mock_azure()
        ml_client.data.get.return_value = SimpleNamespace(path=DATA_PATH)
        ml_client.datastores.get.return_value = AzureBlobDatastore(
            name="workspaceblobstore",
            account_name="mystorage",
            container_name="mycontainer",
            credentials=AccountKeyConfiguration(account_key="fake-key"),
        )
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with patch("azure_ai.ml_studio.ContainerClient") as mock_container_class:
            container = mock_container_class.return_value
            container.list_blobs.return_value = [
                SimpleNamespace(name="LocalUpload/abc123/maths/MLTable"),
                SimpleNamespace(name="LocalUpload/abc123/maths/maths.csv"),
            ]
            container.download_blob.return_value.readall.return_value = b"x,y\n1,2\n3,4\n"

            df = dataset.dataset_to_dataframe()

        ml_client.datastores.get.assert_called_once_with("workspaceblobstore", include_secrets=True)
        mock_container_class.assert_called_once_with(
            account_url="https://mystorage.blob.core.windows.net",
            container_name="mycontainer",
            credential="fake-key",
        )
        container.list_blobs.assert_called_once_with(name_starts_with="LocalUpload/abc123/maths/")
        container.download_blob.assert_called_once_with("LocalUpload/abc123/maths/maths.csv")
        self.assertEqual(df.to_dict("list"), {"x": [1, 3], "y": [2, 4]})

    def test_dataset_to_dataframe_not_in_datastore_offline(self):
        """Test that dataset_to_dataframe() raises ValueError for an asset outside a datastore, e.g. a web url."""
        ml_client = self.mock_azure()
        ml_client.data.get.return_value = SimpleNamespace(path="https://example.com/maths.csv")
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with self.assertRaises(ValueError):
            dataset.dataset_to_dataframe()

    def test_delete_archives_offline(self):
        """Test that delete() archives every version of the data asset, since v2 cannot delete one."""
        ml_client = self.mock_azure()
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        dataset.delete()

        ml_client.data.archive.assert_called_once_with(name="maths")

    def test_get_default_name_offline(self):
        """Test that get_default_name() returns the first data asset's name, and raises when there is none."""
        ml_client = MagicMock()
        first, second = MagicMock(), MagicMock()
        first.name, second.name = "first", "second"
        ml_client.data.list.return_value = iter([first, second])
        self.assertEqual(AzureAIMLAssetsDataset.get_default_name(ml_client), "first")

        ml_client.data.list.return_value = iter([])
        with self.assertRaises(ValueError):
            AzureAIMLAssetsDataset.get_default_name(ml_client)

    def test_existing_dataset(self):
        """Test that we can get an existing data asset from the workspace."""
        if not self.is_testable:
            logger.warning("Skipping test_existing_dataset() as the environment is not testable.")
            return

        dataset = AzureAIMLAssetsDataset(
            dataset_name="student-performance-base",
        )
        self.assertIsInstance(dataset, AzureAIMLAssetsDataset)

    def test_kaggle_dataset(self):
        """
        Test that we can create a data asset from a Kaggle dataset.

        https://www.kaggle.com/datasets/whenamancodes/student-performance
        """
        if not self.is_testable:
            logger.warning("Skipping test_kaggle_dataset() as the environment is not testable.")
            return

        dataset = AzureAIMLAssetsDataset(
            dataset_name="whenamancodes-student-performance",
            kaggle_dataset="whenamancodes/student-performance",
        )
        self.assertIsInstance(dataset, AzureAIMLAssetsDataset)
        dataset.delete()

    def test_file_dataset(self):
        """
        Test that we can create a data asset from a local file, and read it back.

        The test file is an Excel workbook incorrectly named `maths.csv` that contains
        a single sheet with some sample data. This is a common mistake with
        kaggle datasets where the file is actually an Excel file but has a `.csv` extension.
        The file is located in the `data` directory of this repository.
        """

        if not self.is_testable:
            logger.warning("Skipping test_file_dataset() as the environment is not testable.")
            return

        dataset = AzureAIMLAssetsDataset(
            dataset_name="test-maths-dataset",
            file_name=os.path.join(HERE, "data", "maths.csv"),
        )
        self.assertIsInstance(dataset, AzureAIMLAssetsDataset)
        try:
            self.assertFalse(dataset.dataset_to_dataframe().empty)
        finally:
            dataset.delete()
