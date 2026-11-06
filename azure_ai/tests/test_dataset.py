"""Test script to verify Azure ML installation is working properly."""

import os
import tempfile
from logging import getLogger
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    AccountKeyConfiguration,
    AzureBlobDatastore,
    AzureFileDatastore,
    SasTokenConfiguration,
)
from azure.ai.ml.exceptions import MlException, ValidationErrorType, ValidationException
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from azure_ai.ml_studio import AzureAIMLAssetsDataset, is_not_found

from .test_base import AzureMLTestBase

HERE = os.path.abspath(os.path.dirname(__file__))
logger = getLogger(__name__)

DATA_PATH = (
    "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
    "datastores/workspaceblobstore/paths/LocalUpload/abc123/maths/"
)
STUDIO_DATA_PATH = (
    "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
    "datastores/workspaceblobstore/paths/UI/2025-07-04/train.csv/"
)


def latest_not_found_error() -> MlException:
    """
    Build the error that data.get(label="latest") raises for a missing data asset.

    The SDK raises a ValidationException of type RESOURCE_NOT_FOUND, then
    raises an MlException while handling it.
    """
    try:
        try:
            raise ValidationException(
                message="Asset maths does not exist",
                no_personal_data_message="Asset does not exist",
                error_type=ValidationErrorType.RESOURCE_NOT_FOUND,
            )
        except ValidationException as validation_error:
            raise MlException(message=str(validation_error), no_personal_data_message="not found")
    except MlException as error:
        return error


def fake_kaggle_download(files: dict[str, str]):
    """
    Build a stand-in for kaggle.api.dataset_download_files().

    Args:
        files: file name -> text, written to the download folder

    Returns:
        A function that writes the files where Kaggle would
    """

    def download(dataset: str, path: str, unzip: bool):
        """Write the files to path, as an unzipped Kaggle download would."""
        for file_name, text in files.items():
            with open(os.path.join(path, file_name), "w", encoding="utf-8") as downloaded_file:
                downloaded_file.write(text)

    return download


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

        with patch("azure_ai.ml_studio.assets_dataset.ContainerClient") as mock_container_class:
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
        container.list_blobs.assert_called_once_with(name_starts_with="LocalUpload/abc123/maths")
        container.download_blob.assert_called_once_with("LocalUpload/abc123/maths/maths.csv")
        self.assertEqual(df.to_dict("list"), {"x": [1, 3], "y": [2, 4]})

    def test_dataset_to_dataframe_studio_asset_offline(self):
        """
        Test that dataset_to_dataframe() reads a data asset made in the studio.

        Its path is the CSV file itself, with a trailing slash: .../paths/UI/<date>/train.csv/
        """
        ml_client = self.mock_azure()
        ml_client.data.get.return_value = SimpleNamespace(path=STUDIO_DATA_PATH)
        ml_client.datastores.get.return_value = AzureBlobDatastore(
            name="workspaceblobstore",
            account_name="mystorage",
            container_name="mycontainer",
            credentials=AccountKeyConfiguration(account_key="fake-key"),
        )
        dataset = AzureAIMLAssetsDataset(dataset_name="titanic")

        with patch("azure_ai.ml_studio.assets_dataset.ContainerClient") as mock_container_class:
            container = mock_container_class.return_value
            container.list_blobs.return_value = [SimpleNamespace(name="UI/2025-07-04/train.csv")]
            container.download_blob.return_value.readall.return_value = b"x,y\n1,2\n"

            df = dataset.dataset_to_dataframe()

        container.list_blobs.assert_called_once_with(name_starts_with="UI/2025-07-04/train.csv")
        container.download_blob.assert_called_once_with("UI/2025-07-04/train.csv")
        self.assertEqual(df.to_dict("list"), {"x": [1], "y": [2]})

    def test_create_when_latest_lookup_not_found_offline(self):
        """
        Test that a missing data asset is created when the lookup by label="latest" says so its own way.

        data.get(label="latest") raises an MlException, raised while handling a
        ValidationException of type RESOURCE_NOT_FOUND, not a ResourceNotFoundError.
        """
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = latest_not_found_error()
        source_data = pd.DataFrame({"x": [1, 2, 3]})

        AzureAIMLAssetsDataset(dataset_name="maths", source_data=source_data)

        ml_client.data.create_or_update.assert_called_once()

    def test_lookup_error_is_raised_offline(self):
        """Test that a lookup error other than "not found" is raised, and nothing is created."""
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = HttpResponseError("server error")

        with self.assertRaises(HttpResponseError):
            AzureAIMLAssetsDataset(dataset_name="maths", source_data=pd.DataFrame({"x": [1]}))

        ml_client.data.create_or_update.assert_not_called()

    def test_is_not_found_offline(self):
        """Test that is_not_found() recognizes each way the SDK reports a missing asset, and nothing else."""
        self.assertTrue(is_not_found(ResourceNotFoundError("not found")))
        self.assertTrue(is_not_found(latest_not_found_error()))
        self.assertFalse(is_not_found(HttpResponseError("server error")))
        self.assertFalse(is_not_found(ValueError("bad value")))

    def test_dataset_to_dataframe_without_csv_offline(self):
        """Test that dataset_to_dataframe() raises ValueError when the data asset's folder holds no CSV file."""
        ml_client = self.mock_azure()
        ml_client.data.get.return_value = SimpleNamespace(path=DATA_PATH)
        ml_client.datastores.get.return_value = AzureBlobDatastore(
            name="workspaceblobstore", account_name="mystorage", container_name="mycontainer"
        )
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with patch("azure_ai.ml_studio.assets_dataset.ContainerClient") as mock_container_class:
            mock_container_class.return_value.list_blobs.return_value = [
                SimpleNamespace(name="LocalUpload/abc123/maths/MLTable")
            ]
            with self.assertRaises(ValueError):
                dataset.dataset_to_dataframe()

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

    def test_delete_missing_dataset_offline(self):
        """Test that delete() only logs a warning when the data asset doesn't exist."""
        ml_client = self.mock_azure()
        ml_client.data.archive.side_effect = ResourceNotFoundError("not found")
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with self.assertLogs("azure_ai.ml_studio", level="WARNING"):
            dataset.delete()

    def test_create_from_kaggle_offline(self):
        """Test that a missing data asset is created from the first CSV file in its Kaggle download."""
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = ResourceNotFoundError("not found")
        uploaded = {}

        def capture_upload(data_asset):
            """Read the CSV file while the upload folder still exists."""
            uploaded["description"] = data_asset.description
            uploaded["df"] = pd.read_csv(os.path.join(data_asset.path, "titanic.csv"))
            return data_asset

        ml_client.data.create_or_update.side_effect = capture_upload

        with patch("azure_ai.ml_studio.assets_dataset.kaggle") as mock_kaggle:
            mock_kaggle.api.dataset_download_files.side_effect = fake_kaggle_download(
                {"train.csv": "Age,Fare\n22,7.25\n"}
            )
            AzureAIMLAssetsDataset(dataset_name="titanic", kaggle_dataset="heptapod/titanic", description=None)

        download = mock_kaggle.api.dataset_download_files.call_args.kwargs
        self.assertEqual(download["dataset"], "heptapod/titanic")
        self.assertTrue(download["unzip"])
        self.assertEqual(uploaded["description"], "Dataset created from heptapod/titanic")
        self.assertEqual(uploaded["df"].to_dict("list"), {"Age": [22], "Fare": [7.25]})

    def test_from_kaggle_without_csv_offline(self):
        """Test that a Kaggle download with no CSV file raises ValueError."""
        self.mock_azure()
        dataset = AzureAIMLAssetsDataset(dataset_name="titanic")

        with patch("azure_ai.ml_studio.assets_dataset.kaggle") as mock_kaggle:
            mock_kaggle.api.dataset_download_files.side_effect = fake_kaggle_download({"README.md": "no data here"})
            with self.assertRaises(ValueError):
                dataset.from_kaggle("someone/no-csv")

    def test_create_from_file_offline(self):
        """Test that a missing data asset is created from a local file, described by its path."""
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = ResourceNotFoundError("not found")
        file_name = os.path.join(HERE, "data", "maths.csv")

        AzureAIMLAssetsDataset(dataset_name="maths", file_name=file_name, description=None)

        data_asset = ml_client.data.create_or_update.call_args.args[0]
        self.assertEqual(data_asset.description, f"Dataset created from {file_name}")

    def test_create_from_dataframe_default_description_offline(self):
        """Test that a data asset created from a DataFrame with no description says so."""
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = ResourceNotFoundError("not found")

        AzureAIMLAssetsDataset(dataset_name="maths", source_data=pd.DataFrame({"x": [1]}), description=None)

        data_asset = ml_client.data.create_or_update.call_args.args[0]
        self.assertEqual(data_asset.description, "Dataset created from DataFrame")

    def test_create_from_empty_dataframe_offline(self):
        """Test that an empty DataFrame raises ValueError, and nothing is uploaded."""
        ml_client = self.mock_azure()
        ml_client.data.get.side_effect = ResourceNotFoundError("not found")

        with self.assertRaises(ValueError):
            AzureAIMLAssetsDataset(dataset_name="maths", source_data=pd.DataFrame())

        ml_client.data.create_or_update.assert_not_called()

    def test_from_file_unreadable_offline(self):
        """Test that from_file() raises when a file is neither CSV nor Excel, or holds no data at all."""
        self.mock_azure()
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with tempfile.TemporaryDirectory() as temp_dir:
            not_text = os.path.join(temp_dir, "binary.csv")
            with open(not_text, "wb") as binary_file:
                binary_file.write(b"\xff\xfe\x00\x81 not a spreadsheet")
            with self.assertRaises(Exception):
                dataset.from_file(not_text)

            empty = os.path.join(temp_dir, "empty.csv")
            with open(empty, "w", encoding="utf-8"):
                pass
            with self.assertRaises(pd.errors.EmptyDataError):
                dataset.from_file(empty)

    def test_datastore_container_credentials_offline(self):
        """
        Test that get_datastore_container() uses the datastore's own credential.

        A SAS token when the datastore has one, and your Azure login when it has none.
        """
        ml_client = self.mock_azure()
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with patch("azure_ai.ml_studio.assets_dataset.ContainerClient") as mock_container_class:
            ml_client.datastores.get.return_value = AzureBlobDatastore(
                name="sas-store",
                account_name="mystorage",
                container_name="mycontainer",
                credentials=SasTokenConfiguration(sas_token="fake-sas"),
            )
            dataset.get_datastore_container("sas-store")
            self.assertEqual(mock_container_class.call_args.kwargs["credential"], "fake-sas")

            ml_client.datastores.get.return_value = AzureBlobDatastore(
                name="identity-store", account_name="mystorage", container_name="mycontainer"
            )
            with patch("azure_ai.ml_studio.assets_dataset.DefaultAzureCredential") as mock_credential_class:
                dataset.get_datastore_container("identity-store")
            self.assertIs(mock_container_class.call_args.kwargs["credential"], mock_credential_class.return_value)

    def test_datastore_container_not_blob_offline(self):
        """Test that get_datastore_container() raises TypeError for a datastore that isn't Blob Storage."""
        ml_client = self.mock_azure()
        ml_client.datastores.get.return_value = AzureFileDatastore(
            name="workspacefilestore", account_name="mystorage", file_share_name="share"
        )
        dataset = AzureAIMLAssetsDataset(dataset_name="maths")

        with self.assertRaises(TypeError):
            dataset.get_datastore_container("workspacefilestore")

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
