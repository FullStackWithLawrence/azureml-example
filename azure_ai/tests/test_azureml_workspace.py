"""Test script to verify Azure ML installation is working properly."""

from logging import getLogger
from unittest.mock import patch

from azure.ai.ml.exceptions import ValidationException

from azure_ai.ml_studio import AzureAIMLWorkspace

from .test_base import AzureMLTestBase

logger = getLogger(__name__)


class AzureMLTestWorkspace(AzureMLTestBase):
    """Tests of AzureAIMLWorkspace."""

    def test_connect_workspace_offline(self):
        """Test that the workspace connects with MLClient.from_config() and loads the workspace entity."""
        ml_client = self.mock_azure()
        ml_client.workspace_name = "my-workspace"

        workspace = AzureAIMLWorkspace()

        self.assertIs(workspace.ml_client, ml_client)
        ml_client.workspaces.get.assert_called_once_with("my-workspace")
        self.assertIs(workspace.workspace, ml_client.workspaces.get.return_value)

    def test_connect_workspace_without_config_offline(self):
        """Test that a missing or invalid config.json raises the SDK's ValidationException."""
        self.mock_azure()
        error = ValidationException(message="config.json not found", no_personal_data_message="config.json not found")

        with patch("azure_ai.ml_studio.MLClient.from_config", side_effect=error):
            with self.assertRaises(ValidationException):
                AzureAIMLWorkspace()

    def test_azureml_workspace(self):
        """Test that we can connect to the Azure ML workspace in config.json."""

        if not self.is_testable:
            logger.warning("Skipping test_azureml_workspace() as the environment is not testable.")
            return

        workspace = AzureAIMLWorkspace()
        self.assertIsInstance(workspace, AzureAIMLWorkspace)
