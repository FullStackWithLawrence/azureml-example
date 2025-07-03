"""
Test script to verify Azure ML installation is working properly.
"""

from azure_ai.ml_studio import AzureAIMLWorkspace

from .test_base import AzureMLTestBase


class AzureMLTestWorkspace(AzureMLTestBase):
    """Base class for all unit tests."""

    def test_azureml_workspace(self):
        """Test that we can create and use an Azure ML workspace."""

        if not self.is_testable:
            return

        workspace = AzureAIMLWorkspace()
        self.assertIsInstance(workspace, AzureAIMLWorkspace)
