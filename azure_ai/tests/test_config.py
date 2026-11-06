"""Test script to verify Azure ML installation is working properly."""

import os
import sys
from logging import getLogger

from dotenv import load_dotenv

from .test_base import AzureMLTestBase

load_dotenv()
logger = getLogger(__name__)


class AzureMLTestConfig(AzureMLTestBase):
    """Tests of the Python environment and the workspace configuration."""

    def test_package_version(self):
        """Test that azure_ai has a semantic version, which CI writes to azure_ai/__version__.py."""
        from azure_ai.__version__ import __version__  # pylint: disable=C0415

        self.assertRegex(__version__, r"^\d+\.\d+\.\d+")

    def test_python_version(self):
        """Test that the tests run on Python 3.14 or later, the version this repository targets."""
        self.assertGreaterEqual(sys.version_info[:2], (3, 14))

    # pylint: disable=C0415,W0611,R0914
    def test_azureml_imports(self):
        """Test that we can import the Azure ML SDK v2 modules that azure_ai.ml_studio uses."""
        try:
            import azure.ai.ml

            print(f"✅ Azure ML SDK v2 (azure-ai-ml) version: {azure.ai.ml.__version__}")

            from azure.ai.ml import Input, MLClient, Output, automl
            from azure.ai.ml.entities import (
                AmlCompute,
                BatchEndpoint,
                Data,
                Model,
                ModelBatchDeployment,
            )
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import ContainerClient

            print("✅ Azure ML SDK v2 classes imported successfully")

        except ImportError as e:
            print(f"❌ Import error: {e}")
            self.fail(
                "Azure ML SDK v2 is not installed or the Python import failed. "
                "Please ensure you have installed the requirements with `pip install -r requirements/local.txt`, "
                "and that your Python environment is activated before running this test."
            )

    def test_workspace_connection(self):
        """Test workspace connection using config.json."""

        if not self.is_testable:
            logger.warning("Skipping test_workspace_connection() as the environment is not testable.")
            return

        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        # Check if config.json exists
        config_paths = [
            "./config.json",  # Project root
            "../config.json",  # Parent directory (if running from tests/)
            "../../config.json",  # Two levels up
            "./.azureml/config.json",  # MLClient.from_config() also looks here
        ]

        config_found = False
        for config_path in config_paths:
            if os.path.exists(config_path):
                print(f"✅ Found config.json at: {config_path}")
                config_found = True
                break

        if not config_found:
            print("⚠️  No config.json found. Please place your Azure ML config.json in the project root.")
            print("   1. Go to Azure ML Studio (https://ml.azure.com)")
            print("   2. Select your workspace")
            print("   3. Click the download icon next to workspace name")
            print("   4. Save config.json to your project root")
            self.fail("config.json not found in expected locations")

        try:
            print("🔄 Attempting to connect to Azure ML workspace...")
            ml_client = MLClient.from_config(credential=DefaultAzureCredential())
            ws = ml_client.workspaces.get(ml_client.workspace_name)
            if ws is None:
                self.fail("The workspace in config.json was not found")
            print(f"✅ Successfully connected to workspace: {ws.name}")
            print(f"   Location: {ws.location}")
        # pylint: disable=W0718
        except Exception as e:
            print(f"❌ Workspace connection error: {e}")
            print("   This could be due to:")
            print("   - Invalid config.json file")
            print("   - Network connectivity issues")
            print("   - Azure authentication problems (run `az login`)")
            print("   - Insufficient permissions")
            self.fail("Workspace connection failed due to unexpected error")
