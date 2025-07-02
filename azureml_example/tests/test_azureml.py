"""
Test script to verify Azure ML installation is working properly.
"""

import os
import unittest
from logging import getLogger

from dotenv import load_dotenv


load_dotenv()
logger = getLogger(__name__)


class SmarterTestBase(unittest.TestCase):
    """Base class for all unit tests."""

    is_github_actions: bool = bool(os.getenv("GITHUB_ACTIONS", "false").lower() == "true")

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        logger.info("Setting up class-level resources...")

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources."""
        logger.info("Tearing down class-level resources...")

    def setUp(self):
        """Set up test case resources."""
        logger.info("Setting up test case: %s", self.id())
        logger.info("GITHUB_ACTIONS: %s", self.is_github_actions)

    # pylint: disable=C0415,W0611,W0401,W0718
    def test_azureml_imports(self):
        """Test that we can import core Azure ML modules."""
        try:
            import azureml.core

            print(f"✅ Azure ML Core version: {azureml.core.VERSION}")

            from azureml.core import Environment, Experiment, Workspace

            print("✅ Azure ML Core classes imported successfully")

            from azureml.train.dnn import TensorFlow
            from azureml.train.sklearn import SKLearn

            print("✅ Azure ML Training estimators imported successfully")

            try:
                from azureml.pipeline.core import Pipeline, PipelineData

                print("✅ Azure ML Pipeline components imported successfully")
            except ImportError:
                print("⚠️  Some pipeline components not available (dependency conflicts)")
                logger.warning(
                    "Pipeline and PipelineData are not available due to dependency conflicts. "
                    "Please check your environment setup. "
                    "This will not prevent running experiments, but some pipeline features may be limited."
                )

            try:
                from azureml.pipeline.steps import PythonScriptStep

                print("✅ Azure ML Pipeline components imported successfully")
            except ImportError:
                print("⚠️  Some pipeline components not available (dependency conflicts)")
                logger.warning(
                    "PythonScriptStep is not available due to dependency conflicts. "
                    "Please check your environment setup. "
                    "This will not prevent running experiments, but some pipeline features may be limited."
                )

            # Skip AutoML components as they have dependency issues
            print("⚠️  AutoML components skipped (has dependency conflicts)")

        except ImportError as e:
            print(f"❌ Import error: {e}")
            self.fail(
                "Azure ML SDK is not installed or the Python import failed. "
                "Please ensure you have installed the Azure ML SDK with `pip install azureml-sdk`. "
                "Also ensure that your Python environment is set up correctly with all dependencies installed "
                "and that it is activated before running this test."
            )
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.fail(f"Unexpected error during imports: {e}")

    def test_workspace_connection(self):
        """Test workspace connection using config.json."""
        if self.is_github_actions:
            print("⚠️  Skipping workspace connection test in GitHub Actions environment")
            return

        try:
            from azureml.core import Workspace

            # Check if config.json exists
            config_paths = [
                "./config.json",  # Project root
                "../config.json",  # Parent directory (if running from tests/)
                "../../config.json",  # Two levels up
                os.path.expanduser("~/.azureml/config.json"),  # User's home directory
            ]

            config_found = False
            for config_path in config_paths:
                if os.path.exists(config_path):
                    print(f"✅ Found config.json at: {config_path}")
                    config_found = True
                    break

            if not config_found:
                print("⚠️  No config.json found. Please place your Azure ML config.json in:")
                print("   - Project root: ./config.json")
                print("   - User directory: ~/.azureml/config.json")
                self.fail("config.json not found in expected locations")

            # Try to connect to workspace
            print("🔄 Attempting to connect to Azure ML workspace...")
            ws = Workspace.from_config()
            print(f"✅ Successfully connected to workspace: {ws.name}")
            print(f"   Subscription: {ws.subscription_id}")
            print(f"   Resource Group: {ws.resource_group}")
            print(f"   Location: {ws.location}")

        except FileNotFoundError:
            print("❌ config.json not found. Please download it from Azure ML Studio:")
            print("   1. Go to Azure ML Studio (https://ml.azure.com)")
            print("   2. Select your workspace")
            print("   3. Click the download icon next to workspace name")
            print("   4. Save config.json to your project root")
            self.fail("config.json not found in expected locations")

        except ImportError as e:
            print(f"❌ Azure ML import error: {e}")
            self.fail("Azure ML SDK not installed or import failed")
        except Exception as e:
            print(f"❌ Workspace connection error: {e}")
            print("   This could be due to:")
            print("   - Invalid config.json file")
            print("   - Network connectivity issues")
            print("   - Azure authentication problems")
            print("   - Insufficient permissions")
            self.fail("Workspace connection failed due to unexpected error")
