"""
Test script to verify Azure ML installation is working properly.
"""

import os
import unittest
from logging import getLogger
from unittest.mock import patch

from dotenv import load_dotenv


load_dotenv()
logger = getLogger(__name__)


class AzureMLTestBase(unittest.TestCase):
    """Base class for all unit tests."""

    is_github_actions: bool = bool(os.getenv("GITHUB_ACTIONS", "false").lower() == "true")
    is_testable: bool = not is_github_actions

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        if not cls.is_testable:
            logger.warning("CI - skipping tests that require Azure ML workspace connection")
            cls._setup_azure_mocks()
            return

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources."""
        if not cls.is_testable:
            # Stop all patches
            patchers = [
                "ml_client_patcher",
                "credential_patcher",
                "dataset_v2_patcher",
                "workspace_patcher",
                "dataset_v1_patcher",
                "compute_patcher",
                "experiment_patcher",
            ]
            for patcher_name in patchers:
                if hasattr(cls, patcher_name):
                    getattr(cls, patcher_name).stop()

    @classmethod
    def _setup_azure_mocks(cls):
        """Set up mocks for Azure ML clients when not testable."""
        logger.info("Setting up Azure ML mocks for non-testable CI-CD environment")

        # Patch at the source before any imports happen
        cls.ml_client_patcher = patch("azure.ai.ml.MLClient")
        cls.mock_ml_client = cls.ml_client_patcher.start()

        cls.credential_patcher = patch("azure.identity.DefaultAzureCredential")
        cls.mock_credential = cls.credential_patcher.start()

        # Also patch other Azure classes at their source
        cls.workspace_patcher = patch("azureml.core.Workspace")
        cls.mock_workspace = cls.workspace_patcher.start()

        cls.dataset_v1_patcher = patch("azureml.core.Dataset")
        cls.mock_dataset_v1 = cls.dataset_v1_patcher.start()

        cls.compute_patcher = patch("azureml.core.compute.AmlCompute")
        cls.mock_compute = cls.compute_patcher.start()

        cls.experiment_patcher = patch("azureml.core.Experiment")
        cls.mock_experiment = cls.experiment_patcher.start()
