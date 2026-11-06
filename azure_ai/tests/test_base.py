"""Base class for the unit tests."""

import os
import unittest
from logging import getLogger
from unittest.mock import MagicMock, patch

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
            for patcher_name in ("ml_client_patcher", "credential_patcher"):
                if hasattr(cls, patcher_name):
                    getattr(cls, patcher_name).stop()

    @classmethod
    def _setup_azure_mocks(cls):
        """
        Replace MLClient and DefaultAzureCredential with mocks.

        No test can then reach Azure in a CI-CD environment. The patches target azure_ai.ml_studio, where the names are used:
        ml_studio imported them when it was loaded, so patching them in
        azure.ai.ml or azure.identity would not affect it.
        """
        logger.info("Setting up Azure ML mocks for non-testable CI-CD environment")

        cls.ml_client_patcher = patch("azure_ai.ml_studio.MLClient")
        cls.mock_ml_client = cls.ml_client_patcher.start()

        cls.credential_patcher = patch("azure_ai.ml_studio.DefaultAzureCredential")
        cls.mock_credential = cls.credential_patcher.start()

    def mock_azure(self) -> MagicMock:
        """
        Fake the workspace connection for one offline test, in any environment.

        Patches MLClient, DefaultAzureCredential and load_dotenv in
        azure_ai.ml_studio until the test ends.

        Returns:
            MagicMock: The ml_client that every helper class will get
        """
        mock_ml_client_class = self.enterContext(patch("azure_ai.ml_studio.MLClient"))
        self.enterContext(patch("azure_ai.ml_studio.DefaultAzureCredential"))
        self.enterContext(patch("azure_ai.ml_studio.load_dotenv"))
        return mock_ml_client_class.from_config.return_value
