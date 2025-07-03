"""
Test script to verify Azure ML installation is working properly.
"""

import os
import unittest
from logging import getLogger

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
            return
