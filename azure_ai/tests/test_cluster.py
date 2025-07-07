"""
Test script to verify Azure ML installation is working properly.
"""

import os
from logging import getLogger

from azure_ai.ml_studio import AzureAIMLStudioComputeCluster

from .test_base import AzureMLTestBase


HERE = os.path.abspath(os.path.dirname(__file__))
logger = getLogger(__name__)


class AzureMLTestCluster(AzureMLTestBase):
    """Base class for all unit tests."""

    def test_existing_cluster(self):
        """
        Test that we can create and use an Azure ML workspace.
        This will get a cluster named "tiny-cluster" if it exists,
        or create it if it does not.
        """
        if not self.is_testable:
            logger.warning("Skipping test_existing_cluster() as the environment is not testable.")
            return

        cluster = AzureAIMLStudioComputeCluster(cluster_name="tiny-cluster")
        self.assertIsInstance(cluster, AzureAIMLStudioComputeCluster)
