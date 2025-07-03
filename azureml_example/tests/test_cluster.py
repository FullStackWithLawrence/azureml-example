"""
Test script to verify Azure ML installation is working properly.
"""

import os

from azureml_example.automated_ml import AzureMLCluster

from .test_base import AzureMLTestBase


HERE = os.path.abspath(os.path.dirname(__file__))


class AzureMLTestCluster(AzureMLTestBase):
    """Base class for all unit tests."""

    def test_existing_cluster(self):
        """
        Test that we can create and use an Azure ML workspace.
        This will get a cluster named "standard-cluster" if it exists,
        or create it if it does not.
        """
        if not self.is_testable:
            return

        cluster = AzureMLCluster(cluster_name="standard-cluster")
        self.assertIsInstance(cluster, AzureMLCluster)
