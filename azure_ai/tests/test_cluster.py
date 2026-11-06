"""Test script to verify Azure ML installation is working properly."""

import os
from logging import getLogger
from types import SimpleNamespace
from unittest.mock import MagicMock

from azure.ai.ml.entities import AmlCompute, ComputeInstance
from azure.core.exceptions import ResourceNotFoundError

from azure_ai.ml_studio import AzureAIMLStudioComputeCluster

from .test_base import AzureMLTestBase

HERE = os.path.abspath(os.path.dirname(__file__))
logger = getLogger(__name__)


class AzureMLTestCluster(AzureMLTestBase):
    """Tests of AzureAIMLStudioComputeCluster."""

    def test_existing_cluster_offline(self):
        """Test that an existing cluster is fetched by name and not re-created."""
        ml_client = self.mock_azure()
        ml_client.compute.get.return_value = AmlCompute(name="tiny-cluster")

        cluster = AzureAIMLStudioComputeCluster(cluster_name="tiny-cluster")

        ml_client.compute.get.assert_called_once_with("tiny-cluster")
        ml_client.compute.begin_create_or_update.assert_not_called()
        self.assertIs(cluster.cluster, ml_client.compute.get.return_value)

    def test_create_cluster_offline(self):
        """Test that a missing cluster is created as an AmlCompute with the requested size and scaling."""
        ml_client = self.mock_azure()
        ml_client.compute.get.side_effect = ResourceNotFoundError("not found")
        ml_client.compute.begin_create_or_update.return_value.result.return_value = AmlCompute(name="tiny-cluster")

        cluster = AzureAIMLStudioComputeCluster(
            cluster_name="tiny-cluster", vm_size="STANDARD_DS2_V2", min_nodes=0, max_nodes=2
        )

        compute_config = ml_client.compute.begin_create_or_update.call_args.args[0]
        self.assertEqual(compute_config.name, "tiny-cluster")
        self.assertEqual(compute_config.size, "STANDARD_DS2_V2")
        self.assertEqual(compute_config.min_instances, 0)
        self.assertEqual(compute_config.max_instances, 2)
        self.assertEqual(compute_config.idle_time_before_scale_down, 300)
        self.assertIs(cluster.cluster, ml_client.compute.begin_create_or_update.return_value.result.return_value)

    def test_compute_instance_is_not_a_cluster_offline(self):
        """Test that a compute instance (a single VM) with the requested name raises TypeError, not used as a cluster."""
        ml_client = self.mock_azure()
        ml_client.compute.get.return_value = ComputeInstance(name="tiny-cluster")

        with self.assertRaises(TypeError):
            AzureAIMLStudioComputeCluster(cluster_name="tiny-cluster")
        ml_client.compute.begin_create_or_update.assert_not_called()

    def test_get_default_name_offline(self):
        """Test that get_default_name() skips compute that isn't a cluster, and raises when there is no cluster."""
        ml_client = MagicMock()
        ml_client.compute.list.return_value = [
            SimpleNamespace(name="my-notebook-vm", type="computeinstance"),
            SimpleNamespace(name="tiny-cluster", type="amlcompute"),
        ]
        self.assertEqual(AzureAIMLStudioComputeCluster.get_default_name(ml_client), "tiny-cluster")

        ml_client.compute.list.return_value = [SimpleNamespace(name="my-notebook-vm", type="computeinstance")]
        with self.assertRaises(ValueError):
            AzureAIMLStudioComputeCluster.get_default_name(ml_client)

    def test_existing_cluster(self):
        """
        Test that we can get or create a compute cluster.

        This will get a cluster named "tiny-cluster" if it exists,
        or create it if it does not.
        """
        if not self.is_testable:
            logger.warning("Skipping test_existing_cluster() as the environment is not testable.")
            return

        cluster = AzureAIMLStudioComputeCluster(cluster_name="tiny-cluster")
        self.assertIsInstance(cluster, AzureAIMLStudioComputeCluster)
