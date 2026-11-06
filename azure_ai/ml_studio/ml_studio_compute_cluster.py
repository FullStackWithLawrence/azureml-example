"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Azure AI ML Studio Compute -> Clusters: get or create a compute cluster.
"""

from logging import getLogger

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Compute
from azure.core.exceptions import ResourceNotFoundError

from .workspace import AzureAIMLWorkspace

logger = getLogger(__name__)


class AzureAIMLStudioComputeCluster(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Compute -> Cluster class.

    Class to encapsulate Azure ML example functionality.
    """

    cluster: AmlCompute | None = None

    def __init__(
        self,
        *args,
        cluster_name: str | None = None,
        vm_size: str = "STANDARD_DS3_V2",
        min_nodes: int = 0,
        max_nodes: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cluster_name = cluster_name or AzureAIMLStudioComputeCluster.get_default_name(self.ml_client)
        self.vm_size = vm_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.cluster = self.get_or_create()

    @classmethod
    def get_default_name(cls, ml_client: MLClient) -> str:
        """
        Get the name of the first compute cluster in the workspace.

        Args:
            ml_client: Azure ML workspace client

        Returns:
            str: Name of the first compute cluster

        Raises:
            ValueError: If no compute clusters are found in the workspace
        """
        try:
            for compute in ml_client.compute.list():
                # Skip compute that isn't a cluster, e.g. a compute instance
                if compute.type != "amlcompute":
                    continue
                if compute.name:
                    logger.info("Found first compute cluster: %s", compute.name)
                    return compute.name

            raise ValueError("No compute clusters found in the workspace")

        except Exception as e:
            logger.error("Failed to get first cluster name: %s", e)
            raise e

    def get_or_create(self) -> AmlCompute:
        """
        Get or create an Azure ML compute cluster by name.

        Uses the cluster_name, vm_size, min_nodes and max_nodes given to the constructor.

        Returns:
            AmlCompute: The compute cluster

        Raises:
            TypeError: If the workspace's compute with this name is not a cluster, e.g. a compute instance
        """
        if self.cluster is not None:
            logger.info("Using existing cluster: %s", self.cluster_name)
            return self.cluster

        try:
            cluster = self.ml_client.compute.get(self.cluster_name)
            logger.info("Found existing cluster: %s", self.cluster_name)
        except ResourceNotFoundError:
            cluster = self.create()
        except Exception as e:
            logger.error("Failed to get cluster %s: %s", self.cluster_name, e)
            raise e

        if not isinstance(cluster, AmlCompute):
            raise TypeError(f"Compute {self.cluster_name} is a {cluster.type}, not a compute cluster")
        return cluster

    def create(self) -> Compute:
        """
        Create an Azure ML compute cluster, and wait until it's ready.

        Uses the cluster_name, vm_size, min_nodes and max_nodes given to the constructor.

        Returns:
            Compute: The new compute cluster
        """
        logger.info("Creating new cluster: %s", self.cluster_name)
        compute_config = AmlCompute(
            name=self.cluster_name,
            size=self.vm_size,
            min_instances=self.min_nodes,
            max_instances=self.max_nodes,
            idle_time_before_scale_down=300,
        )

        try:
            poller = self.ml_client.compute.begin_create_or_update(compute_config)
            cluster = poller.result()
        except Exception as e:
            logger.error("Failed to create cluster %s: %s", self.cluster_name, e)
            raise e

        logger.info("Successfully created cluster: %s", self.cluster_name)
        return cluster
