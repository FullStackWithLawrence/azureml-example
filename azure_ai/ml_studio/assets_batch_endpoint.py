"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Azure AI ML Studio Assets -> Endpoints: batch endpoints that serve a registered model.
"""

from logging import getLogger

from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.entities import BatchEndpoint, BatchRetrySettings, ModelBatchDeployment
from azure.core.exceptions import ResourceNotFoundError

from .exceptions import is_not_found
from .ml_studio_compute_cluster import AzureAIMLStudioComputeCluster
from .workspace import AzureAIMLWorkspace

logger = getLogger(__name__)


class AzureAIMLStudioAssetsBatchEndpoint(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Assets -> Endpoint class.

    Class to encapsulate Azure ML assets batch endpoint functionality.
    """

    endpoint: BatchEndpoint | None = None

    def __init__(
        self,
        endpoint_name: str,
        model_name: str,
        *args,
        compute_target_name: str | None = None,
        deployment_name: str | None = None,
        instance_count: int = 1,
        max_concurrency_per_instance: int = 1,
        mini_batch_size: int = 10,
        retry_settings_max_retries: int = 3,
        retry_settings_timeout: int = 300,
        description: str | None = "Batch endpoint for model inference",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.compute_target_name = compute_target_name or AzureAIMLStudioComputeCluster.get_default_name(self.ml_client)
        self.deployment_name = deployment_name or f"{model_name}-deployment"
        self.instance_count = instance_count
        self.max_concurrency_per_instance = max_concurrency_per_instance
        self.mini_batch_size = mini_batch_size
        self.retry_settings_max_retries = retry_settings_max_retries
        self.retry_settings_timeout = retry_settings_timeout
        self.description = description

        self.endpoint = self.get_or_create()

    @classmethod
    def get_default_name(cls, ml_client: MLClient) -> str | None:
        """
        Get the name of the first batch endpoint in the workspace.

        Args:
            ml_client: Azure ML workspace client

        Returns:
            str: Name of the first batch endpoint

        Raises:
            ValueError: If no batch endpoints are found in the workspace
        """
        try:
            batch_endpoints = list(ml_client.batch_endpoints.list())

            if not batch_endpoints:
                raise ValueError("No batch endpoints found in the workspace")

            first_endpoint_name = batch_endpoints[0].name
            logger.info("Found first batch endpoint: %s", first_endpoint_name)
            return first_endpoint_name

        except Exception as e:
            logger.error("Failed to get first batch endpoint name: %s", e)
            raise e

    def get_or_create(self) -> BatchEndpoint:
        """
        Get or create an Azure ML batch endpoint by name.

        Returns:
            BatchEndpoint: The batch endpoint

        Raises:
            Exception: If endpoint creation fails
        """
        if self.endpoint is not None:
            logger.info("Using existing batch endpoint: %s", self.endpoint_name)
            return self.endpoint

        if not self.endpoint_name:
            raise ValueError("Endpoint name must be provided")

        try:
            logger.info("Looking for existing batch endpoint: %s", self.endpoint_name)
            endpoint = self.ml_client.batch_endpoints.get(name=self.endpoint_name)
            logger.info("Found existing batch endpoint: %s", self.endpoint_name)
            return endpoint
        # pylint: disable=W0718
        except Exception as e:
            if not is_not_found(e):
                logger.error("Failed to get batch endpoint %s: %s", self.endpoint_name, e)
                raise e
            logger.info("Batch endpoint not found, creating new batch endpoint: %s", self.endpoint_name)

        if not self.description:
            self.description = f"Batch endpoint for {self.model_name} model"

        try:
            logger.info("Creating batch endpoint: %s", self.endpoint_name)
            endpoint = BatchEndpoint(
                name=self.endpoint_name,
                description=self.description,
            )
            endpoint_poller = self.ml_client.batch_endpoints.begin_create_or_update(endpoint)
            endpoint = endpoint_poller.result()
            logger.info("Successfully created batch endpoint: %s", self.endpoint_name)

            # Create batch deployment
            self._create_batch_deployment(endpoint)

            return endpoint

        except Exception as e:
            logger.error("Failed to create batch endpoint %s: %s", self.endpoint_name, e)
            raise e

    def _create_batch_deployment(self, endpoint: BatchEndpoint):
        """
        Create a batch deployment for the endpoint.

        Args:
            endpoint: The batch endpoint to deploy to
        """
        try:
            logger.info("Looking for model: %s", self.model_name)
            model = self.ml_client.models.get(name=self.model_name, label="latest")
            logger.info("Retrieved model: %s", self.model_name)

            logger.info("Verifying compute target: %s", self.compute_target_name)
            AzureAIMLStudioComputeCluster(
                cluster_name=self.compute_target_name, vm_size="STANDARD_DS3_V2", min_nodes=0, max_nodes=4
            ).get_or_create()

            # Create batch deployment
            logger.info("Creating batch deployment: %s", self.deployment_name)
            model_batch_deployment = ModelBatchDeployment(
                name=self.deployment_name,
                description=f"Batch deployment for {self.model_name}",
                endpoint_name=self.endpoint_name,
                model=model,
                compute=self.compute_target_name,
                instance_count=self.instance_count,
                max_concurrency_per_instance=self.max_concurrency_per_instance,
                mini_batch_size=self.mini_batch_size,
                retry_settings=BatchRetrySettings(
                    max_retries=self.retry_settings_max_retries, timeout=self.retry_settings_timeout
                ),
                environment_variables={"AZUREML_MODEL_DIR": "$AZUREML_MODEL_DIR"},
            )
            logger.info("Created model batch deployment object: %s", self.deployment_name)

            try:
                logger.info("Creating batch deployment: %s", self.deployment_name)
                model_batch_deployment = self.ml_client.batch_deployments.begin_create_or_update(
                    model_batch_deployment
                ).result()
                logger.info("Successfully created batch deployment: %s", self.deployment_name)
            except ResourceNotFoundError as e:
                logger.error("Failed to create batch deployment %s: %s", self.deployment_name, e)
                raise e

            # Set as default deployment
            logger.info("Setting %s as default deployment for endpoint %s", self.deployment_name, self.endpoint_name)
            endpoint.defaults = {"deployment_name": self.deployment_name}
            self.ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
            logger.info("Set %s as default deployment for endpoint %s", self.deployment_name, self.endpoint_name)

        except Exception as e:
            logger.error("Failed to create batch deployment %s: %s", self.deployment_name, e)
            raise e

    def invoke_batch_job(self, input_data_path: str, output_data_path: str | None = None):
        """
        Invoke a batch inference job on the endpoint.

        Args:
            input_data_path: Path to input data for batch inference
            output_data_path: Path for output data (optional)

        Returns:
            BatchJob: The batch job object
        """
        try:
            logger.info("Invoking batch job on endpoint: %s", self.endpoint_name)

            inputs = {"input_data": Input(type="uri_folder", path=input_data_path)}

            # Create outputs configuration if provided
            outputs = None
            if output_data_path:
                outputs = {"output_data": Output(type="uri_file", path=output_data_path)}

            # Submit the batch job using the invoke method
            job = self.ml_client.batch_endpoints.invoke(
                endpoint_name=self.endpoint_name, deployment_name=self.deployment_name, inputs=inputs, outputs=outputs
            )

            logger.info("Successfully invoked batch job: %s", job.name)
            return job

        except Exception as e:
            logger.error("Failed to invoke batch job on endpoint %s: %s", self.endpoint_name, e)
            raise e

    def delete(self):
        """
        Delete the Azure ML batch endpoint.

        Raises:
            Exception: If endpoint deletion fails
        """
        try:
            self.ml_client.batch_endpoints.begin_delete(name=self.endpoint_name).result()
            logger.info("Successfully deleted batch endpoint: %s", self.endpoint_name)
        # pylint: disable=W0718
        except Exception as e:
            logger.warning("Could not delete batch endpoint %s: %s", self.endpoint_name, e)
