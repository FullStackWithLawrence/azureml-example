"""Test script to verify Azure ML Batch Endpoint functionality."""

import hashlib
import os
from datetime import datetime
from logging import getLogger
from unittest.mock import MagicMock

from azure.ai.ml.entities import AmlCompute, BatchEndpoint, Model, ModelBatchDeployment
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from azure_ai.ml_studio import (
    AzureAIMLAssetsDataset,
    AzureAIMLStudioAssetsBatchEndpoint,
    AzureAIMLWorkspace,
)

from .test_base import AzureMLTestBase

HERE = os.path.abspath(os.path.dirname(__file__))
logger = getLogger(__name__)


class AzureMLTestBatchEndpoint(AzureMLTestBase):
    """Test class for Azure ML Batch Endpoint functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        super().setUpClass()
        hash_suffix = hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:8]
        cls.endpoint_name = "test-batch-endpoint" + "-" + hash_suffix
        cls.model_name = "best-01"  # FIX NOTE: need to create a temporary model that we can destroy after the test
        cls.compute_target_name = (
            "tiny-cluster"  # FIX NOTE: need to create a temporary compute cluster that we can destroy after the test
        )
        cls.description = "Custom test batch endpoint"

    @classmethod
    def tearDownClass(cls):
        """Tear down the test class."""
        super().tearDownClass()
        if not cls.is_testable:
            # Offline, nothing was created. And the mocks are gone by now, so
            # building an endpoint here would reach Azure, and could create one.
            return
        try:
            endpoint = AzureAIMLStudioAssetsBatchEndpoint(
                endpoint_name=cls.endpoint_name,
                model_name=cls.model_name,
            )
            endpoint.delete()
        # pylint: disable=W0718
        except Exception as e:
            logger.error("Failed to clean up batch endpoint: %s", e)

    def test_existing_batch_endpoint_offline(self):
        """Test that an existing endpoint is fetched with the workspace's own ml_client, and not re-created."""
        ml_client = self.mock_azure()

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster"
        )

        self.assertIs(batch_endpoint.ml_client, ml_client)
        ml_client.batch_endpoints.get.assert_called_once_with(name="my-endpoint")
        ml_client.batch_endpoints.begin_create_or_update.assert_not_called()
        self.assertEqual(batch_endpoint.deployment_name, "best-01-deployment")

        # Asking again returns the endpoint it already has, without another lookup
        self.assertIs(batch_endpoint.get_or_create(), batch_endpoint.endpoint)
        ml_client.batch_endpoints.get.assert_called_once()

    def test_default_compute_target_offline(self):
        """Test that the endpoint uses the workspace's first compute cluster when none is named."""
        ml_client = self.mock_azure()
        cluster = MagicMock(type="amlcompute")
        cluster.name = "tiny-cluster"
        ml_client.compute.list.return_value = [cluster]

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(endpoint_name="my-endpoint", model_name="best-01")

        self.assertEqual(batch_endpoint.compute_target_name, "tiny-cluster")

    def mock_new_endpoint(self) -> MagicMock:
        """Fake a workspace with no endpoint named my-endpoint, model best-01, and cluster tiny-cluster."""
        ml_client = self.mock_azure()
        ml_client.batch_endpoints.get.side_effect = ResourceNotFoundError("not found")
        ml_client.batch_endpoints.begin_create_or_update.return_value.result.return_value = BatchEndpoint(
            name="my-endpoint"
        )
        ml_client.models.get.return_value = Model(name="best-01", version="1")
        ml_client.compute.get.return_value = AmlCompute(name="tiny-cluster")
        return ml_client

    def test_create_batch_endpoint_offline(self):
        """
        Test that a missing endpoint is created, with a deployment of the latest model that becomes its default.

        The deployment uses the settings given to the constructor.
        """
        ml_client = self.mock_new_endpoint()

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name="my-endpoint",
            model_name="best-01",
            compute_target_name="tiny-cluster",
            instance_count=1,
            mini_batch_size=5,
            retry_settings_max_retries=2,
            retry_settings_timeout=60,
        )

        new_endpoint = ml_client.batch_endpoints.begin_create_or_update.call_args_list[0].args[0]
        self.assertEqual(new_endpoint.name, "my-endpoint")
        self.assertEqual(new_endpoint.description, "Batch endpoint for model inference")
        ml_client.models.get.assert_called_once_with(name="best-01", label="latest")

        deployment = ml_client.batch_deployments.begin_create_or_update.call_args.args[0]
        self.assertIsInstance(deployment, ModelBatchDeployment)
        self.assertEqual(deployment.name, "best-01-deployment")
        self.assertEqual(deployment.endpoint_name, "my-endpoint")
        self.assertEqual(deployment.compute, "tiny-cluster")
        self.assertEqual(deployment.resources.instance_count, 1)
        self.assertEqual(deployment.settings.mini_batch_size, 5)
        self.assertEqual(deployment.settings.retry_settings.max_retries, 2)
        self.assertEqual(deployment.settings.retry_settings.timeout, 60)

        updated_endpoint = ml_client.batch_endpoints.begin_create_or_update.call_args_list[1].args[0]
        self.assertEqual(updated_endpoint.defaults, {"deployment_name": "best-01-deployment"})
        if batch_endpoint.endpoint is None:
            self.fail("The helper class did not keep the endpoint it created")
        self.assertEqual(batch_endpoint.endpoint.name, "my-endpoint")

    def test_create_batch_endpoint_default_description_offline(self):
        """Test that an endpoint created with no description gets one that names its model."""
        ml_client = self.mock_new_endpoint()

        AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster", description=None
        )

        new_endpoint = ml_client.batch_endpoints.begin_create_or_update.call_args_list[0].args[0]
        self.assertEqual(new_endpoint.description, "Batch endpoint for best-01 model")

    def test_create_batch_deployment_failure_offline(self):
        """Test that a failed deployment is raised, and the endpoint's default deployment is never set."""
        ml_client = self.mock_new_endpoint()
        ml_client.batch_deployments.begin_create_or_update.side_effect = ResourceNotFoundError("no such cluster")

        with self.assertRaises(ResourceNotFoundError):
            AzureAIMLStudioAssetsBatchEndpoint(
                endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster"
            )

        self.assertEqual(ml_client.batch_endpoints.begin_create_or_update.call_count, 1)

    def test_lookup_error_is_raised_offline(self):
        """Test that an endpoint lookup error other than "not found" is raised, and nothing is created."""
        ml_client = self.mock_azure()
        ml_client.batch_endpoints.get.side_effect = HttpResponseError("server error")

        with self.assertRaises(HttpResponseError):
            AzureAIMLStudioAssetsBatchEndpoint(
                endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster"
            )

        ml_client.batch_endpoints.begin_create_or_update.assert_not_called()

    def test_endpoint_name_required_offline(self):
        """Test that an empty endpoint name raises ValueError before anything reaches the workspace."""
        ml_client = self.mock_azure()

        with self.assertRaises(ValueError):
            AzureAIMLStudioAssetsBatchEndpoint(
                endpoint_name="", model_name="best-01", compute_target_name="tiny-cluster"
            )

        ml_client.batch_endpoints.get.assert_not_called()

    def test_get_default_name_offline(self):
        """Test that get_default_name() returns the first endpoint's name, and raises when there is none."""
        ml_client = MagicMock()
        ml_client.batch_endpoints.list.return_value = [BatchEndpoint(name="first"), BatchEndpoint(name="second")]
        self.assertEqual(AzureAIMLStudioAssetsBatchEndpoint.get_default_name(ml_client), "first")

        ml_client.batch_endpoints.list.return_value = []
        with self.assertRaises(ValueError):
            AzureAIMLStudioAssetsBatchEndpoint.get_default_name(ml_client)

    def test_invoke_batch_job_offline(self):
        """Test that invoke_batch_job() sends the input folder, and the output file only when one is given."""
        ml_client = self.mock_azure()
        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster"
        )

        job = batch_endpoint.invoke_batch_job("azureml://datastores/workspaceblobstore/paths/input/")

        self.assertIs(job, ml_client.batch_endpoints.invoke.return_value)
        kwargs = ml_client.batch_endpoints.invoke.call_args.kwargs
        self.assertEqual(kwargs["endpoint_name"], "my-endpoint")
        self.assertEqual(kwargs["deployment_name"], "best-01-deployment")
        self.assertEqual(kwargs["inputs"]["input_data"].path, "azureml://datastores/workspaceblobstore/paths/input/")
        self.assertIsNone(kwargs["outputs"])

        batch_endpoint.invoke_batch_job(
            "input/", output_data_path="azureml://datastores/workspaceblobstore/paths/out.csv"
        )

        outputs = ml_client.batch_endpoints.invoke.call_args.kwargs["outputs"]
        self.assertEqual(outputs["output_data"].path, "azureml://datastores/workspaceblobstore/paths/out.csv")

    def test_invoke_batch_job_failure_offline(self):
        """Test that a failed invocation is raised."""
        ml_client = self.mock_azure()
        ml_client.batch_endpoints.invoke.side_effect = HttpResponseError("no deployment")
        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster"
        )

        with self.assertRaises(HttpResponseError):
            batch_endpoint.invoke_batch_job("input/")

    def test_delete_offline(self):
        """Test that delete() deletes the endpoint, and only logs a warning when that fails."""
        ml_client = self.mock_azure()
        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name="my-endpoint", model_name="best-01", compute_target_name="tiny-cluster"
        )

        batch_endpoint.delete()
        ml_client.batch_endpoints.begin_delete.assert_called_once_with(name="my-endpoint")

        ml_client.batch_endpoints.begin_delete.side_effect = ResourceNotFoundError("already gone")
        with self.assertLogs("azure_ai.ml_studio", level="WARNING"):
            batch_endpoint.delete()

    def test_create_batch_endpoint(self):
        """Test that we can create a batch endpoint with a registered model."""
        if not self.is_testable:
            logger.warning("Skipping test_create_batch_endpoint() as the environment is not testable.")
            return

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name=self.endpoint_name,
            model_name=self.model_name,
            compute_target_name=self.compute_target_name,
            description=self.description + " - test_create_batch_endpoint()",
        )
        self.assertIsInstance(batch_endpoint, AzureAIMLStudioAssetsBatchEndpoint)
        batch_endpoint.delete()

    def test_existing_batch_endpoint(self):
        """
        Test that we can retrieve an existing batch endpoint.

        The test creates the endpoint first, since the workspace may have none,
        then checks that get_default_name() finds one and that a second
        instance with the same name gets the endpoint rather than creating it.
        """
        if not self.is_testable:
            logger.warning("Skipping test_existing_batch_endpoint() as the environment is not testable.")
            return

        created_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name=self.endpoint_name,
            model_name=self.model_name,
            compute_target_name=self.compute_target_name,
            description=self.description + " - test_existing_batch_endpoint()",
        )
        try:
            self.assertIsNotNone(AzureAIMLStudioAssetsBatchEndpoint.get_default_name(AzureAIMLWorkspace().ml_client))

            existing_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
                endpoint_name=self.endpoint_name,
                model_name=self.model_name,
                compute_target_name=self.compute_target_name,
            )

            self.assertEqual(existing_endpoint.endpoint_name, self.endpoint_name)
            if existing_endpoint.endpoint is None or created_endpoint.endpoint is None:
                self.fail("The helper class did not keep the endpoint it found or created")
            self.assertEqual(existing_endpoint.endpoint.name, created_endpoint.endpoint.name)
        finally:
            created_endpoint.delete()

    def test_batch_endpoint_with_custom_settings(self):
        """Test batch endpoint creation with custom deployment settings."""
        if not self.is_testable:
            logger.warning("Skipping test_batch_endpoint_with_custom_settings() as the environment is not testable.")
            return

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name=self.endpoint_name,
            model_name=self.model_name,
            compute_target_name=self.compute_target_name,
            deployment_name="custom-deployment",
            # tiny-cluster has one node, and a deployment can't ask for more instances than that
            instance_count=1,
            max_concurrency_per_instance=2,
            mini_batch_size=5,
            retry_settings_max_retries=5,
            retry_settings_timeout=600,
            description=self.description + " - test_batch_endpoint_with_custom_settings()",
        )

        self.assertIsInstance(batch_endpoint, AzureAIMLStudioAssetsBatchEndpoint)
        self.assertEqual(batch_endpoint.instance_count, 1)
        self.assertEqual(batch_endpoint.max_concurrency_per_instance, 2)
        self.assertEqual(batch_endpoint.mini_batch_size, 5)

        batch_endpoint.delete()

    def test_batch_job_invocation(self):
        """Test that we can invoke a batch job on the endpoint."""
        if not self.is_testable:
            logger.warning("Skipping test_batch_job_invocation() as the environment is not testable.")
            return

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name=self.endpoint_name,
            model_name=self.model_name,
            compute_target_name=self.compute_target_name,
            description=self.description + " - test_batch_job_invocation()",
        )

        test_dataset = AzureAIMLAssetsDataset(
            dataset_name="test-dataset",
            file_name=os.path.join(HERE, "data", "maths.csv"),
        )

        try:
            job = batch_endpoint.invoke_batch_job(
                input_data_path=f"azureml://datastores/workspaceblobstore/paths/{test_dataset.dataset_name}",
                output_data_path="azureml://datastores/workspaceblobstore/paths/test-output/",
            )  # If we get here, the method worked
            self.assertIsNotNone(job)
        # pylint: disable=W0718
        except Exception as e:
            self.fail(f"Batch job invocation failed: {e}")
        finally:
            batch_endpoint.delete()
            test_dataset.delete()
