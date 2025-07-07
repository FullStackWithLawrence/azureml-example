"""
Test script to verify Azure ML Batch Endpoint functionality.
"""

import hashlib
import os
from datetime import datetime
from logging import getLogger

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
        try:
            endpoint = AzureAIMLStudioAssetsBatchEndpoint(
                endpoint_name=cls.endpoint_name,
                model_name=cls.model_name,
            )
            endpoint.delete()
        # pylint: disable=W0718
        except Exception as e:
            logger.error("Failed to clean up batch endpoint: %s", e)

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
        """Test that we can retrieve an existing batch endpoint."""
        if not self.is_testable:
            logger.warning("Skipping test_existing_batch_endpoint() as the environment is not testable.")
            return

        # First create an endpoint
        endpoint_name = AzureAIMLStudioAssetsBatchEndpoint.get_default_name(AzureAIMLWorkspace().workspace)
        if not endpoint_name:
            logger.warning("No default endpoint name found, skipping this test.")
            return

        # Then try to retrieve it
        existing_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name=endpoint_name,
            model_name=self.model_name,
        )

        self.assertIsInstance(existing_endpoint, AzureAIMLStudioAssetsBatchEndpoint)
        self.assertEqual(existing_endpoint.endpoint_name, endpoint_name)

    def test_batch_endpoint_with_custom_settings(self):
        """Test batch endpoint creation with custom deployment settings."""
        if not self.is_testable:
            logger.warning("Skipping test_batch_endpoint_with_custom_settings() as the environment is not testable.")
            return

        batch_endpoint = AzureAIMLStudioAssetsBatchEndpoint(
            endpoint_name=self.endpoint_name,
            model_name=self.model_name,
            deployment_name="custom-deployment",
            instance_count=3,
            max_concurrency_per_instance=2,
            mini_batch_size=5,
            retry_settings_max_retries=5,
            retry_settings_timeout=600,
            description=self.description + " - test_batch_endpoint_with_custom_settings()",
        )

        self.assertIsInstance(batch_endpoint, AzureAIMLStudioAssetsBatchEndpoint)
        self.assertEqual(batch_endpoint.instance_count, 3)
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

        batch_endpoint.delete()
