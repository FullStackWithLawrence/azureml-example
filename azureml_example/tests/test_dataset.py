"""
Test script to verify Azure ML installation is working properly.
"""

import os

from azureml_example.automated_ml import AzureMLDataset

from .test_base import AzureMLTestBase


HERE = os.path.abspath(os.path.dirname(__file__))


class SmarterTestBase(AzureMLTestBase):
    """Base class for all unit tests."""

    def test_existing_dataset(self):
        """Test that we can create and use an Azure ML workspace."""
        if not self.is_testable:
            return

        dataset = AzureMLDataset(
            dataset_name="student-performance-base",
        )
        self.assertIsInstance(dataset, AzureMLDataset)

    def test_kaggle_dataset(self):
        """
        Test that we can create and use an Azure ML workspace.
        https://www.kaggle.com/datasets/whenamancodes/student-performance
        """
        if not self.is_testable:
            return

        dataset = AzureMLDataset(
            dataset_name="whenamancodes-student-performance",
            kaggle_dataset="whenamancodes/student-performance",
        )
        self.assertIsInstance(dataset, AzureMLDataset)
        dataset.delete()

    def test_file_dataset(self):
        """Test that we can create and use an Azure ML workspace with a file dataset."""

        if not self.is_testable:
            return

        dataset = AzureMLDataset(
            dataset_name="test-maths-dataset",
            file_name=os.path.join(HERE, "data", "maths.csv"),
        )
        self.assertIsInstance(dataset, AzureMLDataset)
        dataset.delete()
