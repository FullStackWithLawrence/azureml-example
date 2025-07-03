"""
Test script to verify Azure ML installation is working properly.
"""

import os

from azure_ai.ml_studio import AzureAIMLAssetsDataset

from .test_base import AzureMLTestBase


HERE = os.path.abspath(os.path.dirname(__file__))


class AzureMLTestDataset(AzureMLTestBase):
    """Base class for all unit tests."""

    def test_existing_dataset(self):
        """Test that we can create and use an Azure ML workspace."""
        if not self.is_testable:
            return

        dataset = AzureAIMLAssetsDataset(
            dataset_name="student-performance-base",
        )
        self.assertIsInstance(dataset, AzureAIMLAssetsDataset)

    def test_kaggle_dataset(self):
        """
        Test that we can create and use an Azure ML workspace.
        https://www.kaggle.com/datasets/whenamancodes/student-performance
        """
        if not self.is_testable:
            return

        dataset = AzureAIMLAssetsDataset(
            dataset_name="whenamancodes-student-performance",
            kaggle_dataset="whenamancodes/student-performance",
        )
        self.assertIsInstance(dataset, AzureAIMLAssetsDataset)
        dataset.delete()

    def test_file_dataset(self):
        """
        Test that we can create and use an Azure ML workspace with a file dataset.
        The test file is an Excel workbook incorrected named `maths.csv` that contains
        a single sheet with some sample data. This is a common mistake with
        kaggle datasets where the file is actually an Excel file but has a `.csv` extension.
        The file is located in the `data` directory of this repository.
        """

        if not self.is_testable:
            return

        dataset = AzureAIMLAssetsDataset(
            dataset_name="test-maths-dataset",
            file_name=os.path.join(HERE, "data", "maths.csv"),
        )
        self.assertIsInstance(dataset, AzureAIMLAssetsDataset)
        dataset.delete()
