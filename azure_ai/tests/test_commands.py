"""
Tests of the command-line entry points in azure_ai/commands, offline.

Each test runs a command the way `python3 -m azure_ai.commands.<name>` does,
with runpy, and replaces the helper class that the command uses with a mock,
so nothing reaches Azure or Kaggle.
"""

import io
import os
import runpy
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pandas as pd

from .test_base import AzureMLTestBase

HERE = os.path.abspath(os.path.dirname(__file__))


class AzureMLTestCommands(AzureMLTestBase):
    """Tests of the azure_ai.commands modules."""

    def run_command(self, module_name: str, *args: str) -> str:
        """
        Run a command module as `python3 -m azure_ai.commands.<module_name> <args>` would.

        Args:
            module_name: The command's module name, e.g. workspace
            args: The command-line arguments

        Returns:
            str: What the command printed
        """
        output = io.StringIO()
        with patch.object(sys, "argv", [module_name, *args]), redirect_stdout(output):
            runpy.run_module(f"azure_ai.commands.{module_name}", run_name="__main__")
        return output.getvalue()

    def test_help(self):
        """Test that help prints a usage line for every command."""
        output = self.run_command("help")

        for command in ("workspace", "compute_cluster", "dataset_from_file", "dataset_from_kaggle", "help"):
            self.assertIn(f"python3 -m azure_ai.commands.{command}", output)

    def test_workspace(self):
        """Test that workspace prints the workspace from config.json."""
        with patch("azure_ai.ml_studio.AzureAIMLWorkspace") as mock_workspace_class:
            mock_workspace_class.return_value.workspace = "my-workspace"
            output = self.run_command("workspace")

        self.assertIn("my-workspace", output)

    def test_compute_cluster(self):
        """Test that compute_cluster gets or creates the named cluster, and prints it as YAML."""
        with patch("azure_ai.ml_studio.AzureAIMLStudioComputeCluster") as mock_cluster_class:
            cluster = mock_cluster_class.return_value.get_or_create.return_value
            self.run_command("compute_cluster", "tiny-cluster")

        mock_cluster_class.assert_called_once_with(cluster_name="tiny-cluster")
        # dump() writes to the command's sys.stdout, which run_command() captured
        dump_stream = cluster.dump.call_args.args[0]
        self.assertIsInstance(dump_stream, io.StringIO)

    def test_dataset_from_file(self):
        """Test that dataset_from_file registers the file, resolving a relative path, and prints the first rows."""
        with patch("azure_ai.ml_studio.AzureAIMLAssetsDataset") as mock_dataset_class:
            mock_dataset_class.return_value.dataset_to_dataframe.return_value = pd.DataFrame({"G3": [6, 10]})
            relative_path = os.path.relpath(os.path.join(HERE, "data", "maths.csv"))
            output = self.run_command("dataset_from_file", "maths", relative_path)

        mock_dataset_class.assert_called_once_with(
            dataset_name="maths", file_name=os.path.join(HERE, "data", "maths.csv")
        )
        self.assertIn("G3", output)

    def test_dataset_from_kaggle(self):
        """Test that dataset_from_kaggle registers the Kaggle dataset, and prints the first rows."""
        with patch("azure_ai.ml_studio.AzureAIMLAssetsDataset") as mock_dataset_class:
            mock_dataset_class.return_value.dataset_to_dataframe.return_value = pd.DataFrame({"2urvived": [0, 1]})
            output = self.run_command("dataset_from_kaggle", "heptapod-titanic", "heptapod/titanic")

        mock_dataset_class.assert_called_once_with(dataset_name="heptapod-titanic", kaggle_dataset="heptapod/titanic")
        self.assertIn("2urvived", output)

    def test_missing_argument(self):
        """Test that a command run without its required argument exits with argparse's usage error."""
        with patch("azure_ai.ml_studio.AzureAIMLStudioComputeCluster", MagicMock()):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as context:
                self.run_command("compute_cluster")

        self.assertEqual(context.exception.code, 2)
