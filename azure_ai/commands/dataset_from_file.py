"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Create a dataset from a local file.
usage:
  python3 -m azure_ai.commands.dataset_from_file maths ~/Desktop/gh/fswl/azureml-example/azure_ai/tests/data/maths.csv
"""

import argparse
import os

from azure_ai.ml_studio import AzureAIMLAssetsDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset from a local file.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="A dataset name, e.g., maths",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="A path to a local file, e.g., ~/Desktop/gh/fswl/azureml-example/azure_ai/tests/data/maths.csv",
    )
    args = parser.parse_args()

    # A relative path is relative to where you run the command, and ~ is your home folder
    file_path = os.path.abspath(os.path.expanduser(args.file_path))
    dataset_name = args.dataset_name
    data_set = AzureAIMLAssetsDataset(dataset_name=dataset_name, file_name=file_path)
    pandas_df = data_set.dataset_to_dataframe()
    print(pandas_df.head(n=5))
