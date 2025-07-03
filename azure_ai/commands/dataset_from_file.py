"""
Create a dataset from a local file.
~/Desktop/gh/fswl/azureml-example/azure_ai/tests/data/maths.csv
"""

import argparse
import os

from azure_ai.ml_studio import AzureAIMLAssetsDataset


HERE = os.path.abspath(os.path.dirname(__file__))

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

    file_path = os.path.join(HERE, args.file_path)
    dataset_name = args.dataset_name
    data_set = AzureAIMLAssetsDataset(dataset_name=dataset_name, file_name=file_path)
    pandas_df = data_set.dataset_to_dataframe()
    print(pandas_df.head(n=5))
