"""
Create a dataset from Kaggle.
https://www.kaggle.com/datasets/heptapod/titanic
"""

import argparse

from azure_ai.ml_studio import AzureAIMLAssetsDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset from Kaggle.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="A dataset name, e.g., titanic",
    )
    parser.add_argument("dataset_name", type=str, help="A Kaggle dataset name, e.g., heptapod/titanic")
    args = parser.parse_args()

    kaggle_dataset = args.dataset_name
    dataset_name = args.dataset_name
    data_set = AzureAIMLAssetsDataset(dataset_name=dataset_name, kaggle_dataset=args.dataset_name)

    pandas_df = data_set.dataset_to_dataframe()
    print(pandas_df.head(n=5))
