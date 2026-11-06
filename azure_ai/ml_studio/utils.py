"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Small helpers for the Azure ML helper classes, with no Azure calls of their own.
"""

import os
import re

import pandas as pd

from .const import MLTABLE_TEMPLATE

# e.g. azureml://subscriptions/.../datastores/workspaceblobstore/paths/LocalUpload/<hash>/<folder>/
DATASTORE_PATH_PATTERN = re.compile(r"datastores/(?P<datastore>[^/]+)/paths/(?P<path>.*)$")


def write_mltable_folder(df: pd.DataFrame, folder: str, csv_filename: str) -> None:
    """
    Write a DataFrame to a folder as an MLTable: the data as a CSV file plus an MLTable file that reads it.

    Args:
        df: The data
        folder: An existing folder, e.g. a temporary one to upload
        csv_filename: Name of the CSV file, e.g. maths.csv
    """
    csv_path = os.path.join(folder, csv_filename)
    df.to_csv(csv_path, index=False)

    mltable = MLTABLE_TEMPLATE.format(csv_filename=csv_filename)
    mltable_path = os.path.join(folder, "MLTable")
    with open(mltable_path, "w", encoding="utf-8") as mltable_file:
        mltable_file.write(mltable)


def parse_datastore_path(dataset_path: str) -> tuple[str, str] | None:
    """
    Split a data asset's path into its datastore name and the path inside that datastore.

    A data asset made in the studio points at the file itself, with a trailing
    slash: .../paths/UI/<date>/train.csv/. Dropping the slash gives a prefix
    that finds the files in a folder and a file on its own alike.

    Args:
        dataset_path: The data asset's path, e.g. azureml://.../datastores/workspaceblobstore/paths/LocalUpload/abc/maths/

    Returns:
        tuple: (datastore name, path inside the datastore), or None if the path isn't in a datastore
    """
    match = DATASTORE_PATH_PATTERN.search(dataset_path)
    if not match:
        return None
    return match.group("datastore"), match.group("path").rstrip("/")
