"""Tests of the helpers in azure_ai.ml_studio.utils, which make no Azure calls."""

import os
import tempfile

import pandas as pd

from azure_ai.ml_studio.utils import parse_datastore_path, write_mltable_folder

from .test_base import AzureMLTestBase


class AzureMLTestUtils(AzureMLTestBase):
    """Tests of write_mltable_folder() and parse_datastore_path()."""

    def test_write_mltable_folder(self):
        """Test that write_mltable_folder() writes the CSV file, and an MLTable file that reads it by name."""
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})

        with tempfile.TemporaryDirectory() as folder:
            write_mltable_folder(df, folder, "maths.csv")

            self.assertEqual(sorted(os.listdir(folder)), ["MLTable", "maths.csv"])
            with open(os.path.join(folder, "MLTable"), encoding="utf-8") as mltable_file:
                mltable = mltable_file.read()
            written = pd.read_csv(os.path.join(folder, "maths.csv"))

        self.assertIn("type: mltable", mltable)
        self.assertIn("- file: ./maths.csv", mltable)
        pd.testing.assert_frame_equal(written, df)

    def test_parse_datastore_path(self):
        """Test that parse_datastore_path() finds the datastore and the path in it, without a trailing slash."""
        prefix = "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/datastores/workspaceblobstore/paths/"

        self.assertEqual(
            parse_datastore_path(prefix + "LocalUpload/abc123/maths/"),
            ("workspaceblobstore", "LocalUpload/abc123/maths"),
        )
        self.assertEqual(
            parse_datastore_path(prefix + "UI/2025-07-04/train.csv/"),
            ("workspaceblobstore", "UI/2025-07-04/train.csv"),
        )

    def test_parse_datastore_path_outside_datastore(self):
        """Test that parse_datastore_path() returns None for a path that isn't in a datastore, e.g. a web url."""
        self.assertIsNone(parse_datastore_path("https://example.com/maths.csv"))
        self.assertIsNone(parse_datastore_path(""))
