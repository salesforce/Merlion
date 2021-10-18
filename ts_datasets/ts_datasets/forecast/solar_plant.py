#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import logging
import os
import zipfile

import pandas as pd

from ts_datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class SolarPlant(BaseDataset):
    """
    Wrapper to load the open source solar plant power dataset.

    - source: https://www.nrel.gov/grid/solar-power-data.html
    - contains one 405-variable time series

    .. note::

        The loader currently only includes the first 100 (of 405) variables.
    """

    def __init__(self, rootdir=None, num_columns=100):
        """
        :param rootdir: The root directory at which the dataset can be found.
        :param num_columns: indicates how many univariate columns should be returned
        """
        super().__init__()
        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "multivariate", "solar_plant")

        assert (
            "solar_plant" in rootdir.split("/")[-1]
        ), "solar_plant should be found as the last level of the directory for this dataset"

        # Get all filenames, extracting the zipfile if needed
        fnames = glob.glob(f"{rootdir}/*.csv")
        if len(fnames) == 0 and os.path.isfile(f"{rootdir}/merged.zip"):
            with zipfile.ZipFile(f"{rootdir}/merged.zip", "r") as zip_ref:
                zip_ref.extractall(rootdir)
            fnames = glob.glob(f"{rootdir}/*.csv")
        assert len(fnames) == 1, f"rootdir {rootdir} does not contain dataset file."

        for i, fn in enumerate(sorted(fnames)):

            df = pd.read_csv(fn)

            df["timestamp"] = pd.to_datetime(df["Datetime"])
            df.set_index("timestamp", inplace=True)
            df.drop(["LocalTime", "Datetime"], axis=1, inplace=True)
            num_columns = min(num_columns, len(df.columns))
            cols = [f"Power_{i}" for i in range(num_columns)]
            df = df[cols]
            assert isinstance(df.index, pd.DatetimeIndex)
            df.sort_index(inplace=True)

            self.time_series.append(df)
            self.metadata.append(
                {
                    "trainval": pd.Series(df.index <= "2006-10-01 00:00:00", index=df.index),
                    "granularity": "30min",
                    "aggregation": "Sum",
                }
            )
