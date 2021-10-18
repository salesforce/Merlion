#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import logging
import os

import pandas as pd

from ts_datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class EnergyPower(BaseDataset):
    """
    Wrapper to load the open source energy grid power usage dataset.

    - source: https://www.kaggle.com/robikscube/hourly-energy-consumption
    - contains one 10-variable time series
    """

    def __init__(self, rootdir=None):
        """
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()
        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "multivariate", "energy_power")

        assert (
            "energy_power" in rootdir.split("/")[-1]
        ), "energy_power should be found as the last level of the directory for this dataset"

        dsetdirs = [rootdir]
        extension = "csv.gz"

        fnames = sum([sorted(glob.glob(f"{d}/*.{extension}")) for d in dsetdirs], [])
        assert len(fnames) == 1, f"rootdir {rootdir} does not contain dataset file."

        start_timestamp = "2014-01-01 00:00:00"

        for i, fn in enumerate(sorted(fnames)):
            df = pd.read_csv(fn, index_col="Datetime", parse_dates=True)
            df = df[df.index >= start_timestamp]
            df.drop(["NI", "PJM_Load"], axis=1, inplace=True)
            df.index.rename("timestamp", inplace=True)
            assert isinstance(df.index, pd.DatetimeIndex)
            df.sort_index(inplace=True)

            self.time_series.append(df)
            self.metadata.append(
                {
                    "trainval": pd.Series(df.index <= "2018-01-01 00:00:00", index=df.index),
                    "start_timestamp": start_timestamp,
                }
            )
