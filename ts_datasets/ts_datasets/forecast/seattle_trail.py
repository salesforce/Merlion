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


class SeattleTrail(BaseDataset):
    """
    Wrapper to load the open source Seattle Trail pedestrian/bike traffic
    dataset.

    - source: https://www.kaggle.com/city-of-seattle/seattle-burke-gilman-trail
    - contains one 5-variable time series
    """

    def __init__(self, rootdir=None):
        """
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()
        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "multivariate", "seattle_trail")

        assert (
            "seattle_trail" in rootdir.split("/")[-1]
        ), "seattle_trail should be found as the last level of the directory for this dataset"

        dsetdirs = [rootdir]
        extension = "csv"

        fnames = sum([sorted(glob.glob(f"{d}/*.{extension}")) for d in dsetdirs], [])
        assert len(fnames) == 1, f"rootdir {rootdir} does not contain dataset file."
        for i, fn in enumerate(sorted(fnames)):
            df = pd.read_csv(fn)

            df["timestamp"] = pd.to_datetime(df["Date"])
            df.set_index("timestamp", inplace=True)
            df.drop("Date", axis=1, inplace=True)
            assert isinstance(df.index, pd.DatetimeIndex)
            df.sort_index(inplace=True)

            self.time_series.append(df)
            self.metadata.append(
                {"trainval": pd.Series(df.index <= "2019-01-01 00:00:00", index=df.index), "quantile_clip": 300}
            )
