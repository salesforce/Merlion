#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import os

import pandas as pd

from ts_datasets.anomaly.base import TSADBaseDataset


class Synthetic(TSADBaseDataset):
    """
    Wrapper to load a sythetically generated dataset.
    The dataset was generated using three base time series, each of which
    was separately injected with shocks, spikes, dips and level shifts, making
    a total of 15 time series (including the base time series without anomalies).
    Subsets can are defined by the base time series used ("horizontal",
    "seasonal", "upward_downward"), or the type of injected anomaly ("shock",
    "spike", "dip", "level"). The "anomaly" subset refers to all times series with
    injected anomalies (12) while "base" refers to all time series without them (3).
    """

    base_ts_subsets = ["horizontal", "seasonal", "upward_downward"]
    anomaly_subsets = ["shock", "spike", "dip", "level", "trend"]
    valid_subsets = ["anomaly", "all", "base"] + base_ts_subsets + anomaly_subsets

    def __init__(self, subset="anomaly", rootdir=None):
        super().__init__()

        assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"
        self.subset = subset

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "synthetic_anomaly")

        csvs = sorted(glob.glob(f"{rootdir}/*.csv"))
        if subset == "base":
            csvs = [csv for csv in csvs if "anom" not in os.path.basename(csv)]
        elif subset != "all":
            csvs = [csv for csv in csvs if "anom" in os.path.basename(csv)]
        if subset in self.base_ts_subsets + self.anomaly_subsets:
            csvs = [csv for csv in csvs if subset in os.path.basename(csv)]

        for csv in csvs:
            df = pd.read_csv(csv)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.set_index("timestamp")

            ts = df[df.columns[0:1]]
            metadata = pd.DataFrame(
                {
                    "anomaly": df["anomaly"].astype(bool) if df.shape[1] > 1 else [False] * len(df),
                    "trainval": [j < len(df) * 0.5 for j in range(len(df))],
                },
                index=df.index,
            )

            self.time_series.append(ts)
            self.metadata.append(metadata)
