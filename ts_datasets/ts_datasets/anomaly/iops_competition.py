#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import zipfile

import numpy as np
import pandas as pd

from ts_datasets.anomaly.base import TSADBaseDataset


class IOpsCompetition(TSADBaseDataset):
    """
    Wrapper to load the dataset used for the final round of the IOPs competition
    (http://iops.ai/competition_detail/?competition_id=5).

    The dataset contains 29 time series of KPIs gathered from large tech
    companies (Alibaba, Sogou, Tencent, Baidu, and eBay). These time series are
    sampled at either 1min or 5min intervals, and are split into train and test
    sections.

    Note that the original competition prohibited algorithms which directly
    hard-coded the KPI ID to set model parameters. So training a new model for
    each time series was against competition rules. They did, however, allow
    algorithms which analyzed each time series (in an automated way), and used
    the results of that automated analysis to perform algorithm/model selection.
    """

    def __init__(self, rootdir=None):
        """
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()
        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "iops_competition")

        # Try to extract the zip file if possible
        train = os.path.join(rootdir, "phase2_train.csv")
        test = os.path.join(rootdir, "phase2_test.csv")
        if not os.path.isfile(train) or not os.path.isfile(test):
            z = os.path.join(rootdir, "phase2.zip")
            if os.path.isfile(z):
                with zipfile.ZipFile(z, "r") as zip_ref:
                    zip_ref.extractall(rootdir)
            else:
                raise FileNotFoundError(
                    f"Directory {rootdir} contains neither the extracted files "
                    f"phase2_train.csv and phase2_ground_truth.hdf, nor the "
                    f"compressed archive phase2.zip"
                )

        # Read the extracted CSVs to pandas dataframes
        train_df, test_df = pd.read_csv(train), pd.read_csv(test)

        for df in [train_df, test_df]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            new_columns = df.columns.values
            new_columns[2] = "anomaly"
            df.columns = new_columns
            df["anomaly"] = df["anomaly"].astype(bool)

        self.kpi_ids = sorted(train_df["KPI ID"].unique())
        for kpi in self.kpi_ids:
            train = train_df[train_df["KPI ID"] == kpi].drop(columns="KPI ID")
            train.insert(3, "trainval", np.ones(len(train), dtype=bool))
            test = test_df[test_df["KPI ID"] == kpi].drop(columns="KPI ID")
            test.insert(3, "trainval", np.zeros(len(test), dtype=bool))
            full = pd.concat([train, test]).set_index("timestamp")

            md_cols = ["anomaly", "trainval"]
            self.metadata.append(full[md_cols])
            self.time_series.append(full[[c for c in full.columns if c not in md_cols]])

    @property
    def max_lag_sec(self):
        """
        The IOps competition allows anomalies to be detected up to 35min after
        they start. We are currently not using this, but we are leaving the
        override here as a placeholder, if we want to change it later.
        """
        return None  # 35 * 60
