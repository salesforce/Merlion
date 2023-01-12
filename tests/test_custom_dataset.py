#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import os
import pandas as pd
from ts_datasets.forecast import CustomDataset
from ts_datasets.anomaly import CustomAnomalyDataset

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_custom_anom_dataset():
    data_dir = os.path.join(rootdir, "data", "synthetic_anomaly")
    dataset = CustomAnomalyDataset(rootdir=data_dir, test_frac=0.75, time_unit="s", assume_no_anomaly=True)
    assert len(dataset) == len(glob.glob(os.path.join(data_dir, "*.csv")))
    assert all("anomaly" in md.columns and "trainval" in md.columns for ts, md in dataset)
    assert all(abs((~md.trainval).mean() - dataset.test_frac) < 2 / len(ts) for ts, md in dataset)


def test_custom_dataset():
    csv = os.path.join(rootdir, "data", "walmart", "walmart_mini.csv")
    index_cols = ["Store", "Dept"]
    data_cols = ["Weekly_Sales", "Temperature", "CPI"]
    df = pd.read_csv(csv, index_col=[0, 1, 2], parse_dates=True)
    dataset = CustomDataset(rootdir=csv, test_frac=0.25, data_cols=data_cols, index_cols=index_cols)
    assert len(dataset) == len(df.groupby(index_cols).groups)
    assert all(list(ts.columns) == data_cols for ts, md in dataset)
    assert all((c in md.columns for c in ["trainval"] + index_cols) for ts, md in dataset)
    assert all(abs((~md.trainval).mean() - dataset.test_frac) < 2 / len(ts) for ts, md in dataset)
