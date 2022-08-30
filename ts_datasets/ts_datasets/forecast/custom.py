#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import os

import pandas as pd

from ts_datasets.base import BaseDataset


class CustomDataset(BaseDataset):
    """
    Wrapper to load a custom dataset. Please review the `tutorial <examples/CustomDataset>` to get started.
    """

    def __init__(self, rootdir, test_frac=0.5, time_col=None, time_unit="s", data_cols=None, index_cols=None):
        """
        :param rootdir: Filename of a single CSV, or a directory containing many CSVs. Each CSV must contain 1
            or more time series.
        :param test_frac: If we don't find a column "trainval" in the time series, this is the fraction of each
            time series which we use for testing.
        :param time_col: Name of the column used to index time. We use the first non-index, non-metadata column
            if none is given.
        :param time_unit: If the time column is numerical, we assume it is a timestamp expressed in this unit.
        :param data_cols: Name of the columns to fetch from the dataset. If ``None``, use all non-time, non-index columns.
        :param index_cols: If a CSV file contains multiple time series, these are the columns used to index those
            time series. For example, a CSV file may contain time series of sales for many (store, department) pairs.
            In this case, ``index_cols`` may be ``["Store", "Dept"]``. The values of the index columns will be added
            to the metadata of the data loader.
        """
        super().__init__()
        assert (
            rootdir is not None and os.path.isfile(rootdir) or os.path.isdir(rootdir)
        ), "You must give CSV file or directory where the data lives."
        csvs = sorted(glob.glob(os.path.join(rootdir, "*.csv*"))) if os.path.isdir(rootdir) else [rootdir]
        assert len(csvs) > 0, f"The rootdir {rootdir} must contain at least 1 CSV file. None found."
        self.test_frac = test_frac
        self.rootdir = rootdir

        for csv in csvs:
            df = pd.read_csv(csv)
            index_cols = index_cols or []
            # By default, the time column is the first non-index, non-metadata column
            if time_col is None:
                time_col = [c for c in df.columns if c not in index_cols + self.metadata_cols][0]
            df[time_col] = pd.to_datetime(df[time_col], unit=None if df[time_col].dtype == "O" else time_unit)

            # Make sure we have metadata columns, and make sure they don't overlap with time/index columns
            assert all(
                c not in index_cols + [time_col] for c in self.metadata_cols
            ), f"None of the metadata columns {self.metadata_cols} can be the time column or index columns"

            # Split into multiple time series dataframes based on index
            df.set_index(index_cols + [time_col], inplace=True)
            df = df.loc[:, data_cols] if data_cols is not None else df
            if len(index_cols) > 0:
                dfs = [df.loc[idx] for idx in df.groupby(index_cols).groups.values()]
            else:
                dfs = [df]

            # Add the dataframes to self.time_series and self.metadata. We include index columns in metadata.
            for ts in dfs:
                ts = ts.reset_index().set_index(time_col)
                for col in self.metadata_cols:
                    ts = self.check_ts_for_metadata(ts, col)
                md = ts.loc[:, self.metadata_cols + index_cols]
                ts = ts.drop(columns=md.columns)
                self.time_series.append(ts)
                self.metadata.append(md)

    @property
    def metadata_cols(self):
        return ["trainval"]

    def check_ts_for_metadata(self, ts, col):
        if col not in ts:
            if col == "trainval":
                ts[col] = ts.index <= ts.index[0] + (ts.index[-1] - ts.index[0]) * (1 - self.test_frac)
            else:
                raise ValueError(f"Expected time series {ts} to have metadata column {col}. Columns: {ts.columns}")
        return ts
