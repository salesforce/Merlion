#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import logging
import os

import pandas as pd

from ts_datasets.forecast.custom import CustomDataset
from ts_datasets.anomaly.base import TSADBaseDataset

logger = logging.getLogger(__name__)


class CustomAnomalyDataset(CustomDataset, TSADBaseDataset):
    """
    Wrapper to load a custom dataset for anomaly detection. Please review the `tutorial <examples/CustomDataset>`
    to get started.
    """

    def __init__(self, root, test_frac=0.5, assume_no_anomaly=False, time_col=None, time_unit="s", index_cols=None):
        """
        :param root: Filename of a single CSV, or a directory containing many CSVs. Each CSV must contain 1
            or more time series.
        :param test_frac: If we don't find a column "trainval" in the time series, this is the fraction of each
            time series which we use for testing.
        :param assume_no_anomaly: If we don't find a column "anomaly" in the time series, we assume there are no
            anomalies in the data if this value is ``True``, and we throw an exception if this value is ``False``.
        :param time_col: Name of the column used to index time. We use the first non-index, non-metadata column
            if none is given.
        :param time_unit: If the time column is numerical, we assume it is a timestamp expressed in this unit.
        :param index_cols: If a CSV file contains multiple time series, these are the columns used to index those
            time series. For example, a CSV file may contain time series of sales for many (store, department) pairs.
            In this case, ``index_cols`` may be ``["Store", "Dept"]``. The values of the index columns will be added
            to the metadata of the data loader.
        """
        self.assume_no_anomaly = assume_no_anomaly
        super().__init__(root=root, test_frac=test_frac, time_col=time_col, time_unit=time_unit, index_cols=index_cols)

    @property
    def metadata_cols(self):
        return ["anomaly", "trainval"]

    def check_ts_for_metadata(self, ts, col):
        if col == "anomaly":
            if col not in ts:
                if self.assume_no_anomaly:
                    ts[col] = False
                else:
                    raise ValueError(f"Time series {ts} does not have metadata column {col}.")
            ts[col] = ts[col].astype(bool)
        else:
            ts = super().check_ts_for_metadata(ts, col)
        return ts
