#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
import logging
from collections import OrderedDict
from merlion.dashboard.models.utils import DataMixin
from merlion.dashboard.pages.utils import create_empty_figure
from merlion.dashboard.utils.log import DashLogger
from merlion.dashboard.utils.plot import data_table, plot_timeseries

dash_logger = DashLogger(stream=sys.stdout)


class DataAnalyzer(DataMixin):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)

    @staticmethod
    def get_stats(df):
        stats = {
            "@global": OrderedDict(
                {
                    "NO. of Variables": len(df.columns),
                    "Time Series Length": df.shape[0],
                    "Has NaNs": bool(df.isnull().values.any()),
                }
            ),
            "@columns": list(df.columns),
        }
        for col in df.columns:
            info = df[col].describe()
            data = OrderedDict(zip(info.index.values, info.values))
            stats[col] = data
        return stats

    @staticmethod
    def get_data_table(df):
        return data_table(df)

    @staticmethod
    def get_data_figure(df):
        if df is None:
            return create_empty_figure()
        else:
            return plot_timeseries(df)
