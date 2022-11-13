#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import sys
import pdb
import unittest

import pandas as pd

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.defaults import DefaultForecaster, DefaultForecasterConfig
from merlion.models.forecast.autoformer import AutoformerConfig, AutoformerForecaster

from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.transform.bound import LowerUpperClip
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.resample import TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils.time_series import TimeSeries, to_pd_datetime
from ts_datasets.forecast import SeattleTrail
from ts_datasets.forecast.custom import CustomDataset

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestDeepModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_root_path = "/Users/yihaofeng/workspace/ts_projects/Autoformer/dataset/weather/"
        data_path = "weather.csv"
        weather_ds = CustomDataset(data_root_path)
        df, metadata = weather_ds[0]
        bound = 96 * 2
        df = df[0:bound]

        self.df = df
        self.ts_data = TimeSeries.from_pd(df)

        self.config = AutoformerConfig(
            n_past=96,
            max_forecast_steps=96,
            start_token_len=48,
            dim=self.ts_data.dim,
        )
        self.forecaster = AutoformerForecaster(self.config)

    def test_model(self):
        logger.info(self.config.__dict__)

        self.forecaster.evaluate(self.ts_data, "mse")
        logger.info("Finish Evalaution")

        self.forecaster.train(self.ts_data, self.config)
        logger.info("Hello world")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
