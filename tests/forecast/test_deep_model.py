#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging

import sys
import shutil
import unittest

import gdown
import pandas as pd
from os.path import abspath, dirname, join, exists

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.autoformer import AutoformerConfig, AutoformerForecaster
from merlion.models.forecast.transformer import TransformerConfig, TransformerForecaster
from merlion.models.forecast.informer import InformerConfig, InformerForecaster
from merlion.models.forecast.etsformer import ETSformerConfig, ETSformerForecaster


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

        self.n_past = 96
        self.max_forecast_steps = 96
        self.early_stop_patience = 4
        self.num_epochs = 2
        self.use_gpu = True

        df = self._obtain_df("weather")
        bound = 96 * 10
        train_df = df[0:bound]
        test_df = df[bound : 2 * bound]

        self.train_df = train_df
        self.test_df = test_df

        self.train_data = TimeSeries.from_pd(self.train_df)
        self.test_data = TimeSeries.from_pd(self.test_df)

    def test_autoformer(self):

        logger.info("Testing Autoformer forecasting")
        start_token_len = 48
        config = AutoformerConfig(
            n_past=self.n_past,
            max_forecast_steps=self.max_forecast_steps,
            start_token_len=start_token_len,
            early_stop_patience=self.early_stop_patience,
            num_epochs=self.num_epochs,
            use_gpu=self.use_gpu,
        )

        forecaster = AutoformerForecaster(config)

        self._test_model(forecaster, self.train_data, self.test_data)

    def test_transformer(self):
        logger.info("Testing Transformer forecasting")
        start_token_len = 48
        config = TransformerConfig(
            n_past=self.n_past,
            max_forecast_steps=self.max_forecast_steps,
            start_token_len=start_token_len,
            early_stop_patience=self.early_stop_patience,
            num_epochs=self.num_epochs,
            use_gpu=self.use_gpu,
        )

        forecaster = TransformerForecaster(config)

        self._test_model(forecaster, self.train_data, self.test_data)

    def test_informer(self):
        logger.info("Testing Informer forecasting")
        start_token_len = 48

        config = InformerConfig(
            n_past=self.n_past,
            max_forecast_steps=self.max_forecast_steps,
            start_token_len=start_token_len,
            early_stop_patience=self.early_stop_patience,
            num_epochs=self.num_epochs,
            use_gpu=self.use_gpu,
        )

        forecaster = InformerForecaster(config)

        self._test_model(forecaster, self.train_data, self.test_data)

    def test_ETSformer(self):
        logger.info("Testing ETSformer forecasting")
        start_token_len = 0

        config = ETSformerConfig(
            n_past=self.n_past,
            max_forecast_steps=self.max_forecast_steps,
            start_token_len=start_token_len,
            top_K=3,  # top fourier basis
            early_stop_patience=self.early_stop_patience,
            num_epochs=self.num_epochs,
            use_gpu=self.use_gpu,
        )

        forecaster = ETSformerForecaster(config)

        self._test_model(forecaster, self.train_data, self.test_data)

    def _obtain_df(self, dataset_name="weather"):
        data_dir = join(rootdir, "data")
        if dataset_name == "weather":
            data_url = "https://drive.google.com/drive/folders/1Xz84ci5YKWL6O2I-58ZsVe42lYIfqui1"
            data_folder = join(data_dir, "weather")
            data_file_path = join(data_folder, "weather.csv")
        else:
            raise NotImplementedError

        if not exists(data_file_path):
            while True:
                try:
                    gdown.download_folder(data_url, quiet=False, use_cookies=False)
                except TimeoutError:
                    logger.error("Timeout Error, try downloading again...")
                else:
                    logger.info("Successfully downloaded %s!" % (dataset_name))
                    break

            shutil.move("./%s" % (dataset_name), data_folder)

        weather_ds = CustomDataset(data_folder)
        df, metadata = weather_ds[0]

        return df

    def _test_model(self, forecaster, train_data, test_data):
        config = forecaster.config
        logger.info(config.__dict__)

        # training
        forecaster.train(train_data)

        # Single data forecasting testing
        dataset = RollingWindowDataset(
            test_data,
            target_seq_index=config.target_seq_index,
            n_past=config.n_past,
            n_future=config.max_forecast_steps,
            ts_index=True,
        )
        test_prev, test = dataset[0]

        pred, _ = forecaster.forecast(test.time_stamps, time_series_prev=test_prev)

        logger.info("Finishing testing")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
