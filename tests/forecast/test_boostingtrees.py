#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np

from merlion.utils import TimeSeries
from ts_datasets.forecast import SeattleTrail
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.trees import LGBMForecaster, LGBMForecasterConfig
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.transform.normalize import MinMaxNormalize

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestLGBMForecaster(unittest.TestCase):
    """
    we test data loading, model instantiation, forecasting consistency, in particular
    (1) load a testing data
    (2) transform data
    (3) instantiate the model and train
    (4) forecast, and the forecasting result agrees with the reference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_forecast_steps = 2
        self.maxlags = 6
        self.i = 0

        df, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", "seattle_trail"))[0]
        t = int(df[md["trainval"]].index[-1].to_pydatetime().timestamp())
        k = "BGT North of NE 70th Total"
        data = TimeSeries.from_pd(df)
        cleanup = TransformSequence([TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300)])
        cleanup.train(data)
        self.train_data, self.test_data = cleanup(data).bisect(t)
        self.train_data_uni, self.test_data_uni = [d.univariates[k].to_ts() for d in [self.train_data, self.test_data]]

        self.model_autoregression = LGBMForecaster(
            LGBMForecasterConfig(
                max_forecast_steps=self.max_forecast_steps,
                maxlags=self.maxlags,
                target_seq_index=self.i,
                prediction_stride=1,
                n_estimators=20,
                max_depth=5,
                n_jobs=1,
                transform=MinMaxNormalize(),
                invert_transform=True,
            )
        )

        self.model_seq2seq = LGBMForecaster(
            LGBMForecasterConfig(
                max_forecast_steps=self.max_forecast_steps,
                maxlags=self.maxlags,
                target_seq_index=self.i,
                prediction_stride=self.max_forecast_steps,
                n_estimators=20,
                max_depth=5,
                n_jobs=1,
                transform=MinMaxNormalize(),
                invert_transform=True,
            )
        )

    def test_forecast_multi_autoregression(self):
        print("-" * 80)
        logger.info("test_forecast_multi_autoregression" + "\n" + "-" * 80)
        yhat, _ = self.model_autoregression.train(self.train_data)

        # Check RMSE with multivariate forecast inversion
        forecast, _ = self.model_autoregression.forecast(self.max_forecast_steps)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast RMSE: {rmse:.2f}")
        # self.assertAlmostEqual(rmse, 2.9, delta=0.1)

        # Check look-ahead RMSE using time_series_prev
        dataset = RollingWindowDataset(self.test_data, self.i, self.maxlags, self.max_forecast_steps, ts_index=True)
        testing_instance, testing_label = next(iter(dataset))
        pred, _ = self.model_autoregression.forecast(testing_label.time_stamps, testing_instance)
        lookahead_rmse = ForecastMetric.RMSE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead RMSE with time_series_prev: {lookahead_rmse:.2f}")
        # self.assertAlmostEqual(lookahead_rmse, 18.9, delta=0.1)

        # save and load
        self.model_autoregression.save(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_model = LGBMForecaster.load(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(loaded_pred), self.max_forecast_steps)
        self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_multi_seq2seq(self):
        print("-" * 80)
        logger.info("test_forecast_multi_seq2seq" + "\n" + "-" * 80)
        yhat, _ = self.model_seq2seq.train(self.train_data)

        # Check RMSE with multivariate forecast inversion
        forecast, _ = self.model_seq2seq.forecast(self.max_forecast_steps)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast RMSE: {rmse:.2f}")
        # self.assertAlmostEqual(rmse, 3.6, delta=0.1)

        # Check look-ahead RMSE using time_series_prev
        dataset = RollingWindowDataset(self.test_data, self.i, self.maxlags, self.max_forecast_steps, ts_index=True)
        testing_instance, testing_label = next(iter(dataset))
        pred, _ = self.model_seq2seq.forecast(testing_label.time_stamps, testing_instance)
        lookahead_rmse = ForecastMetric.RMSE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead RMSE with time_series_prev: {lookahead_rmse:.2f}")
        # self.assertAlmostEqual(lookahead_rmse, 19.55, delta=0.1)

        # save and load
        self.model_seq2seq.save(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_model = LGBMForecaster.load(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(loaded_pred), self.max_forecast_steps)
        self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_uni(self):
        print("-" * 80)
        logger.info("test_forecast_uni" + "\n" + "-" * 80)
        self.model_autoregression.config.prediction_stride = 2
        yhat, _ = self.model_autoregression.train(self.train_data_uni)

        # Check RMSE with univariate forecast inversion
        forecast, _ = self.model_autoregression.forecast(self.max_forecast_steps)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast RMSE: {rmse:.2f}")
        # self.assertAlmostEqual(rmse, 1.4, delta=0.1)

        # Check look-ahead RMSE using time_series_prev
        dataset = RollingWindowDataset(self.test_data_uni, self.i, self.maxlags, self.max_forecast_steps, ts_index=True)
        testing_instance, testing_label = next(iter(dataset))
        pred, _ = self.model_autoregression.forecast(testing_label.time_stamps, testing_instance)
        lookahead_rmse = ForecastMetric.RMSE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead RMSE with time_series_prev: {lookahead_rmse:.2f}")
        # self.assertAlmostEqual(lookahead_rmse, 17.3, delta=0.1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
