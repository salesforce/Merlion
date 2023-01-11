#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import sys
import unittest

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.trees import RandomForestForecaster, RandomForestForecasterConfig
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.utils import TimeSeries
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from ts_datasets.forecast import SeattleTrail

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestRandomForestForecaster(unittest.TestCase):
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
        cleanup = TransformSequence(
            [TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300), MinMaxNormalize()]
        )
        cleanup.train(data)
        self.train_data, self.test_data = cleanup(data).bisect(t)
        self.train_data_uni, self.test_data_uni = [d.univariates[k].to_ts() for d in [self.train_data, self.test_data]]

        self.model = RandomForestForecaster(
            RandomForestForecasterConfig(
                max_forecast_steps=self.max_forecast_steps,
                maxlags=self.maxlags,
                target_seq_index=self.i,
                prediction_stride=1,
                n_estimators=20,
                invert_forecast=False,
            )
        )

    def test_forecast_multi(self):
        logger.info("Training multivariate model...")
        yhat, _ = self.model.train(self.train_data)

        # Check RMSE with multivariate forecast inversion
        forecast, _ = self.model.forecast(self.max_forecast_steps)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast RMSE: {rmse:.2f}")
        # self.assertAlmostEqual(rmse, 0.08, delta=0.1)

        # Check look-ahead RMSE using time_series_prev
        dataset = RollingWindowDataset(self.test_data, self.i, self.maxlags, self.max_forecast_steps, ts_index=True)
        testing_instance, testing_label = next(iter(dataset))
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        lookahead_rmse = ForecastMetric.RMSE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead RMSE with time_series_prev: {lookahead_rmse:.2f}")
        # self.assertAlmostEqual(lookahead_rmse, 0.14, delta=0.1)

        # save and load
        self.model.save(dirname=join(rootdir, "tmp", "randomforestforecaster"))
        loaded_model = RandomForestForecaster.load(dirname=join(rootdir, "tmp", "randomforestforecaster"))
        loaded_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(loaded_pred), self.max_forecast_steps)
        self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_uni(self):
        logger.info("Training univariate model with prediction stride 2...")
        self.model.config.prediction_stride = 2
        yhat, _ = self.model.train(self.train_data_uni)

        # Check RMSE with univariate forecast inversion
        forecast, _ = self.model.forecast(self.max_forecast_steps)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast RMSE: {rmse:.2f}")
        self.assertAlmostEqual(rmse, 0.01, delta=0.1)

        # Check look-ahead RMSE using time_series_prev
        dataset = RollingWindowDataset(self.test_data_uni, self.i, self.maxlags, self.max_forecast_steps, ts_index=True)
        testing_instance, testing_label = next(iter(dataset))
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        lookahead_rmse = ForecastMetric.RMSE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead RMSE with time_series_prev: {lookahead_rmse:.2f}")
        self.assertAlmostEqual(lookahead_rmse, 0.06, delta=0.1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
