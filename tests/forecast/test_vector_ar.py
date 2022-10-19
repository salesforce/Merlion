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

from ts_datasets.forecast import SeattleTrail
from merlion.evaluate.forecast import ForecastMetric
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.models.factory import instantiate_or_copy_model, ModelFactory
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestVectorAR(unittest.TestCase):
    """
    we test data loading, model instantiation, forecasting consistency, in particular
    (1) load a testing data
    (2) transform data
    (3) instantiate the VectorAR model and train
    (4) forecast, and the forecasting result agrees with the reference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_forecast_steps = 3
        self.maxlags = 28
        self.i = 0

        df, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", "seattle_trail"))[0]
        t = int(df[md["trainval"]].index[-1].to_pydatetime().timestamp())
        data = TimeSeries.from_pd(df)
        cleanup = TransformSequence(
            [TemporalResample(granularity="1d", missing_value_policy="FFill"), LowerUpperClip(upper=300)]
        )
        cleanup.train(data)
        self.train_data, self.test_data = cleanup(data).bisect(t)

        self.model = instantiate_or_copy_model(
            dict(
                name="VectorAR",
                max_forecast_steps=self.max_forecast_steps,
                maxlags=self.maxlags,
                target_seq_index=self.i,
                transform=MinMaxNormalize(),
                invert_transform=True,
            )
        )

    def run_test(self, univariate):
        logger.info("Training model...")
        if univariate:
            name = self.train_data.names[self.i]
            self.train_data = self.train_data.univariates[name].to_ts()
            self.test_data = self.test_data.univariates[name].to_ts()
            self.model.config.maxlags = self.maxlags = 7

        yhat, _ = self.model.train(self.train_data)

        # Check RMSE with multivariate forecast inversion
        forecast, _ = self.model.forecast(self.max_forecast_steps)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast RMSE: {rmse:.2f}")

        # Check look-ahead sMAPE using time_series_prev
        dataset = RollingWindowDataset(self.test_data, self.i, self.maxlags, self.max_forecast_steps, ts_index=True)
        testing_instance, testing_label = next(iter(dataset))
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        lookahead_rmse = ForecastMetric.RMSE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead RMSE with time_series_prev: {lookahead_rmse:.2f}")

        # save and load
        if not univariate:
            self.model.save(dirname=join(rootdir, "tmp", "vector_ar"))
            loaded_model = ModelFactory.load(name="VectorAR", model_path=join(rootdir, "tmp", "vector_ar"))
            loaded_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
            self.assertEqual(len(loaded_pred), self.max_forecast_steps)
            self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_univariate(self):
        print("-" * 80)
        logger.info("test_forecast_univariate\n" + "-" * 80)
        self.run_test(True)

    def test_forecast_multivariate(self):
        print("-" * 80)
        logger.info("test_forecast_multivariate\n" + "-" * 80)
        self.run_test(False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
