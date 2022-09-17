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
from merlion.models.utils.seq_ar_common import gen_next_seq_label_pairs
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

        dataset = "seattle_trail"
        df, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", dataset))[0]
        t = int(df[md["trainval"]].index[-1].to_pydatetime().timestamp())
        data = TimeSeries.from_pd(df)
        cleanup = TransformSequence([TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300)])
        cleanup.train(data)
        self.train_data, self.test_data = cleanup(data).bisect(t)

        data_uni = TimeSeries.from_pd(df["BGT North of NE 70th Total"])
        cleanup = TransformSequence([TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300)])
        cleanup.train(data_uni)
        self.train_data_uni, self.test_data_uni = cleanup(data_uni).bisect(t)

        self.model = LGBMForecaster(
            LGBMForecasterConfig(
                max_forecast_steps=self.max_forecast_steps,
                maxlags=self.maxlags,
                target_seq_index=self.i,
                sampling_mode="normal",
                prediction_stride=1,
                n_estimators=20,
                max_depth=5,
                n_jobs=1,
                transform=MinMaxNormalize(),
                invert_transform=True,
            )
        )

    def test_forecast_multi(self):
        logger.info("Training multivariate model...")
        yhat, _ = self.model.train(self.train_data)

        # Check sMAPE with multi-dimensional forecast inversion
        forecast, _ = self.model.forecast(self.max_forecast_steps)
        smape = ForecastMetric.sMAPE.value(self.test_data, forecast, target_seq_index=self.i)
        logger.info(f"Immediate forecast sMAPE: {smape:.2f}")
        self.assertAlmostEqual(smape, 7.09, delta=0.1)

        # Check look-ahead sMAPE using time_series_prev
        testing_data_gen = gen_next_seq_label_pairs(self.test_data, self.i, self.maxlags, self.max_forecast_steps)
        testing_instance, testing_label = next(testing_data_gen)
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        lookahead_smape = ForecastMetric.sMAPE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead sMAPE with time_series_prev: {lookahead_smape:.2f}")
        self.assertAlmostEqual(lookahead_smape, 13.8, delta=0.1)

        # save and load
        self.model.save(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_model = LGBMForecaster.load(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(loaded_pred), self.max_forecast_steps)
        self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_uni(self):
        logger.info("Training univariate model with prediction stride 2...")
        self.model.config.prediction_stride = 2
        yhat, _ = self.model.train(self.train_data_uni)

        forecast, _ = self.model.forecast(self.max_forecast_steps)
        smape = ForecastMetric.sMAPE.value(self.test_data_uni, forecast)
        logger.info(f"Immediate forecast sMAPE: {smape:.2f}")
        self.assertAlmostEqual(smape, 4.8, delta=0.1)

        testing_data_gen = gen_next_seq_label_pairs(self.test_data_uni, self.i, self.maxlags, self.max_forecast_steps)
        testing_instance, testing_label = next(testing_data_gen)
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        lookahead_smape = ForecastMetric.sMAPE.value(testing_label, pred, target_seq_index=self.i)
        logger.info(f"Look-ahead sMAPE with time_series_prev: {lookahead_smape:.2f}")
        self.assertAlmostEqual(lookahead_smape, 12.7, delta=0.1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
