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

from merlion.utils import TimeSeries
from ts_datasets.forecast import SeattleTrail
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.models.forecast.trees import RandomForestForecaster, RandomForestForecasterConfig
from merlion.models.utils.seq_ar_common import gen_next_seq_label_pairs

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
                sampling_mode="stats",
                prediction_stride=1,
                n_estimators=20,
            )
        )

    def test_forecast_multi(self):
        logger.info("Training model...")
        yhat, _ = self.model.train(self.train_data)

        name = self.model.target_name
        self.assertAlmostEqual(yhat.univariates[name].np_values.mean(), 0.50, 1)
        forecast = self.model.forecast(self.max_forecast_steps)[0]
        self.assertAlmostEqual(forecast.to_pd().mean().item(), 0.5, delta=0.1)
        testing_data_gen = gen_next_seq_label_pairs(self.test_data, self.i, self.maxlags, self.max_forecast_steps)
        testing_instance, testing_label = next(testing_data_gen)
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(pred), self.max_forecast_steps)
        pred = pred.univariates[name].np_values
        self.assertAlmostEqual(pred.mean(), 0.50, 1)

        # save and load
        self.model.save(dirname=join(rootdir, "tmp", "randomforestforecaster"))
        loaded_model = RandomForestForecaster.load(dirname=join(rootdir, "tmp", "randomforestforecaster"))
        loaded_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(loaded_pred), self.max_forecast_steps)
        self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_uni(self):
        logger.info("Training model...")
        self.model.config.prediction_stride = 2
        yhat, _ = self.model.train(self.train_data_uni)
        name = self.model.target_name

        self.assertAlmostEqual(yhat.univariates[name].np_values.mean(), 0.50, 1)
        forecast = self.model.forecast(self.max_forecast_steps)[0]
        self.assertAlmostEqual(forecast.to_pd().mean().item(), 0.5, delta=0.1)
        testing_data_gen = gen_next_seq_label_pairs(self.test_data_uni, self.i, self.maxlags, self.max_forecast_steps)
        testing_instance, testing_label = next(testing_data_gen)
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(pred), self.max_forecast_steps)
        pred = pred.univariates[name].np_values
        self.assertAlmostEqual(pred.mean(), 0.50, 1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
