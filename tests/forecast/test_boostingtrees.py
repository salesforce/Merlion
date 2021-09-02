#
# Copyright (c) 2021 salesforce.com, inc.
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
from merlion.transform.moving_average import DifferenceTransform
from merlion.models.forecast.boostingtrees import LGBMForecaster, LGBMForecasterConfig
from merlion.models.forecast.seq_ar_common import gen_next_seq_label_pairs

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
        # t = int(datetime(2019, 1, 1, 0, 0, 0).timestamp())

        dataset = "seattle_trail"
        d, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", dataset))[0]
        d_uni = d["BGT North of NE 70th Total"]
        t = int(d[md["trainval"]].index[-1].to_pydatetime().timestamp())
        data = TimeSeries.from_pd(d)
        cleanup_transform = TransformSequence(
            [TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300), DifferenceTransform()]
        )
        cleanup_transform.train(data)
        data = cleanup_transform(data)

        train_data, test_data = data.bisect(t)

        minmax_transform = MinMaxNormalize()
        minmax_transform.train(train_data)
        self.train_data_norm = minmax_transform(train_data)
        self.test_data_norm = minmax_transform(test_data)

        data_uni = TimeSeries.from_pd(d_uni)
        cleanup_transform = TransformSequence(
            [TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300), DifferenceTransform()]
        )
        cleanup_transform.train(data_uni)
        data_uni = cleanup_transform(data_uni)

        train_data_uni, test_data_uni = data_uni.bisect(t)

        minmax_transform = MinMaxNormalize()
        minmax_transform.train(train_data_uni)
        self.train_data_uni_norm = minmax_transform(train_data_uni)
        self.test_data_uni_norm = minmax_transform(test_data_uni)

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
            )
        )

    def test_forecast_multi(self):
        logger.info("Training model...")
        yhat, _ = self.model.train(self.train_data_norm)
        name = self.model.target_name

        self.assertAlmostEqual(yhat.univariates[name].np_values.mean(), 0.50, 1)
        self.assertEqual(len(self.model._forecast), self.max_forecast_steps)
        self.assertAlmostEqual(self.model._forecast.mean(), 0.50, 1)
        testing_data_gen = gen_next_seq_label_pairs(self.test_data_norm, self.i, self.maxlags, self.max_forecast_steps)
        testing_instance, testing_label = next(testing_data_gen)
        pred, _ = self.model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(pred), self.max_forecast_steps)
        pred = pred.univariates[name].np_values
        self.assertAlmostEqual(pred.mean(), 0.50, 1)

        # save and load
        self.model.save(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_model = LGBMForecaster.load(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        new_pred, _ = loaded_model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(new_pred), self.max_forecast_steps)
        new_pred = new_pred.univariates[name].np_values
        self.assertAlmostEqual(pred.mean(), new_pred.mean(), 5)

    def test_forecast_uni(self):
        logger.info("Training model...")
        self.model.config.prediction_stride = 2
        yhat, _ = self.model.train(self.train_data_uni_norm)
        name = self.model.target_name

        self.assertAlmostEqual(yhat.univariates[name].np_values.mean(), 0.50, 1)
        self.assertEqual(len(self.model._forecast), self.max_forecast_steps)
        self.assertAlmostEqual(self.model._forecast.mean(), 0.50, 1)
        testing_data_gen = gen_next_seq_label_pairs(
            self.test_data_uni_norm, self.i, self.maxlags, self.max_forecast_steps
        )
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
