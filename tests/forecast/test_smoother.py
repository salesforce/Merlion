#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import math
from merlion.models.factory import ModelFactory
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np
from numpy.core.fromnumeric import mean

from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import UnivariateTimeSeries, ts_csv_load
from merlion.models.forecast.smoother import MSES, MSESConfig, MSESTrainConfig

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestMSES(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        csv_name = join(rootdir, "data", "example.csv")
        self.data = TemporalResample("1h")(ts_csv_load(csv_name, n_vars=1))
        logger.info(f"Data looks like: {self.data[:5]}")

        n = math.ceil(len(self.data) / 5)
        self.train_data = self.data[:-n]
        self.test_data = self.data[-n:]
        self.test_times = self.test_data.time_stamps

        self.model = MSES(
            MSESConfig(
                max_forecast_steps=50,
                max_backstep=10,
                recency_weight=0.8,
                accel_weight=0.9,
                optimize_acc=False,
                eta=0.0,
                rho=1.0,
                inflation=1.1,
            )
        )
        self.model.train(self.train_data)

    def test_forecast_uncertainty(self):
        print("-" * 80)
        logger.info("test_forecast_uncertainty\n" + "-" * 80 + "\n")

        xhat, lb, ub = self.model.forecast(self.test_times[:50], return_iqr=True)
        xhat, lb, ub = [v.univariates[v.names[0]] for v in (xhat, lb, ub)]
        self.assertTrue(all(l <= x <= u for (l, x, u) in zip(lb.values, xhat.values, ub.values)))

    def test_x_squared(self):
        print("-" * 80)
        logger.info("test_x_squared\n" + "-" * 80 + "\n")

        N, n, s = 50, 20, 3
        data = UnivariateTimeSeries(range(N), np.arange(N) ** 2).to_ts()
        model = MSES(
            MSESConfig(max_forecast_steps=5, max_backstep=0, recency_weight=1.0, rho=1.0, eta=0.0, optimize_acc=False)
        )
        model.train(data[:n])

        # test forecast with timestamps
        xtrue = data[n + 1 : n + 5].univariates[data.names[0]].values
        xhat, _ = model.forecast(data[n + 1 : n + 5].univariates[data.names[0]].time_stamps)
        xhat = xhat.univariates[xhat.names[0]].values
        self.assertSequenceEqual(xtrue, xhat)

        # test forecast with n_steps
        xtrue = data[n : n + 5].univariates[data.names[0]].values
        xhat, _ = model.forecast(5)
        xhat = xhat.univariates[xhat.names[0]].values
        self.assertSequenceEqual(xtrue, xhat)

        # test forecast after update
        xhat = []
        for i in range(n, N, s):
            # forecast
            t = data[i : i + s].univariates[data.names[0]].time_stamps
            xhat += model.forecast(t)[0].univariates[model.target_name].values
            # update
            model.update(data[i : i + s])
        xtrue = data[n:].univariates[data.names[0]].values
        self.assertSequenceEqual(xtrue, xhat)

    def test_full(self):
        print("-" * 80)
        logger.info("test_full\n" + "-" * 80 + "\n")

        config = MSESConfig(
            max_forecast_steps=50,
            max_backstep=10,
            recency_weight=0.8,
            accel_weight=0.9,
            optimize_acc=False,
            eta=0.0,
            rho=1.0,
            inflation=1.1,
        )
        model, imodel = MSES(config), MSES(config)

        # test same forecasts after updates
        model.train(self.train_data)
        imodel.train(self.train_data[:-100])
        xhat = model.forecast(self.test_times[:50])[0].univariates[model.target_name]
        ixhat = imodel.forecast(self.test_times[:50], self.train_data[-100:])[0].univariates[imodel.target_name]
        self.assertTrue(all(np.isfinite(xhat.np_values)))
        self.assertSequenceEqual(xhat.values, ixhat.values)

        # reset
        config.eta = 0.01
        model, imodel = MSES(config), MSES(config)

        # test same forecasts with/out incremental initial training
        model.train(self.train_data, train_config=MSESTrainConfig(incremental=False))
        imodel.train(self.train_data, train_config=MSESTrainConfig(incremental=True, process_losses=False))
        xhat = model.forecast(self.test_times[:50])[0].univariates[model.target_name]
        ixhat = imodel.forecast(self.test_times[:50])[0].univariates[imodel.target_name]
        self.assertTrue(all(np.isfinite(xhat.np_values)))
        self.assertSequenceEqual(xhat.values, ixhat.values)

        # save and load
        model.save(dirname=join(rootdir, "tmp", "mses"))
        loaded_model = MSES.load(dirname=join(rootdir, "tmp", "mses"))

        # test save load same forecasts
        xhat, lm_xhat = [
            m.forecast(self.test_times[:50])[0].univariates[m.target_name].values for m in (model, loaded_model)
        ]
        self.assertTrue(all(np.isfinite(xhat)))
        self.assertSequenceEqual(xhat, lm_xhat)

        # test same save load forecasts after update
        xhat, lm_xhat = [
            m.forecast(self.test_times[100:130], self.test_data[:100])[0].univariates[m.target_name]
            for m in (model, loaded_model)
        ]
        self.assertTrue(all(np.isfinite(xhat.np_values)))
        self.assertSequenceEqual(xhat.values, lm_xhat.values)

        # serialize and deserialize
        obj = model.to_bytes()
        loaded_model = ModelFactory.load_bytes(obj)

        # test model and deserialized model same forecasts after update
        xhat, lm_xhat = [
            m.forecast(self.test_times[150:180], self.test_data[:150])[0].univariates[m.target_name]
            for m in (model, loaded_model)
        ]
        self.assertTrue(all(np.isfinite(xhat)))
        self.assertSequenceEqual(xhat.values, lm_xhat.values)

    def test_limited_data(self):
        print("-" * 80)
        logger.info("test_limited_data\n" + "-" * 80 + "\n")
        np.random.seed(1234)
        x = np.random.randint(low=-5, high=6, size=10)  # 9 random numbers in [-5, 6]
        times = list(range(len(x)))
        data = UnivariateTimeSeries(times, x).to_ts()
        model = MSES(MSESConfig(max_forecast_steps=5, max_backstep=5, recency_weight=1.0, rho=0.5))

        model.train(data[:3], MSESTrainConfig(incremental=True))
        xhat = model.forecast(times[3 : 3 + 5])[0].univariates[model.target_name].values

        # xhat_b = x+v+a for b=0; x+v for b=1; x for b=2; None for b=3,4,5
        exhat_0 = (x[2] + (x[2] - x[1]) + ((x[2] - x[1]) - (x[1] - x[0])) + x[1] + (x[2] - x[0])) / 2
        self.assertAlmostEqual(exhat_0, xhat[0])

        # xhat_b = x+v for b=0; x for b=1,2; None for b=3,4,5
        exhat_1 = x[2] + (x[2] - x[0])
        self.assertAlmostEqual(exhat_1, xhat[1])

        # xhat_b = x for b=0,1,2; None for b=3,4,5
        exhat_4 = mean(x[:3])
        self.assertAlmostEqual(exhat_4, xhat[4])

        # insufficient data for all relevant scales
        with self.assertRaises(AssertionError) as context:
            # ask model to forecast 6 steps in advance
            model.forecast([times[3 + 5]])
            self.assertTrue("Expected `time_stamps` to be between" in str(context.exception))

        # recency weight may only be tuned at viable scales
        model = MSES(MSESConfig(max_forecast_steps=5, max_backstep=5, recency_weight=0.7, eta=0.2, optimize_acc=False))
        model.train(data[:3])
        model.update(data[3:6])
        model.update(data[6:])

        w0 = model.config.recency_weight
        for scale, stat in model.delta_estimator.stats.items():
            vw, sw, aw = [s.recency_weight for s in (stat.velocity, stat.vel_var, stat.acceleration)]
            if scale <= 2:
                self.assertNotAlmostEqual(w0, vw)
                self.assertNotAlmostEqual(w0, sw)
                self.assertNotAlmostEqual(w0, aw)
            elif 2 < scale <= 5:
                self.assertNotAlmostEqual(w0, vw)
                self.assertNotAlmostEqual(w0, sw)
                self.assertEqual(w0, aw)
            else:
                self.assertEqual(w0, vw)
                self.assertEqual(w0, sw)
                self.assertEqual(w0, aw)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
