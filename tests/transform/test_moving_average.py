#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import unittest

from merlion.utils.time_series import UnivariateTimeSeries
from merlion.transform.moving_average import (
    DifferenceTransform,
    LagTransform,
    MovingPercentile,
    ExponentialMovingAverage,
)
from merlion.utils.ts_generator import TimeSeriesGenerator


class TestMovingAverage(unittest.TestCase):
    def test_difference_transform(self):
        n = 8
        ts = UnivariateTimeSeries(range(n), range(n)).to_ts()
        diff = DifferenceTransform()

        transformed_ts = diff(ts)
        expected_ts = UnivariateTimeSeries(range(1, n), np.ones(n - 1)).to_ts()
        self.assertEqual(expected_ts, transformed_ts)

    def test_lag_transform(self):
        n = 8
        ts = UnivariateTimeSeries(range(n), range(n)).to_ts()

        for k in range(1, 9):
            lag = LagTransform(k)
            transformed_ts = lag(ts)
            expected_ts = UnivariateTimeSeries(range(k, n), np.repeat(k, n - k)).to_ts()
            self.assertEqual(expected_ts, transformed_ts)

        lag = LagTransform(k=3, pad=True)
        transformed_ts = lag(ts)
        expected_vals = list(range(3)) + [3] * (n - 3)
        expected_ts = UnivariateTimeSeries(range(n), expected_vals).to_ts()
        self.assertEqual(expected_ts, transformed_ts)

    def test_moving_percentile(self):
        n = 20
        ts = UnivariateTimeSeries(range(n), range(n)).to_ts()

        transformed_ts = MovingPercentile(n_steps=1, q=23)(ts)
        expected_ts = UnivariateTimeSeries(range(n), range(n)).to_ts()
        self.assertEqual(expected_ts, transformed_ts)

        transformed_ts = MovingPercentile(n_steps=4, q=100)(ts)
        expected_ts = UnivariateTimeSeries(range(n), range(n)).to_ts()
        self.assertEqual(expected_ts, transformed_ts)

        transformed_ts = MovingPercentile(n_steps=6, q=0)(ts)
        expected_ts = UnivariateTimeSeries(range(n), [0] * 6 + list(range(1, 14 + 1))).to_ts()
        self.assertEqual(expected_ts, transformed_ts)

        transformed_ts = MovingPercentile(n_steps=3, q=50)(ts)
        expected_ts = UnivariateTimeSeries(range(n), [0, 0.5] + list(range(1, 18 + 1))).to_ts()
        self.assertEqual(expected_ts, transformed_ts)

    def test_exponential_moving_average_ci(self):
        np.random.seed(12345)
        name = "metric"
        ts = TimeSeriesGenerator(f=lambda x: x, n=100, name=name).generate()
        ema = ExponentialMovingAverage(alpha=0.1, ci=True)(ts)
        y = ema.univariates[name]
        lb = ema.univariates[f"{name}_lb"]
        ub = ema.univariates[f"{name}_ub"]
        self.assertTrue(all(l <= x <= u for (l, x, u) in zip(lb.values, y.values, ub.values)))


if __name__ == "__main__":
    unittest.main()
