#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import numpy as np
import unittest

from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.transform.resample import Shingle


class TestShingle(unittest.TestCase):
    def test_shingle(self):
        n = 8
        ts = TimeSeries([UnivariateTimeSeries(range(n), range(n))])

        shingle = Shingle(size=1, stride=1)
        transformed_ts = shingle(ts)
        times = range(n)
        expected_ts = TimeSeries([UnivariateTimeSeries(times, range(n))])
        self.assertEqual(expected_ts, transformed_ts)

        shingle = Shingle(size=1, stride=1)
        transformed_ts = shingle(ts)
        self.assertEqual(expected_ts, transformed_ts)

        shingle = Shingle(size=3, stride=1)
        transformed_ts = shingle(ts)
        expected_ts = TimeSeries(
            [UnivariateTimeSeries(times, np.append(np.repeat(0, k), range(n - k))) for k in reversed(range(3))]
        )
        self.assertEqual(expected_ts, transformed_ts)

        shingle = Shingle(size=3, stride=2)
        transformed_ts = shingle(ts)
        times = [1, 3, 5, 7]
        expected_ts = TimeSeries(
            [
                UnivariateTimeSeries(times, [0, 1, 3, 5]),
                UnivariateTimeSeries(times, [0, 2, 4, 6]),
                UnivariateTimeSeries(times, [1, 3, 5, 7]),
            ]
        )
        self.assertEqual(expected_ts, transformed_ts)

        shingle = Shingle(size=3, stride=3)
        transformed_ts = shingle(ts)
        times = [1, 4, 7]
        expected_ts = TimeSeries(
            [
                UnivariateTimeSeries(times, [0, 2, 5]),
                UnivariateTimeSeries(times, [0, 3, 6]),
                UnivariateTimeSeries(times, [1, 4, 7]),
            ]
        )
        self.assertEqual(expected_ts, transformed_ts)


if __name__ == "__main__":
    unittest.main()
