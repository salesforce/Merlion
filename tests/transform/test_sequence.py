#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from merlion.transform.base import Identity
from merlion.transform.sequence import TransformSequence, TransformStack
import unittest

from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.transform.moving_average import LagTransform, MovingAverage


class TestSequence(unittest.TestCase):
    def test_transform_sequence(self):
        n = 25
        ts = TimeSeries([UnivariateTimeSeries(range(n), range(n))])

        f, g, h = Identity(), MovingAverage(n_steps=3), LagTransform(k=2)
        seq = TransformSequence([f, g, h])
        seq.train(ts)

        transformed_ts = seq(ts)
        expected_ts = h(g(f(ts)))
        self.assertEqual(expected_ts, transformed_ts)

    def test_transform_stack(self):
        n = 25
        ts = TimeSeries([UnivariateTimeSeries(range(n), range(n))])

        f, g, h = Identity(), MovingAverage(n_steps=3), LagTransform(k=2)
        stack = TransformStack([f, g, h])
        stack.train(ts)

        transformed_ts = stack(ts)
        expected_ts = TimeSeries.from_ts_list([f(ts), g(ts), h(ts)])
        self.assertEqual(expected_ts, transformed_ts)


if __name__ == "__main__":
    unittest.main()
