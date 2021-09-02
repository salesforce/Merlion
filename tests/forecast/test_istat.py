#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging

from merlion.utils.istat import Mean, Variance, ExponentialMovingAverage, RecencyWeightedVariance
from os.path import abspath, dirname
import sys
import unittest
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TestIStat(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(1)
        self.data1 = np.arange(1000).tolist()
        self.data2 = np.random.uniform(-1e-8, 1e-8, 1000).tolist()
        self.data3 = np.random.uniform(-1e8, 1e8, 1000).tolist()

    def test_mean_variance(self):
        logger.info("test_mean_variance\n" + "-" * 80 + "\n")
        for data in self.data1, self.data2, self.data3:
            left, right = data[:300], data[300:]
            # mean
            imean = Mean()
            imean.add_batch(data)
            self.assertAlmostEqual(np.mean(data), imean.value, places=8)
            # mean with intial value
            init_imean = Mean(value=np.mean(left), n=len(left))
            init_imean.add_batch(right)
            self.assertAlmostEqual(np.mean(data), init_imean.value, places=8)
            # drop left values to get right mean
            imean.drop_batch(left)
            right_imean = Mean()
            right_imean.add_batch(right)
            self.assertAlmostEqual(right_imean.value, imean.value, places=8)

            # variance
            ivar = Variance()
            ivar.add_batch(data)
            self.assertAlmostEqual(np.var(data, ddof=1), ivar.value, places=-1)
            # variance with initial values
            init_ivar = Variance(ex_value=np.mean(left), ex2_value=np.mean(np.array(left) ** 2), n=len(left))
            init_ivar.add_batch(right)
            self.assertAlmostEqual(np.var(data, ddof=1), init_ivar.value, places=-1)
            # drop left values to get right var
            ivar.drop_batch(left)
            right_ivar = Variance()
            right_ivar.add_batch(right)
            self.assertAlmostEqual(right_ivar.sd, ivar.sd, places=6)

    def test_recency_weighted_stats(self):
        logger.info("test_recency_weighted_stats\n" + "-" * 80 + "\n")

        np.random.seed(1)
        normal_data = np.random.normal(loc=-716.89, scale=245.7, size=1000000)
        # ema normal data
        ema = ExponentialMovingAverage(recency_weight=1e-4)
        ema.add_batch(normal_data)
        self.assertAlmostEqual(-716.89, ema.value, places=-1)
        # rwv normal data
        rwv = RecencyWeightedVariance(recency_weight=1e-4)
        rwv.add_batch(normal_data)
        self.assertAlmostEqual(245.7, rwv.value ** 0.5, places=-1)

        for data in self.data1, self.data2, self.data3:
            left, right = data[:800], data[800:]
            for rw in np.arange(0.1, 1.1, 0.1):
                # ema
                iema = ExponentialMovingAverage(recency_weight=rw)
                iema.add_batch(left)
                expected_ema = pd.Series(left).ewm(alpha=rw).mean().iloc[-1]
                self.assertAlmostEqual(expected_ema, iema.value, places=5)
                # ema with initial value
                init_iema = ExponentialMovingAverage(recency_weight=rw, value=iema.value, n=iema.n)
                init_iema.add_batch(right)
                iema.add_batch(right)
                self.assertAlmostEqual(iema.value, init_iema.value, places=5)
                # rwv with initial value
                irwv = RecencyWeightedVariance(recency_weight=rw)
                irwv.add_batch(left)
                init_irwv = RecencyWeightedVariance(
                    recency_weight=rw, n=irwv.n, ex_value=irwv.ex.value, ex2_value=irwv.ex2.value
                )
                init_irwv.add_batch(right)
                irwv.add_batch(right)
                self.assertAlmostEqual(irwv.value, init_irwv.value, places=5)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
