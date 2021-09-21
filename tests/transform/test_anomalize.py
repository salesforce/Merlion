#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname
import sys
import unittest

import numpy as np

from merlion.utils.ts_generator import TimeSeriesGenerator
from merlion.transform.anomalize import Shock, TrendChange

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestAnomalize(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Generating Data...\n")
        np.random.seed(111)
        self.ts = TimeSeriesGenerator(f=lambda x: x ** 1.6, n=200, name="metric").generate(return_ts=True)

    def test_shock(self):
        print("-" * 80)
        logger.info("test_shock\n" + "-" * 80 + "\n")

        # test anomalies are statistically deviant from preceding values
        shock = Shock(anom_prob=0.2, pos_prob=0.5, sd_range=(5, 5), anom_width_range=(1, 3))
        anom_ts = shock(self.ts)
        vals = anom_ts.univariates["metric"].values
        labs = anom_ts.univariates["anomaly"].values
        ems = self.ts.univariates["metric"].to_pd().ewm(alpha=shock.alpha, adjust=False).std(bias=True)

        for i, (x, is_anom, sd) in enumerate(zip(vals, labs, ems)):
            if is_anom == 1.0 and labs[i - 1] == 0.0:
                shift = np.abs(x - vals[i - 1])
                assert shift > 3 * sd

    def test_trend_change(self):
        print("-" * 80)
        logger.info("test_trend_change\n" + "-" * 80 + "\n")

        # test strictly positive trend changes
        trend = TrendChange(anom_prob=0.2, pos_prob=1.0, scale_range=(2, 3))
        anom_ts = trend(self.ts)
        self.assertTrue(all(self.ts.univariates["metric"].np_values <= anom_ts.univariates["metric"].np_values))

        # test strictly negative trend changes
        trend = TrendChange(anom_prob=0.2, pos_prob=0.0, scale_range=(2, 3))
        anom_ts = trend(self.ts)
        self.assertTrue(all(self.ts.univariates["metric"].np_values >= anom_ts.univariates["metric"].np_values))

    def test_natural_bounds(self):
        print("-" * 80)
        logger.info("test_natural_bounds\n" + "-" * 80 + "\n")

        # generate data
        np.random.seed(111)
        ts = TimeSeriesGenerator(f=np.sin, n=200, name="metric").generate(return_ts=True)

        shock = Shock(anom_prob=0.5, sd_range=(5, 5), natural_bounds=(-1, 1))
        anom_vals = shock(ts).univariates["metric"].values
        self.assertTrue(all(np.abs(anom_vals) <= 1))

    def test_anom_prob(self):
        print("-" * 80)
        logger.info("test_anom_prob\n" + "-" * 80 + "\n")

        # test no anoms when anom_prob is 0
        for anomaly in (Shock(anom_prob=0.0), TrendChange(anom_prob=0.0)):
            anom_ts = anomaly(self.ts)
            self.assertEqual(self.ts.univariates["metric"], anom_ts.univariates["metric"])
            self.assertTrue(all(0.0 == anom_ts.univariates["anomaly"].np_values))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
