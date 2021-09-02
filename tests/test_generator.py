#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from os.path import abspath, dirname
import sys
import logging
import unittest

import numpy as np
from operator import mul
from math import exp, log, sin

from merlion.utils.ts_generator import TimeSeriesGenerator, GeneratorComposer, GeneratorConcatenator

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(abspath(__file__)))


class TestTimeSeriesGenerator(unittest.TestCase):
    def test_generator_sequence(self):
        logger.info("test_generator_sequence\n" + "-" * 80 + "\n")

        np.random.seed(1234)
        y_generated = GeneratorComposer(
            generators=[
                TimeSeriesGenerator(f=lambda x: x ** 1.3, n=3),
                TimeSeriesGenerator(f=lambda x: 4.5 / (1 + exp(-x)), scale=4.5, n=7),
                TimeSeriesGenerator(f=lambda x: sin(x) * sin(3 * x), n=11),
            ],
            n=20,
            x0=-7,
            step=1.5,
            per_generator_noise=False,
        ).generate(return_ts=False)

        np.random.seed(1234)
        x = np.arange(20) * 1.5 - 7
        y_expected = ((4.5 / (1.0 + np.exp(-np.sin(x) * np.sin(3 * x)))) ** 1.3 + np.random.normal(size=20)).tolist()

        self.assertSequenceEqual(y_expected, y_generated)

    def test_generator_series(self):
        logger.info("test_generator_series\n" + "-" * 80 + "\n")

        np.random.seed(1234)
        y_generated = GeneratorConcatenator(
            generators=[
                TimeSeriesGenerator(f=lambda x: x ** 2, n=3, x0=0),
                TimeSeriesGenerator(f=lambda x: exp(-(x % 5)), n=7, x0=10),
                TimeSeriesGenerator(f=lambda x: 4 * log(x), n=11, x0=-99),
            ],
            n=20,
            x0=-7,
            step=1.5,
            noise=np.random.uniform,
            distort=mul,
            string_outputs=False,
            per_generator_noise=False,
        ).generate(return_ts=False)

        np.random.seed(1234)
        x = np.arange(21) * 1.5 - 7
        y_expected = (
            np.hstack((x[:3] ** 2, np.exp(-(x[3:10] % 5)), np.log(x[10:21]) * 4)) * np.random.uniform(size=21)
        ).tolist()

        self.assertSequenceEqual(y_expected, y_generated)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
