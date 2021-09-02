#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import pickle
import sys
import unittest

from merlion.utils import TimeSeries
from merlion.transform.bound import LowerUpperClip
from merlion.transform.moving_average import DifferenceTransform, ExponentialMovingAverage, LagTransform, MovingAverage
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.resample import TemporalResample, Shingle
from merlion.transform.sequence import TransformSequence, TransformStack


logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestInverse(unittest.TestCase):
    """Tests a number of transforms & their inverses."""

    def test_full(self):
        with open(join(rootdir, "data", "test_transform.pkl"), "rb") as f:
            df = pickle.load(f).drop(columns=["anomaly", "trainval"])

        ts = TimeSeries.from_pd(df)
        transform = TransformSequence(
            [
                MinMaxNormalize(),
                LowerUpperClip(0, 1),
                TemporalResample(),
                DifferenceTransform(),
                MovingAverage(weights=[0.1, 0.2, 0.3, 0.4]),
                LagTransform(k=20, pad=True),
                LagTransform(k=3, pad=False),
                TransformStack(
                    [ExponentialMovingAverage(alpha=0.7), MovingAverage(weights=[0.1, 0.2, 0.3, 0.4])],
                    check_aligned=False,
                ),
                Shingle(size=10, stride=7),
            ]
        )
        transform.train(ts)
        ts1 = transform(ts)
        ts2 = transform.invert(ts1, retain_inversion_state=True)
        df, df2 = ts.to_pd(), ts2.to_pd()
        rae = ((df - df2).abs() / ((df - df.mean()).abs() + 1e-8)).mean().mean()
        self.assertLess(rae, 1e-6)

        df2_prime = transform.invert(ts1).to_pd()
        rae = ((df2_prime - df2) / ((df2 - df2.mean()).abs() + 1e-8)).mean().mean()
        self.assertLess(rae, 1e-6)

        with self.assertRaises(RuntimeError) as context:
            transform.invert(ts1)
        self.assertTrue("Inversion state not set" in str(context.exception))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
