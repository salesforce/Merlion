#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import pandas as pd
import numpy as np
import sys
import unittest

from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.transform.resample import Shingle, TemporalResample

logger = logging.getLogger(__name__)


class TestResample(unittest.TestCase):
    def _test_granularity(self, granularity, offset=pd.to_timedelta(0)):
        # 6:30am on the 3rd of every other month
        index = pd.date_range("1970-12-01", "2010-01-01", freq=granularity) + offset
        full_df = pd.DataFrame(range(len(index)), index=index, columns=["value"])

        # drop some indices to induce missing data, and then split into train/test
        df = full_df.drop(index[[3, 4, 5, 12, 15, 16, -3, -2]])
        train = TimeSeries.from_pd(df.iloc[:-5])
        test = TimeSeries.from_pd(df.iloc[-5:])

        # train the temporal resample & ensure it has the right granularity
        transform = TemporalResample()
        transform.train(train)
        granularity = TemporalResample(granularity=granularity).granularity
        self.assertEqual(transform.granularity, granularity)

        # Make sure the resampled values are correct
        resampled_test = transform(test).to_pd()["value"]
        full_test = full_df.loc[resampled_test.index[0] :, "value"]
        self.assertSequenceEqual(resampled_test.index.to_list(), full_test.index.to_list())
        self.assertSequenceEqual(resampled_test.values.tolist(), full_test.values.tolist())

    def test_fixed_granularity(self):
        self._test_granularity(granularity="1h")
        self._test_granularity(granularity="3h", offset=pd.Timedelta(hours=1, minutes=13))

    def test_two_month(self):
        logger.info("Testing start-of-month resampling...")
        self._test_granularity(granularity="2MS")
        logger.info("Testing start-of-month resampling with an offset...")
        self._test_granularity(granularity="2MS", offset=pd.Timedelta(days=3, hours=6, minutes=30))
        logger.info("Testing end-of-month resampling...")
        if sys.version_info[1] < 8:
            self._test_granularity(granularity="2M")
        else:
            self._test_granularity(granularity="2ME")
        logger.info("Testing end-of-month resampling...")
        if sys.version_info[1] < 8:
            self._test_granularity(granularity="2M", offset=-pd.Timedelta(days=7, hours=7))
        else:
            self._test_granularity(granularity="2ME", offset=-pd.Timedelta(days=7, hours=7))

    def test_yearly(self):
        logger.info("Testing start-of-year resampling...")
        self._test_granularity(granularity="12MS", offset=pd.to_timedelta(0))
        logger.info("Testing end-of-year resampling...")
        self._test_granularity(granularity="12M", offset=pd.to_timedelta(0))


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
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()