#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import datetime
import logging
import math
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np

from merlion.transform.resample import TemporalResample
from merlion.models.anomaly.forecast_based.lstm import LSTMDetector, LSTMTrainConfig, LSTMDetectorConfig
from merlion.models.forecast.lstm import auto_stride
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.time_series import ts_csv_load, TimeSeries

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(dirname(abspath(__file__)))))


class TestLSTM(unittest.TestCase):
    def test_full(self):
        file_name = join(rootdir, "data", "example.csv")

        sequence = TemporalResample("15min")(ts_csv_load(file_name, n_vars=1))
        logger.info(f"Data looks like:\n{sequence[:5]}")

        time_stamps = sequence.univariates[sequence.names[0]].time_stamps
        stride = auto_stride(time_stamps, resolution=12)
        logger.info("stride = " + str(stride))

        # 2 days of data for testing
        test_delta = datetime.timedelta(days=2).total_seconds()
        ts_train, ts_test = sequence.bisect(time_stamps[-1] - test_delta)
        forecast_steps = math.ceil(len(ts_test) / stride)

        self.assertGreater(forecast_steps, 1, "sequence is not long enough")

        model = LSTMDetector(
            LSTMDetectorConfig(max_forecast_steps=forecast_steps, nhid=256, threshold=AggregateAlarms(2, 1, 60, 300))
        )
        train_config = LSTMTrainConfig(
            data_stride=stride,
            epochs=1,
            seq_len=forecast_steps * 2,
            checkpoint_file=join(rootdir, "tmp", "lstm", "checkpoint.pt"),
        )
        train_scores = model.train(train_data=ts_train, train_config=train_config)

        self.assertIsInstance(
            train_scores,
            TimeSeries,
            msg="Expected output of train() to be a TimeSeries of anomaly "
            "scores, but this seems to be a forecast. Check inheritance "
            "order of this forecasting detector.",
        )
        train_scores = train_scores.univariates[train_scores.names[0]]
        train_vals = ts_train.univariates[ts_train.names[0]]
        self.assertNotAlmostEqual(
            train_scores.values[-1],
            train_vals.values[-1],
            delta=100,
            msg="Expected output of train() to be a TimeSeries of anomaly "
            "scores, but this seems to be a forecast. Check inheritance "
            "order of this forecasting detector.",
        )

        ##############
        scores = model.get_anomaly_score(ts_test)
        logger.info("Scores look like:\n" + str(scores[:5]))
        alarms = model.get_anomaly_label(ts_test)
        logger.info("Alarms look like:\n" + str(alarms[:5]))
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info("# of alarms = " + str(n_alarms))
        self.assertLess(n_alarms, 20)

        ##############
        # Note: we compare scores vs scoresv2[1:] because scoresv2 has one
        # extra time step included. This is because when `time_series_prev` is
        # given, we compute `self.model.transform(ts_train + ts_test)` and take
        # the first time step in the transformed FULL time series which matches
        # with `ts_test`. This is different from the first time step of
        # `self.model.transform(ts_test)` due to the difference transform.
        scoresv2 = model.get_anomaly_score(ts_test, ts_train)[1:]
        self.assertSequenceEqual(list(scores), list(scoresv2))

        ##############
        model.save(join(rootdir, "tmp", "lstm"))
        model = LSTMDetector.load(join(rootdir, "tmp", "lstm"))
        loaded_scores = model.get_anomaly_score(ts_test)
        self.assertSequenceEqual(list(scores), list(loaded_scores))
        loaded_alarms = model.get_anomaly_label(ts_test)
        self.assertSequenceEqual(list(alarms), list(loaded_alarms))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
