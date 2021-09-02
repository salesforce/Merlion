#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np
import pandas as pd
import torch

from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.deep_point_anomaly_detector import DeepPointAnomalyDetector, DeepPointAnomalyDetectorConfig
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.resample import Shingle, TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils import TimeSeries

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestDPAD(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Time series with anomalies in both train split and test split
        # We test training with labels here to also test the AdaptiveThreshold
        # post-rule
        torch.manual_seed(12345)
        df = pd.read_csv(join(rootdir, "data", "synthetic_anomaly", "horizontal_spike_anomaly.csv"))
        df.timestamp = pd.to_datetime(df.timestamp, unit="s")
        df = df.set_index("timestamp")

        # Get training split
        train = df[: -len(df) // 2]
        self.train_data = TimeSeries.from_pd(train.iloc[:, 0])
        self.train_labels = TimeSeries.from_pd(train.anomaly)

        # Get testing split
        test = df[-len(df) // 2 :]
        self.test_data = TimeSeries.from_pd(test.iloc[:, 0])
        self.test_labels = TimeSeries.from_pd(test.anomaly)

        self.model = DeepPointAnomalyDetector(
            DeepPointAnomalyDetectorConfig(
                transform=TransformSequence(
                    [TemporalResample("15min"), Shingle(size=3, stride=2), DifferenceTransform()]
                )
            )
        )

    def test_full(self):
        # score function returns the raw anomaly scores
        print("-" * 80)
        logger.info("test_full\n" + "-" * 80 + "\n")
        logger.info("Training model...\n")
        self.model.train(self.train_data, self.train_labels)

        # Scores
        print()
        scores = self.model.get_anomaly_score(self.test_data)
        logger.info(f"\nScores look like:\n{scores[:5]}")
        scores = scores.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")

        # Alarms
        alarms = self.model.get_anomaly_label(self.test_data)
        logger.info(f"Alarms look like:\n{alarms[:5]}")
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertLessEqual(n_alarms, 15)

        # Serialization/deserialization
        self.model.save(dirname=join(rootdir, "tmp", "dpad"))
        loaded_model = DeepPointAnomalyDetector.load(dirname=join(rootdir, "tmp", "dpad"))
        loaded_alarms = loaded_model.get_anomaly_label(self.test_data)
        n_loaded_alarms = sum(loaded_alarms.to_pd().values != 0)
        self.assertAlmostEqual(n_loaded_alarms, n_alarms, delta=1)

        # Evaluation
        f1 = TSADMetric.F1.value(predict=alarms, ground_truth=self.test_labels)
        p = TSADMetric.Precision.value(predict=alarms, ground_truth=self.test_labels)
        r = TSADMetric.Recall.value(predict=alarms, ground_truth=self.test_labels)
        logger.info(f"F1={f1:.4f}, Precision={p:.4f}, Recall={r:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
