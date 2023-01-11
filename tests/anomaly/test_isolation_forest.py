#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np

from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.transform.moving_average import MovingAverage, ExponentialMovingAverage
from merlion.transform.resample import Shingle
from merlion.transform.sequence import TransformSequence
from merlion.utils.data_io import csv_to_time_series

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestIsolationForest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_name = join(rootdir, "data", "example.csv")
        self.test_len = 32768
        self.data = csv_to_time_series(self.csv_name, timestamp_unit="ms", data_cols=["kpi"])
        logger.info(f"Data looks like:\n{self.data[:5]}")
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]

        # You probably wouldn't use this transform in practice, but we use it
        # here to test ExponentialMovingAverage and MovingAverage on
        # multi-variate time series
        self.model = IsolationForest(
            IsolationForestConfig(
                transform=TransformSequence(
                    [
                        Shingle(size=5, stride=1),
                        ExponentialMovingAverage(alpha=0.9, normalize=True),
                        MovingAverage(weights=[0.1, 0.2, 0.3, 0.4]),
                    ]
                )
            )
        )
        print()
        logger.info("Training model...\n")
        self.model.train(self.vals_train, post_rule_train_config={"unsup_quantile": 0.999})

    def test_score(self):
        # score function returns the raw anomaly scores
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
        scores = self.model.get_anomaly_score(self.vals_test)
        logger.info(f"Scores look like:\n{scores[:5]}")
        scores = scores.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")

    def test_alarm(self):
        # alarm function returns the post-rule processed anomaly scores
        print("-" * 80)
        logger.info("test_alarm\n" + "-" * 80 + "\n")
        alarms = self.model.get_anomaly_label(self.vals_test)
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info(f"Alarms look like:\n{alarms[:5]}")
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertLess(n_alarms, 15)

    def test_save_load(self):
        print("-" * 80)
        logger.info("test_save_load\n" + "-" * 80 + "\n")
        self.model.save(dirname=join(rootdir, "tmp", "isf"))
        loaded_model = IsolationForest.load(dirname=join(rootdir, "tmp", "isf"))

        scores = self.model.get_anomaly_score(self.vals_test)
        loaded_model_scores = loaded_model.get_anomaly_score(self.vals_test)
        self.assertSequenceEqual(list(scores), list(loaded_model_scores))

        alarms = self.model.get_anomaly_label(self.vals_test)
        loaded_model_alarms = loaded_model.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(alarms), list(loaded_model_alarms))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
