#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import math
from os.path import abspath, dirname, join
import pytest
import sys
import unittest

import numpy as np

from merlion.models.anomaly.random_cut_forest import RandomCutForest, RandomCutForestConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.normalize import MeanVarNormalize
from merlion.transform.resample import Shingle, TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils.time_series import ts_csv_load

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestRandomCutForest(unittest.TestCase):
    def run_init(self):
        # Resample @ 5min granularity b/c default (1min) takes too long to train
        self.csv_name = join(rootdir, "data", "example.csv")
        self.data = ts_csv_load(self.csv_name, n_vars=1)
        self.test_len = math.ceil(len(self.data) / 5)

        logger.info(f"Data looks like:\n{self.data[:5]}")
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]

        transform = TransformSequence(
            [
                TemporalResample("15min", trainable_granularity=False),
                Shingle(size=10, stride=1),
                MeanVarNormalize(),
                DifferenceTransform(),
            ]
        )
        self.model = RandomCutForest(
            RandomCutForestConfig(
                transform=transform,
                n_estimators=50,
                max_n_samples=256,
                seed=0,
                threshold=AggregateAlarms(alm_threshold=3.0),
            )
        )
        print()
        logger.info("Training model...\n")
        self.model.train(self.vals_train)

    @pytest.fixture(autouse=True)
    def fixture(self):
        # Necessary to avoid jpype-induced segfault due to running JVM in a thread when
        # running this test with pytest. See the docs here:
        # https://jpype.readthedocs.io/en/latest/userguide.html#errors-reported-by-python-fault-handler
        try:
            import faulthandler

            faulthandler.enable()
            faulthandler.disable()
        except:
            pass

    def test_score(self):
        self.run_init()
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
        scores = self.model.get_anomaly_score(self.vals_train[-20:] + self.vals_test)
        logger.info(f"Scores look like:\n{scores[:5]}")
        scores = scores.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")

    def test_alarm(self):
        self.run_init()
        print("-" * 80)
        logger.info("test_alarm\n" + "-" * 80 + "\n")
        alarms = self.model.get_anomaly_label(self.vals_test)
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info(f"Alarms look like:\n{alarms[:5]}")
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertLess(n_alarms, 5)

    def test_save_load(self):
        self.run_init()
        print("-" * 80)
        logger.info("test_save_load\n" + "-" * 80 + "\n")
        scores = self.model.get_anomaly_score(self.vals_test)
        scores = scores.to_pd().values.flatten()

        self.model.save(dirname=join(rootdir, "tmp", "rcf"))
        loaded_model = RandomCutForest.load(dirname=join(rootdir, "tmp", "rcf"))
        loaded_model_scores = loaded_model.get_anomaly_score(self.vals_test)
        loaded_model_scores = loaded_model_scores.to_pd().values.flatten()

        self.assertEqual(len(scores), len(loaded_model_scores))
        max_diff = float(np.max(np.abs(scores - loaded_model_scores)))
        self.assertAlmostEqual(max_diff, 0.0, delta=1)

    def test_online_updates(self):
        self.run_init()
        print("-" * 80)
        logger.info("test_online_updates\n" + "-" * 80 + "\n")
        self.model.config.online_updates = True
        alarms = self.model.get_anomaly_label(self.vals_train[-20:] + self.vals_test)
        n_alarms = np.sum(alarms.to_pd().values != 0)
        self.assertLess(n_alarms, 5)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
