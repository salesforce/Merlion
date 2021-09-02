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

from merlion.utils.time_series import ts_csv_load
from merlion.models.anomaly.zms import ZMS, ZMSConfig
from merlion.post_process.threshold import AggregateAlarms

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestZMS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_name = join(rootdir, "data", "example.csv")
        self.test_len = 32768
        self.data = ts_csv_load(self.csv_name, n_vars=1)
        logger.info(f"Data looks like: {self.data[:5]}")
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]
        self.model = ZMS(ZMSConfig(lag_inflation=1.0, enable_calibrator=True, threshold=AggregateAlarms(3.5)))
        print()

    def test_lag_inflation(self):
        print("-" * 80)
        logger.info("test_lag_inflation\n" + "-" * 80 + "\n")
        # model without lag inflation
        logger.info("Training model...\n")
        self.model.train(self.vals_train)
        model = ZMS(ZMSConfig(lag_inflation=0.0))
        model.train(self.vals_train)
        scores1, scores2 = [m.get_anomaly_score(self.vals_test).to_pd().values for m in (model, self.model)]
        self.assertNotEqual(list(scores1), list(scores2))

    def test_score(self):
        # score function returns the raw anomaly scores
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
        logger.info("Training model...\n")
        self.model.train(self.vals_train)
        scores = self.model.get_anomaly_score(self.vals_test)
        logger.info(f"Scores look like: {scores[:5]}")
        scores = scores.to_pd().values
        logger.info("max score = " + str(scores.max()))
        logger.info("min score = " + str(scores.min()) + "\n")
        self.assertEqual(len(scores), len(self.model.transform(self.vals_test)))

    def test_alarm(self):
        # alarm function returns the post-rule processed anomaly scores
        print("-" * 80)
        logger.info("test_alarm\n" + "-" * 80 + "\n")
        logger.info("Training model...\n")
        self.model.train(self.vals_train)
        alarms = self.model.get_anomaly_label(self.vals_test)
        n_alarms = (alarms.to_pd().values != 0).sum()
        logger.info(f"Alarms look like: {alarms[:5]}")
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertEqual(n_alarms, 6)

    def test_save_load(self):
        print("-" * 80)
        logger.info("test_save_load\n" + "-" * 80 + "\n")
        logger.info("Training model...\n")
        self.model.train(self.vals_train)
        self.model.save(dirname=join(rootdir, "tmp", "zms"))
        loaded_model = ZMS.load(dirname=join(rootdir, "tmp", "zms"))

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
