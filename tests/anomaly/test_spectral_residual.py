#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import sys
import unittest
from os.path import join, dirname, abspath

import numpy as np

from merlion.models.anomaly.spectral_residual import SpectralResidual, SpectralResidualConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.time_series import ts_csv_load

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestSpectralResidual(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.csv_name = join(rootdir, "data", "example.csv")
        self.test_len = 32768
        self.data = ts_csv_load(self.csv_name)
        logger.info(f"Data looks like:\n{self.data[:5]}")
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]
        self.model = SpectralResidual(
            SpectralResidualConfig(
                local_wind_sz=21,
                estimated_points=5,
                predicting_points=5,
                target_seq_index=0,
                threshold=AggregateAlarms(alm_threshold=3.5, min_alm_in_window=1),
            )
        )
        print()
        logger.info("Training model...\n")
        self.model.train(self.vals_train)

    def test_score(self):
        # score function returns the raw anomaly scores
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
        scores = self.model.get_anomaly_score(self.vals_test)
        logger.info(f"Scores look like:\n{scores[:5]}")
        scores = scores.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")

        self.assertEqual(len(scores), len(self.model.transform(self.vals_test)))

    def test_alarm(self):
        # alarm function returns the post-rule processed anomaly scores
        print("-" * 80)
        logger.info("test_alarm\n" + "-" * 80 + "\n")
        alarms = self.model.get_anomaly_label(self.vals_test)
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info(f"Alarms look like:\n{alarms[:5]}")
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertLessEqual(n_alarms, 6)
        self.assertGreaterEqual(n_alarms, 1)

    def test_save_load(self):
        print("-" * 80)
        logger.info("test_save_load\n" + "-" * 80 + "\n")
        self.model.save(dirname=join(rootdir, "tmp", "spectral_residual"))
        loaded_model = SpectralResidual.load(dirname=join(rootdir, "tmp", "spectral_residual"))

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
