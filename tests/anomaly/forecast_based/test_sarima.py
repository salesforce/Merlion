#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import math
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np

from merlion.models.anomaly.forecast_based.sarima import SarimaDetector, SarimaDetectorConfig
from merlion.utils.time_series import ts_csv_load, TimeSeries

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(dirname(abspath(__file__)))))


class TestSarima(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Re-sample to 1hr because default (1min) takes too long to train
        self.csv_name = join(rootdir, "data", "example.csv")
        self.data = ts_csv_load(self.csv_name, n_vars=1)
        logger.info(f"Data looks like:\n{self.data[:5]}")

        self.test_len = math.ceil(len(self.data) / 5)
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]
        self.model = SarimaDetector(
            SarimaDetectorConfig(
                order=(2, 0, 2), seasonal_order=(2, 0, 2, 24), max_forecast_steps=self.test_len  # daily seasonality
            )
        )

    def test_full(self):
        print("-" * 80)
        logger.info("test_full\n" + "-" * 80 + "\n")
        logger.info("Training model...\n")
        train_scores = self.model.train(self.vals_train)

        # Make sure the anomaly scores are actually anomaly scores and not a
        # forecast (can happen if multiple inheritance order is incorrect)
        self.assertIsInstance(
            train_scores,
            TimeSeries,
            msg="Expected output of train() to be a TimeSeries of anomaly "
            "scores, but this seems to be a forecast. Check inheritance "
            "order of this forecasting detector.",
        )
        train_scores = train_scores.univariates[train_scores.names[0]]
        train_vals = self.vals_train.univariates[self.vals_train.names[0]]
        self.assertNotAlmostEqual(
            train_scores.values[-1],
            train_vals.values[-1],
            delta=100,
            msg="Expected output of train() to be a TimeSeries of anomaly "
            "scores, but this seems to be a forecast. Check inheritance "
            "order of this forecasting detector.",
        )

        self.model.save(dirname=join(rootdir, "tmp", "sarima"))
        loaded_model = SarimaDetector.load(dirname=join(rootdir, "tmp", "sarima"))

        # score function returns the raw anomaly scores
        scores = self.model.get_anomaly_score(self.vals_test)
        self.assertEqual(len(scores), len(self.model.transform(self.vals_test)))
        logger.info(f"Scores look like:\n{scores[:5]}")
        scores = scores.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")

        scoresv2 = self.model.get_anomaly_score(self.vals_test, self.vals_train)
        scoresv2 = scoresv2.to_pd().values.flatten()

        loaded_model_scores = loaded_model.get_anomaly_score(self.vals_test)
        loaded_model_scores = loaded_model_scores.to_pd().values.flatten()
        self.assertAlmostEqual(np.max(np.abs(scores - scoresv2)), 0, delta=1e-4)
        self.assertAlmostEqual(np.max(np.abs(scores - loaded_model_scores)), 0, delta=1e-4)

        # alarm function returns the post-rule processed anomaly scores
        alarms = self.model.get_anomaly_label(self.vals_test)
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info(f"Alarms look like:\n{alarms[:5]}")
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertEqual(n_alarms, 2)
        loaded_model_alarms = loaded_model.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(alarms), list(loaded_model_alarms))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
