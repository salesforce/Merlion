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

from merlion.transform.normalize import PowerTransform
from merlion.transform.resample import TemporalResample
from merlion.models.anomaly.forecast_based.prophet import ProphetDetector, ProphetDetectorConfig
from merlion.utils.time_series import ts_csv_load, TimeSeries

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(dirname(abspath(__file__)))))


class TestProphet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.csv_name = join(rootdir, "data", "example.csv")
        self.data = TemporalResample("15min")(ts_csv_load(self.csv_name, n_vars=1))
        logger.info(f"Data looks like:\n{self.data[:5]}")

        # Test Prophet with a log transform (Box-Cox with lmbda=0)
        self.test_len = math.ceil(len(self.data) / 5)
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]
        self.model = ProphetDetector(
            ProphetDetectorConfig(
                transform=PowerTransform(lmbda=0.0), max_forecast_steps=self.test_len, uncertainty_samples=1000
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
        train_vals = self.model.transform(self.vals_train)
        train_vals = train_vals.univariates[train_vals.names[0]]
        self.assertNotAlmostEqual(
            train_scores.values[-1],
            train_vals.values[-1],
            delta=np.log(100),
            msg="Expected output of train() to be a TimeSeries of anomaly "
            "scores, but this seems to be a forecast. Check inheritance "
            "order of this forecasting detector.",
        )

        # score function returns the raw anomaly scores
        scores = self.model.get_anomaly_score(self.vals_test)
        self.assertEqual(len(scores), len(self.vals_test))
        logger.info(f"Scores look like:\n{scores[:5]}")
        scores = scores.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")

        logger.info("Verifying that scores don't change much on re-evaluation...\n")
        scoresv2 = self.model.get_anomaly_score(self.vals_test, self.vals_train)
        scoresv2 = scoresv2.to_pd().values.flatten()
        self.assertAlmostEqual(np.max(np.abs(scores - scoresv2)), 0, delta=1e-4)

        # We test save/load AFTER our first prediction because we need the old
        # posterior samples for reproducibility
        logger.info("Verifying that scores don't change much after save/load...\n")
        self.model.save(dirname=join(rootdir, "tmp", "prophet"))
        loaded_model = ProphetDetector.load(dirname=join(rootdir, "tmp", "prophet"))
        scoresv3 = loaded_model.get_anomaly_score(self.vals_test)
        scoresv3 = scoresv3.to_pd().values.flatten()
        self.assertAlmostEqual(np.max(np.abs(scores - scoresv3)), 0, delta=1e-4)

        # alarm function returns the post-rule processed anomaly scores
        alarms = self.model.get_anomaly_label(self.vals_test)
        n_alarms = np.sum(alarms.to_pd().values != 0)
        logger.info(f"Alarms look like:\n{alarms[:5]}")
        logger.info(f"Number of alarms: {n_alarms}\n")
        self.assertLessEqual(n_alarms, 15)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
