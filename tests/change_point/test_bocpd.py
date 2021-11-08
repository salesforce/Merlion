#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import os
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np
import pandas as pd

from merlion.models.anomaly.change_point.bocpd import BOCPD, BOCPDConfig, ChangeKind
from merlion.utils.time_series import TimeSeries

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestBOCPD(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(12345)

    def standard_check(self, bocpd: BOCPD, test_data: TimeSeries, n: int, name: str):
        # Evaluate trained BOCPD model on the test data & make sure it has perfect precision & recall
        scores = bocpd.get_anomaly_score(test_data)
        alarms = bocpd.get_anomaly_label(test_data).to_pd().iloc[:, 0].abs()
        n_alarms = (alarms != 0).sum()
        logger.info(f"# Alarms fired: {n_alarms}")
        logger.info(f"Alarms fired at:\n{alarms[alarms != 0]}")
        self.assertNotEqual(alarms.iloc[0], 0)
        self.assertNotEqual(alarms.iloc[n], 0)
        self.assertNotEqual(alarms.iloc[2 * n], 0)
        self.assertEqual(n_alarms, 3)

        # Make sure we get the same results after saving & loading
        bocpd.save(join(rootdir, "tmp", "bocpd", name))
        loaded = BOCPD.load(join(rootdir, "tmp", "bocpd", name))
        loaded_scores = loaded.get_anomaly_score(test_data)
        self.assertSequenceEqual(list(scores), list(loaded_scores))

    def test_level_shift(self):
        print()
        logger.info("test_level_shift\n" + "-" * 80 + "\n")

        # Create a multivariate time series with random level shifts & split it into train & test
        n, d = 300, 5
        mus = [500, -90, 5, -50, 3]
        Sigmas_basis = [np.random.randn(d, d) + np.eye(d) for _ in mus]
        vals = [np.ones((n, d)) * mu + np.random.randn(n, d) @ U for i, (mu, U) in enumerate(zip(mus, Sigmas_basis))]
        vals = np.concatenate([np.ones((1, d)) * mus[0], *vals], axis=0)
        ts = TimeSeries.from_pd(vals, freq="1min")
        train, test = ts.bisect(ts.time_stamps[2 * n])

        # Initialize & train BOCPD with automatic change kind detection.
        # Make sure we choose level shift & correctly detect the level shift in the training data
        # Also make sure that we can make predictions on training data.
        bocpd = BOCPD(BOCPDConfig(change_kind=ChangeKind.Auto, cp_prior=1e-2, lag=1, min_likelihood=1e-12))
        train_scores = bocpd.train(train + test[:n]).to_pd().iloc[:, 0].abs()
        self.assertEqual(bocpd.change_kind, ChangeKind.LevelShift)
        self.assertGreater(train_scores.iloc[n - 1], 2)

        self.standard_check(bocpd=bocpd, test_data=test, n=n, name="level_shift")

    def test_trend_change(self):
        print()
        logger.info("test_trend_change\n" + "-" * 80 + "\n")

        # Create a multivariate time series with some trend changes and split it into train & test
        n, d = 300, 4
        ms = np.array([[10, -8, 12, 50], [-10, 3, 0, 9], [-3, 2, -10, 0], [-2, -3, 5, -3], [6, -1, 1, 15]])
        bs = np.array([[0, 5, 2, 3], [0, 0, 0, 0], [10, -2, -9, 8], [-3, 66, 2, 0], [85, -9, 21, 3]])
        sigma_basis = [U / np.trace(U) + np.eye(d) for U in np.random.randn(len(ms), d, d)]
        t = np.arange(n * len(ms)).reshape(-1, 1)
        x = np.concatenate(
            [
                m * t[i * n : (i + 1) * n] + b + np.random.randn(n, d) @ U
                for i, (m, b, U) in enumerate(zip(ms, bs, sigma_basis))
            ]
        )
        x = np.concatenate((bs[0].reshape(1, -1), x))
        ts = TimeSeries.from_pd(x, freq="1min")
        train, test = ts.bisect(ts.time_stamps[2 * n])

        # Initialize & train BOCPD with automatic change kind detection.
        # Make sure we choose trend change & correctly detect the level shift in the training data
        bocpd = BOCPD(BOCPDConfig(change_kind=ChangeKind.Auto, cp_prior=1e-2, lag=1, min_likelihood=1e-12))
        train_scores = bocpd.train(train).to_pd().iloc[:, 0].abs()
        self.assertEqual(bocpd.change_kind, ChangeKind.TrendChange)
        self.assertGreater(train_scores.iloc[n - 1], 2)

        # Evaluate trained BOCPD model on the test data & make sure it has perfect precision & recall
        self.standard_check(bocpd=bocpd, test_data=test, n=n, name="trend_change")

    def test_vis(self):
        print()
        logger.info("test_vis\n" + "-" * 80 + "\n")
        for fname, change in [("horizontal_level_anomaly", "LevelShift"), ("seasonal_trend_anomaly", "TrendChange")]:
            df = pd.read_csv(join(rootdir, "data", "synthetic_anomaly", f"{fname}.csv"))
            df.index = pd.to_datetime(df["timestamp"], unit="s")
            ts = TimeSeries.from_pd(df.iloc[:, 1])
            model = BOCPD(BOCPDConfig(change_kind=change, cp_prior=1e-2, min_likelihood=1e-10))
            train, test = ts[:500], ts[500:5000]
            model.train(train)
            fig, ax = model.plot_anomaly(
                time_series=test,
                time_series_prev=train,
                plot_time_series_prev=True,
                plot_forecast=True,
                plot_forecast_uncertainty=True,
            )
            os.makedirs(join(rootdir, "tmp", "bocpd"), exist_ok=True)
            fig.savefig(join(rootdir, "tmp", "bocpd", f"{change}.png"))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
