#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from functools import reduce
import logging
import os
from os.path import abspath, dirname, join
import sys
import unittest

import numpy as np
import pandas as pd

from merlion.models.anomaly.dbl import DynamicBaseline, DynamicBaselineConfig, Trend
from merlion.utils.ts_generator import GeneratorConcatenator, TimeSeriesGenerator
from merlion.utils.resample import to_pd_datetime

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


class TestDynamicBaseline(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        np.random.seed(1234)
        self.data = TimeSeriesGenerator(
            f=lambda x: np.sin(x * 0.2) + 0.5 * np.sin(x * 0.05), n=30000, x0=2, step=1.5, tdelta="55min"
        ).generate()
        logger.info(f"Data looks like:\n{self.data[:5]}")

        self.test_len = 4500
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]

        self.model = DynamicBaseline(
            DynamicBaselineConfig(train_window="15d", wind_sz="1h", trends=["daily", "weekly"])
        )

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
        scores = alarms.to_pd().values.flatten()
        logger.info("max score = " + str(max(scores)))
        logger.info("min score = " + str(min(scores)) + "\n")
        self.assertLessEqual(n_alarms, 53)

    def test_save_load(self):
        print("-" * 80)
        logger.info("test_save_load\n" + "-" * 80 + "\n")
        self.model.save(dirname=join(rootdir, "tmp", "dbl"))
        loaded_model = DynamicBaseline.load(dirname=join(rootdir, "tmp", "dbl"))

        scores = self.model.get_anomaly_score(self.vals_test)
        loaded_model_scores = loaded_model.get_anomaly_score(self.vals_test)
        self.assertSequenceEqual(list(scores), list(loaded_model_scores))

        alarms = self.model.get_anomaly_label(self.vals_test)
        loaded_model_alarms = loaded_model.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(alarms), list(loaded_model_alarms))

    def test_online_updates(self):
        print("-" * 80)
        logger.info("test_online_updates\n" + "-" * 80 + "\n")

        # create fixed period corresponding to last 9 days
        scope = "9d"
        tf = to_pd_datetime(self.vals_train.tf)
        fixed_period = tuple(str(t) for t in (tf - pd.Timedelta(scope), tf))

        kwargs = dict(wind_sz="27min", trends=["daily", "weekly"])
        rolling_config = DynamicBaselineConfig(train_window=scope, **kwargs)
        fixed_config = DynamicBaselineConfig(fixed_period=fixed_period, **kwargs)

        rolling_model, rolling_imodel = DynamicBaseline(rolling_config), DynamicBaseline(rolling_config)
        fixed_model, fixed_imodel = DynamicBaseline(fixed_config), DynamicBaseline(fixed_config)

        for model, imodel in ((fixed_model, fixed_imodel), (rolling_model, rolling_imodel)):
            # train model
            model.train(self.vals_train)
            # train & update model incrementally
            imodel.train(self.vals_train[:-121])
            imodel.update(self.vals_train[-125:-74])
            imodel.update(self.vals_train[-74:])
            # score
            scores, iscores = [m.get_anomaly_score(self.vals_test) for m in (model, imodel)]
            score_diffs = scores.squeeze().np_values - iscores.squeeze().np_values
            self.assertAlmostEqual(np.abs(score_diffs).max(), 0, delta=1e-3)

        # test rolling model against fixed model
        fixed_scores = fixed_model.get_anomaly_score(self.vals_test)
        rolling_scores = rolling_model.get_anomaly_score(self.vals_test)
        self.assertEqual(fixed_scores, rolling_scores)

    def test_common_trend_sets(self):
        print("-" * 80)
        logger.info("test_common_trend_sets\n" + "-" * 80 + "\n")

        trend_sets = [
            ["daily"],  # hour of day
            ["daily", "weekly"],  # hour of week
            ["daily", "monthly"],  # hour of month
        ]
        for trends in trend_sets:
            fixed_period = ("1970-07-22", "1971-09-27")
            fixed_model = DynamicBaseline(DynamicBaselineConfig(trends=trends, fixed_period=fixed_period))
            rolling_model = DynamicBaseline(DynamicBaselineConfig(trends=trends))

            for model in (rolling_model, fixed_model):
                # train model and compute scores
                model.train(self.vals_train)
                scores = model.get_anomaly_score(self.vals_test).squeeze().np_values

                # compute expected scores manually
                df = self.vals_train.to_pd()
                if model.has_fixed_period:
                    t0, tf = (pd.Timestamp(t) for t in fixed_period)
                else:
                    t0 = df.index[-1] - pd.Timedelta(model.config.train_window)
                    tf = df.index[-1]
                df = df[(df.index >= t0) & (df.index <= tf)]

                group_keys, keys = [], []
                if "daily" in trends:
                    group_keys += [df.index.hour]
                    keys += [lambda t: (t.hour,)]
                if "weekly" in trends:
                    group_keys += [df.index.dayofweek]
                    keys += [lambda t: (t.dayofweek,)]
                if "monthly" in trends:
                    group_keys += [df.index.day]
                    keys += [lambda t: (t.day,)]
                group = df.groupby(group_keys)
                mu = group[0].mean()
                sd = group[0].std()

                # determine key
                key = (
                    reduce(lambda key1, key2: lambda t: key1(t) + key2(t), keys, lambda t: tuple())
                    if len(keys) > 1
                    else lambda t: keys[0](t)[0]
                )

                expected_scores = np.asarray(
                    [(x - mu[key(t)]) / sd[key(t)] for t, x in self.vals_test.to_pd().iterrows()]
                ).flatten()
                score_diffs = scores - expected_scores
                self.assertAlmostEqual(np.abs(score_diffs).max(), 0, delta=1e-3)

    def test_no_trends(self):
        print("-" * 80)
        logger.info("test_no_trends\n" + "-" * 80 + "\n")

        # create fixed period corresponding to last 24 days
        scope = "24d"
        tf = to_pd_datetime(self.vals_train.tf)
        window = tuple(t.timestamp() for t in (tf - pd.Timedelta(scope), tf))

        kwargs = dict(wind_sz="11min", trends=[])
        rolling_model = DynamicBaseline(DynamicBaselineConfig(train_window=scope, **kwargs))
        fixed_model = DynamicBaseline(DynamicBaselineConfig(fixed_period=window, **kwargs))

        # get z-scores
        rel_vals = self.vals_train.window(t0=window[0], tf=window[1], include_tf=True).squeeze().np_values
        mu, sd = rel_vals.mean(), rel_vals.std()
        z_scores = (self.vals_test.squeeze().np_values - mu) / sd

        for model in (fixed_model, rolling_model):
            model.train(self.vals_train)
            scores = model.get_anomaly_score(self.vals_test).squeeze().np_values
            score_diffs = scores - z_scores
            self.assertAlmostEqual(np.abs(score_diffs).max(), 0, delta=1e-2)

    def test_limited_data(self):
        print("-" * 80)
        logger.info("test_limited_data\n" + "-" * 80 + "\n")
        # Tests that if a segment doesn't have any data in it, the z-score
        # returned for any value relative to this segment's baseline is 0.

        # generate 0, 1, 0, 1 ... for 2 days, 0.5, 0.5 ... afterward
        data = GeneratorConcatenator(
            generators=[
                TimeSeriesGenerator(f=lambda x: x % 2 == 0, n=2 * 24),
                TimeSeriesGenerator(f=lambda x: 0.5, n=6 * 24),
            ],
            noise=lambda: 0,
            n=8 * 24,
            string_outputs=False,
            tdelta="1h",
        ).generate()
        train_vals, test_vals = data[: 2 * 24], data[2 * 24 :]

        model = DynamicBaseline(DynamicBaselineConfig(train_window="7d", trends=["weekly"]))
        model.train(train_vals)

        scores = model.get_anomaly_score(test_vals).squeeze().values
        expected_scores = [0] * (6 * 24)
        self.assertEqual(expected_scores, scores)

    def test_baseline_plot(self):
        print("-" * 80)
        logger.info("test_baseline_plot\n" + "-" * 80 + "\n")

        # check upper and lower bounds and baselines
        figure = self.model.get_baseline_figure(self.vals_test)
        self.assertTrue((figure.yhat_iqr.univariates["lb"].np_values < figure.yhat.np_values).all())
        self.assertTrue((figure.yhat_iqr.univariates["ub"].np_values > figure.yhat.np_values).all())

        # test plot with matplotlib
        figdir = join(rootdir, "tmp", "dbl_plot")
        os.makedirs(figdir, exist_ok=True)
        figure.plot()[0].savefig(join(figdir, "dbl_plot.png"))

        # Test plotting with plotly
        try:
            import plotly

            fig = figure.plot_plotly()
            try:
                import kaleido

                fig.write_image((join(figdir, "dbl_plotly.png")), engine="kaleido")
            except ImportError:
                logger.info("kaleido not installed, not trying to save image")
        except ImportError:
            logger.info("plotly not installed, not trying plotly visualization")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
