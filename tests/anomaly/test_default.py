#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from abc import ABC
import logging
import os
from os.path import abspath, dirname, join
import pytest
import sys
import unittest

import torch
import random
import numpy as np
import pandas as pd

from merlion.models.defaults import DefaultDetector, DefaultDetectorConfig
from merlion.plot import plot_anoms_plotly
from merlion.post_process.threshold import AggregateAlarms, Threshold
from merlion.utils import TimeSeries
from ts_datasets.anomaly import *

rootdir = dirname(dirname(dirname(abspath(__file__))))
logger = logging.getLogger(__name__)


def set_random_seeds():
    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)


def get_train_test_splits(df: pd.DataFrame, metadata: pd.DataFrame, n: int) -> (pd.DataFrame, pd.DataFrame, np.ndarray):
    train_df = df[metadata.trainval]
    test_df = df[~metadata.trainval]
    test_labels = pd.DataFrame(metadata[~metadata.trainval].anomaly)
    return train_df.tail(n), test_df.head(n), test_labels[:n]


class Mixin(ABC):
    def test_score(self):
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
        self.run_init()

        logger.info("Training model...\n")
        train_ts = TimeSeries.from_pd(self.train_df)
        self.model.train(train_ts)

        test_ts = TimeSeries.from_pd(self.test_df)
        score_ts = self.model.get_anomaly_score(test_ts)
        scores = score_ts.to_pd().values.flatten()
        min_score, max_score, sum_score = min(scores), max(scores), sum(scores)

        logger.info(f"scores look like: {scores[:10]}")
        logger.info(f"min score = {min_score}")
        logger.info(f"max score = {max_score}")
        logger.info(f"sum score = {sum_score}")

    def test_save_load(self):
        print("-" * 80)
        logger.info("test_save_load\n" + "-" * 80 + "\n")
        self.run_init()

        logger.info("Training model...\n")
        train_ts = TimeSeries.from_pd(self.train_df)
        self.model.train(train_ts)

        multi = train_ts.dim > 1
        path = join(rootdir, "tmp", "default", "anom", "multi" if multi else "uni")
        self.model.save(dirname=path)
        loaded_model = DefaultDetector.load(dirname=path)

        test_ts = TimeSeries.from_pd(self.test_df)
        scores = self.model.get_anomaly_score(test_ts)
        scores_np = scores.to_pd().values.flatten()
        loaded_model_scores = loaded_model.get_anomaly_score(test_ts)
        loaded_model_scores = loaded_model_scores.to_pd().values.flatten()
        self.assertEqual(len(scores_np), len(loaded_model_scores))
        alarms = self.model.post_rule(scores)
        loaded_model_alarms = loaded_model.post_rule(scores)
        self.assertSequenceEqual(list(alarms), list(loaded_model_alarms))

    def test_plot(self):
        try:
            import plotly

            print("-" * 80)
            logger.info("test_plot\n" + "-" * 80 + "\n")
            self.run_init()

            logger.info("Training model...\n")
            train_ts = TimeSeries.from_pd(self.train_df)
            self.model.train(train_ts)

            multi = train_ts.dim > 1
            savedir = join(rootdir, "tmp", "default", "anom")
            os.makedirs(savedir, exist_ok=True)
            path = join(savedir, ("multi" if multi else "uni") + ".png")

            test_ts = TimeSeries.from_pd(self.test_df)
            fig = self.model.plot_anomaly_plotly(
                time_series=test_ts, time_series_prev=train_ts, plot_time_series_prev=True
            )
            plot_anoms_plotly(fig, TimeSeries.from_pd(self.test_labels))
            try:
                import kaleido

                fig.write_image(path, engine="kaleido")
            except ImportError:
                logger.info("kaleido not installed, not trying to save image")

        except ImportError:
            logger.info("plotly not installed, skipping test case")


class TestUnivariate(unittest.TestCase, Mixin):
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

    def run_init(self):
        set_random_seeds()
        self.model = DefaultDetector(
            DefaultDetectorConfig(granularity="1h", threshold=AggregateAlarms(alm_threshold=1.5))
        )

        # Time series with anomalies in both train split and test split
        df = pd.read_csv(join(rootdir, "data", "synthetic_anomaly", "horizontal_spike_anomaly.csv"))
        df.timestamp = pd.to_datetime(df.timestamp, unit="s")
        df = df.set_index("timestamp")

        # Get training & testing splits
        self.train_df = df.iloc[: -len(df) // 2, :1]
        self.test_df = df.iloc[-len(df) // 2 :, :1]
        self.test_labels = df.iloc[-len(df) // 2 :, -1:]


class TestMultivariate(unittest.TestCase, Mixin):
    def run_init(self):
        set_random_seeds()
        self.model = DefaultDetector(DefaultDetectorConfig(threshold=AggregateAlarms(alm_threshold=2)))
        self.dataset = MSL(rootdir=join(rootdir, "data", "smap"))
        df, metadata = self.dataset[0]
        self.train_df, self.test_df, self.test_labels = get_train_test_splits(df, metadata, 2000)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
