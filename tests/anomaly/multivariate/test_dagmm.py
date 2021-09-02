#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
import logging
import unittest
import torch
import random
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join
from merlion.utils import TimeSeries
from ts_datasets.anomaly import *
from merlion.models.anomaly.dagmm import DAGMM

rootdir = dirname(dirname(dirname(dirname(abspath(__file__)))))
logger = logging.getLogger(__name__)


def set_random_seeds():
    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)


def get_train_test_splits(df: pd.DataFrame, metadata: pd.DataFrame, n: int) -> (pd.DataFrame, pd.DataFrame, np.ndarray):
    train_df = df[metadata.trainval]
    test_df = df[~metadata.trainval]
    test_labels = metadata[~metadata.trainval].anomaly.values
    return train_df.tail(n), test_df.head(n), test_labels[:n]


class TestDAGMM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_random_seeds()

        self.model = DAGMM(config=DAGMM.config_class(num_epochs=10))
        self.dataset = MSL(rootdir=join(rootdir, "data", "smap"))
        df, metadata = self.dataset[0]
        self.train_df, self.test_df, self.test_labels = get_train_test_splits(df, metadata, 500)

        logger.info("Training model...\n")
        train_ts = TimeSeries.from_pd(self.train_df)
        self.model.train(train_ts)

    def test_score(self):
        print("-" * 80)
        logger.info("test_score\n" + "-" * 80 + "\n")
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
        self.model.save(dirname=join(rootdir, "tmp", "dagmm"))
        loaded_model = DAGMM.load(dirname=join(rootdir, "tmp", "dagmm"))

        test_ts = TimeSeries.from_pd(self.test_df)
        scores = self.model.get_anomaly_score(test_ts)
        loaded_model_scores = loaded_model.get_anomaly_score(test_ts)
        self.assertSequenceEqual(list(scores), list(loaded_model_scores))

        alarms = self.model.get_anomaly_label(test_ts)
        loaded_model_alarms = loaded_model.get_anomaly_label(test_ts)
        self.assertSequenceEqual(list(alarms), list(loaded_model_alarms))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
