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

import numpy as np
import pandas as pd

from merlion.evaluate.anomaly import TSADEvaluator, TSADEvaluatorConfig, TSADMetric
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig
from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.ensemble.anomaly import DetectorEnsemble, DetectorEnsembleConfig, EnsembleTrainConfig
from merlion.models.ensemble.combine import ModelSelector, Mean
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestEvaluateAnomaly(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Time series with anomalies in both train split and test split
        df = pd.read_csv(join(rootdir, "data", "synthetic_anomaly", "horizontal_spike_anomaly.csv"))
        df.timestamp = pd.to_datetime(df.timestamp, unit="s")
        df = df.set_index("timestamp")

        # Get training split
        train = df[: -len(df) // 2]
        self.train_data = TimeSeries.from_pd(train.iloc[:, 0])
        self.train_labels = TimeSeries.from_pd(train.anomaly)

        # Get testing split
        test = df[-len(df) // 2 :]
        self.test_data = TimeSeries.from_pd(test.iloc[:, 0])
        self.test_labels = TimeSeries.from_pd(test.anomaly)

    def test_single_model(self):
        print("-" * 80)
        logger.info("test_single_model\n" + "-" * 80 + "\n")
        logger.info("Training model & detection threshold on training data...")
        model = IsolationForest(IsolationForestConfig(n_estimators=25, max_n_samples=None))

        evaluator = TSADEvaluator(model=model, config=TSADEvaluatorConfig(train_window=None, retrain_freq="7d"))
        _, alarms = evaluator.get_predict(
            train_vals=self.train_data,
            test_vals=self.test_data,
            post_process=True,
            train_kwargs={"anomaly_labels": self.train_labels},
        )

        # Determine the number of alarms raised
        n_alarms = np.sum(alarms.to_pd().values != 0)
        print()
        logger.info("# of alarms = " + str(n_alarms))

        # Evaluate anomaly detector's performance on test split
        f1 = evaluator.evaluate(ground_truth=self.test_labels, predict=alarms, metric=TSADMetric.F1)
        p = evaluator.evaluate(ground_truth=self.test_labels, predict=alarms, metric=TSADMetric.Precision)
        r = evaluator.evaluate(ground_truth=self.test_labels, predict=alarms, metric=TSADMetric.Recall)
        logger.info(f"F1={f1:.4f}, precision={p:.4f}, recall={r:.4f}")
        self.assertAlmostEqual(p, 5 / 19, places=4)  # precision should be 5/19
        self.assertAlmostEqual(r, 5 / 9, places=4)  # recall should be 5/9

    def test_ensemble_no_valid(self):
        print("-" * 80)
        logger.info("test_ensemble_no_valid\n" + "-" * 80 + "\n")
        model = self.run_ensemble(valid_frac=0.0, expected_precision=9 / 57, expected_recall=9 / 9)
        self.assertSequenceEqual(model.combiner.models_used, [False, True, False])

        # Do a quick test of saving/loading
        model.save("tmp/eval/anom_ensemble_no_valid", save_only_used_models=True)
        DetectorEnsemble.load("tmp/eval/anom_ensemble_no_valid")

    def test_ensemble_with_valid(self):
        print("-" * 80)
        logger.info("test_ensemble_with_valid\n" + "-" * 80 + "\n")
        model = self.run_ensemble(valid_frac=0.50, expected_precision=9 / 60, expected_recall=9 / 9)
        self.assertSequenceEqual(model.combiner.models_used, [False, True, False])

        # Do a quick test of saving/loading
        model.save("tmp/eval/anom_ensemble_with_valid", save_only_used_models=True)
        DetectorEnsemble.load("tmp/eval/anom_ensemble_with_valid")

    def run_ensemble(self, valid_frac, expected_precision, expected_recall):
        logger.info("Training model & detection threshold on training data...")

        # build windowed statistics detector with window size = 30min
        # and data resampled once every 5min
        model0 = WindStats(WindStatsConfig(wind_sz=60, transform=TemporalResample("5min")))

        # build windowed statistics detector with window size = 120min
        # and data resampled once every 30min
        model1 = WindStats(WindStatsConfig(wind_sz=120, transform=TemporalResample("30min")))

        # build windowed statistics detector with window size = 360min
        # and no data resampling
        model2 = WindStats(config=WindStatsConfig(wind_sz=360))

        # Build ensemble model
        model = DetectorEnsemble(
            models=[model0, model1, model2], config=DetectorEnsembleConfig(combiner=ModelSelector(metric=TSADMetric.F1))
        )
        train_config = EnsembleTrainConfig(valid_frac=valid_frac)
        train_kwargs = {"anomaly_labels": self.train_labels, "train_config": train_config}

        evaluator = TSADEvaluator(model=model, config=TSADEvaluatorConfig(train_window=None, retrain_freq="7d"))
        _, alarms = evaluator.get_predict(
            train_vals=self.train_data, test_vals=self.test_data, post_process=True, train_kwargs=train_kwargs
        )

        # Determine the number of alarms raised
        n_alarms = np.sum(alarms.to_pd().values != 0)
        print()
        logger.info("# of alarms = " + str(n_alarms))

        # Evaluate ensemble anomaly detector's performance on test split
        f1 = evaluator.evaluate(ground_truth=self.test_labels, predict=alarms, metric=TSADMetric.F1)
        p = evaluator.evaluate(ground_truth=self.test_labels, predict=alarms, metric=TSADMetric.Precision)
        r = evaluator.evaluate(ground_truth=self.test_labels, predict=alarms, metric=TSADMetric.Recall)
        logger.info(f"F1={f1:.4f}, precision={p:.4f}, recall={r:.4f}")
        self.assertAlmostEqual(p, expected_precision, places=4)
        self.assertAlmostEqual(r, expected_recall, places=4)
        return model


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
