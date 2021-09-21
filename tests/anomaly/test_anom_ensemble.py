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

from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.ensemble.anomaly import DetectorEnsemble, DetectorEnsembleConfig
from merlion.models.ensemble.combine import Mean, Median
from merlion.models.factory import ModelFactory
from merlion.post_process.threshold import AggregateAlarms
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import ts_csv_load

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))
csv_name = join(rootdir, "data", "example.csv")


class TestMedianAnomEnsemble(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load the time series sequence [(t1,v1), (t2, v2),...]
        data = ts_csv_load(csv_name, n_vars=1)

        # split the sequence into train and test
        self.vals_train = data[:-32768]
        self.vals_test = data[-32768:]

        # build windowed statistics detector with window size = 30min
        # and data resampled once every 5min
        model0 = WindStats(WindStatsConfig(wind_sz=30, transform=TemporalResample("5min")))

        # build windowed statistics detector with window size = 120min
        # and data resampled once every 30min
        model1 = WindStats(WindStatsConfig(wind_sz=120, transform=TemporalResample("30min")))

        # build windowed statistics detector with window size = 15min
        # and no data resampling
        model2 = WindStats(config=WindStatsConfig(wind_sz=15))

        # build an ensemble
        config = DetectorEnsembleConfig(combiner=Median(abs_score=True))
        self.ensemble = DetectorEnsemble(models=[model0, model1, model2], config=config)

    def test_alarm(self):
        print("-" * 80)
        logger.info("TestMedianAnomEnsemble.test_alarm\n" + "-" * 80 + "\n")
        self.ensemble.train(self.vals_train)

        # generate alarms for the test sequence using the ensemble
        # this will return an aggregated alarms from all the models inside the ensemble
        scores = self.ensemble.get_anomaly_label(self.vals_test)
        logger.info("scores look like " + str(scores[:3]))
        scores = scores.to_pd().values.flatten()

        num_of_alerts = np.sum(scores != 0)
        logger.info("# of alerts: " + str(num_of_alerts))
        logger.info(f"max score  = {max(scores):.2f}")
        logger.info(f"min score  = {min(scores):.2f}")
        self.assertEqual(num_of_alerts, 6)

    def test_save_load(self):
        print("-" * 80)
        logger.info("TestMedianAnomEnsemble.test_save_load\n" + "-" * 80 + "\n")
        self.ensemble.train(self.vals_train)
        scores = self.ensemble.get_anomaly_label(self.vals_test)

        self.ensemble.save(join(rootdir, "tmp", "med_anom_ensemble"))
        ensemble = DetectorEnsemble.load(join(rootdir, "tmp", "med_anom_ensemble"))
        loaded_scores = ensemble.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(scores), list(loaded_scores))

        # serialize and deserialize
        obj = self.ensemble.to_bytes()
        ensemble = ModelFactory.load_bytes(obj)
        loaded_scores = ensemble.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(scores), list(loaded_scores))


class TestMeanAnomEnsemble(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load the time series sequence [(t1,v1), (t2, v2),...]
        data = ts_csv_load(csv_name, n_vars=1)

        # split the sequence into train and test
        self.vals_train = data[:-32768]
        self.vals_test = data[-32768:]

        # build windowed statistics detector with window size = 30min
        # and data resampled once every 5min
        model0 = WindStats(WindStatsConfig(wind_sz=30, transform=TemporalResample("5min"), enable_calibrator=True))

        # build windowed statistics detector with window size = 120min
        # and data resampled once every 30min
        model1 = WindStats(WindStatsConfig(wind_sz=120, transform=TemporalResample("30min"), enable_calibrator=True))

        # build windowed statistics detector with window size = 15min
        # and no data resampling
        model2 = WindStats(config=WindStatsConfig(wind_sz=15, enable_calibrator=True))

        # build an ensemble
        config = DetectorEnsembleConfig(combiner=Mean(abs_score=True))
        self.ensemble = DetectorEnsemble(models=[model0, model1, model2], config=config)
        self.ensemble.train(self.vals_train)

    def test_alarm(self):
        print("-" * 80)
        logger.info("TestMeanAnomEnsemble.test_alarm\n" + "-" * 80 + "\n")
        self.ensemble.train(self.vals_train)

        # generate alarms for the test sequence using the ensemble
        # this will return an aggregated alarms from all the models inside the ensemble
        scores = self.ensemble.get_anomaly_label(self.vals_test)
        logger.info("scores look like " + str(scores[:3]))
        scores = scores.to_pd().values.flatten()

        num_of_alerts = np.sum(scores != 0)
        logger.info("# of alerts: " + str(num_of_alerts))
        logger.info(f"max score  = {max(scores):.2f}")
        logger.info(f"min score  = {min(scores):.2f}")
        self.assertEqual(num_of_alerts, 7)

    def test_save_load(self):
        print("-" * 80)
        logger.info("TestMeanAnomEnsemble.test_save_load\n" + "-" * 80 + "\n")

        self.ensemble.train(self.vals_train)
        scores = self.ensemble.get_anomaly_label(self.vals_test)

        self.ensemble.save(join(rootdir, "tmp", "mean_anom_ensemble"))
        ensemble = DetectorEnsemble.load(join(rootdir, "tmp", "mean_anom_ensemble"))
        loaded_scores = ensemble.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(scores), list(loaded_scores))

        # serialize and deserialize
        obj = self.ensemble.to_bytes()
        ensemble = ModelFactory.load_bytes(obj)
        loaded_scores = ensemble.get_anomaly_label(self.vals_test)
        self.assertSequenceEqual(list(scores), list(loaded_scores))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
