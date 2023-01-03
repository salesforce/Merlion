#
# Copyright (c) 2022 salesforce.com, inc.
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

from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.models.ensemble.combine import ModelSelector, Mean
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.automl.autoprophet import AutoProphet, AutoProphetConfig
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.factory import ModelFactory
from merlion.transform.base import Identity
from merlion.transform.normalize import BoxCoxTransform
from merlion.transform.resample import TemporalResample
from merlion.utils.data_io import csv_to_time_series, TimeSeries

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestForecastEnsemble(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_name = join(rootdir, "data", "example.csv")
        self.test_len = 2048
        data = csv_to_time_series(self.csv_name, timestamp_unit="ms", data_cols=["kpi"])[::10]
        self.vals_train = data[: -self.test_len]
        self.vals_test = data[-self.test_len :].univariates[data.names[0]]

    def _test_mean(self, test_name):
        model0 = Arima(ArimaConfig(order=(6, 1, 2), max_forecast_steps=50, transform=TemporalResample("1h")))
        model1 = Arima(ArimaConfig(order=(24, 1, 0), max_forecast_steps=50, transform=TemporalResample("10min")))
        model2 = AutoProphet(
            config=AutoProphetConfig(transform=Identity(), periodicity_strategy="All", information_criterion="BIC")
        )
        self.ensemble = ForecasterEnsemble(
            models=[model0, model1, model2], config=ForecasterEnsembleConfig(combiner=Mean(abs_score=False))
        )

        self.ensemble.models[0].config.max_forecast_steps = None
        self.ensemble.models[1].config.max_forecast_steps = None
        self.run_test(test_name)

    def _test_selector(self, test_name, expected_smapes):
        model0 = Arima(ArimaConfig(order=(6, 1, 2), max_forecast_steps=50, transform=TemporalResample("1h")))
        model1 = Arima(ArimaConfig(order=(24, 1, 0), max_forecast_steps=50, transform=TemporalResample("10min")))
        model2 = AutoProphet(
            config=AutoProphetConfig(
                target_seq_index=0,
                transform=BoxCoxTransform(lmbda=0),
                periodicity_strategy="Max",
                information_criterion="AICc",
            )
        )
        self.ensemble = ForecasterEnsemble(
            config=ForecasterEnsembleConfig(
                models=[model0, model1, model2], combiner=ModelSelector(metric=ForecastMetric.sMAPE), target_seq_index=0
            )
        )
        self.run_test(test_name)
        # We expect the model selector to select Prophet because it gets the lowest validation sMAPE
        valid_smapes = np.asarray(self.ensemble.combiner.metric_values)
        self.assertAlmostEqual(np.nanmax(np.abs(valid_smapes - expected_smapes)), 0, delta=2)
        self.assertSequenceEqual(self.ensemble.models_used, [False, False, True])

    def test_mean(self):
        print("-" * 80)
        logger.info("test_mean\n" + "-" * 80 + "\n")
        self.expected_smape = 38
        self._test_mean(test_name="test_mean")

    def test_mean_small_train(self):
        print("-" * 80)
        logger.info("test_mean_small_train\n" + "-" * 80 + "\n")
        self.vals_train = self.vals_train[-8:]
        self.expected_smape = 164.5
        self._test_mean(test_name="test_mean_small_train")

    def test_univariate_selector(self):
        print("-" * 80)
        logger.info("test_univariate_selector\n" + "-" * 80 + "\n")
        self.expected_smape = 20
        self._test_selector(test_name="test_univariate_selector", expected_smapes=[34.66, 39.81, 21.46])

    def test_multivariate_selector(self):
        print("-" * 80)
        logger.info("test_multivariate_selector\n" + "-" * 80 + "\n")
        x = self.vals_train.to_pd()
        self.expected_smape = 20
        self.vals_train = TimeSeries.from_pd(
            pd.DataFrame(np.concatenate((x.values, x.values * 2), axis=1), columns=["A", "B"], index=x.index)
        )
        self._test_selector(test_name="test_multivariate_selector", expected_smapes=[34.66, 39.81, 21.46])

    def test_selector_small_train(self):
        print("-" * 80)
        logger.info("test_selector_small_train\n" + "-" * 80 + "\n")
        self.vals_train = self.vals_train[-8:]
        self.expected_smape = 194
        self._test_selector(test_name="test_selector_small_train", expected_smapes=[np.inf, 7.27, 6.16])

    def run_test(self, test_name):
        logger.info("Training model...")
        path = join(rootdir, "tmp", "forecast_ensemble", test_name)
        self.ensemble.train(self.vals_train)

        # generate alarms for the test sequence using the ensemble
        # this will return an aggregated alarms from all the models inside the ensemble
        yhat, _ = self.ensemble.forecast(self.vals_test.time_stamps)
        yhat = yhat.univariates[yhat.names[0]].np_values
        logger.info("forecast looks like " + str(yhat[:3]))
        self.assertEqual(len(yhat), len(self.vals_test))

        logger.info("Testing save/load...")
        self.ensemble.save(path, save_only_used_models=True)
        ensemble = ForecasterEnsemble.load(path)
        loaded_yhat = ensemble.forecast(self.vals_test.time_stamps)[0]
        loaded_yhat = loaded_yhat.univariates[loaded_yhat.names[0]].np_values
        self.assertSequenceEqual(list(yhat), list(loaded_yhat))

        # serialize and deserialize
        obj = self.ensemble.to_bytes()
        ensemble = ModelFactory.load_bytes(obj)
        loaded_yhat = ensemble.forecast(self.vals_test.time_stamps)[0]
        loaded_yhat = loaded_yhat.univariates[loaded_yhat.names[0]].np_values
        self.assertSequenceEqual(list(yhat), list(loaded_yhat))

        # test sMAPE
        y = self.vals_test.np_values
        smape = np.mean(200.0 * np.abs((y - yhat) / (np.abs(y) + np.abs(yhat))))
        logger.info(f"sMAPE = {smape:.4f}")
        self.assertAlmostEqual(smape, self.expected_smape, delta=2 if self.expected_smape < 100 else 10)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
