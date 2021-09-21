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

from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.models.ensemble.combine import ModelSelector, Mean
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.factory import ModelFactory
from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import ts_csv_load

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestForecastEnsemble(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_name = join(rootdir, "data", "example.csv")
        self.test_len = 2048
        data = ts_csv_load(self.csv_name, n_vars=1)[::10]
        self.vals_train = data[: -self.test_len]
        self.vals_test = data[-self.test_len :].univariates[data.names[0]]

        model0 = Arima(ArimaConfig(order=(6, 1, 2), max_forecast_steps=50, transform=TemporalResample("1h")))
        model1 = Arima(ArimaConfig(order=(24, 1, 0), transform=TemporalResample("10min"), max_forecast_steps=50))
        model2 = Prophet(ProphetConfig(transform=Identity()))
        model2.model.logger = None
        self.ensemble = ForecasterEnsemble(
            models=[model0, model1, model2], config=ForecasterEnsembleConfig(combiner=Mean(abs_score=False))
        )

    def test_mean(self):
        print("-" * 80)
        self.expected_smape = 37
        self.ensemble.models[0].config.max_forecast_steps = None
        self.ensemble.models[1].config.max_forecast_steps = None
        logger.info("test_mean\n" + "-" * 80 + "\n")
        self.run_test()

    def test_selector(self):
        print("-" * 80)
        self.expected_smape = 35
        logger.info("test_selector\n" + "-" * 80 + "\n")
        self.ensemble.config.combiner = ModelSelector(metric=ForecastMetric.sMAPE)
        self.run_test()
        # We expect the model selector to select Prophet because it gets the lowest validation sMAPE
        valid_smapes = np.asarray(self.ensemble.combiner.metric_values)
        self.assertAlmostEqual(np.max(np.abs(valid_smapes - [34.65, 39.62, 30.71])), 0, delta=0.5)
        self.assertSequenceEqual(self.ensemble.models_used, [False, False, True])

    def run_test(self):
        logger.info("Training model...")
        self.ensemble.train(self.vals_train)

        # generate alarms for the test sequence using the ensemble
        # this will return an aggregated alarms from all the models inside the ensemble
        yhat, _ = self.ensemble.forecast(self.vals_test.time_stamps)
        logger.info("forecast looks like " + str(yhat[:3]))
        self.assertEqual(len(yhat), len(self.vals_test))

        y = self.vals_test.np_values
        yhat = yhat.univariates[yhat.names[0]].np_values
        smape = np.mean(200.0 * np.abs((y - yhat) / (np.abs(y) + np.abs(yhat))))
        logger.info(f"sMAPE = {smape:.4f}")
        self.assertAlmostEqual(smape, self.expected_smape, delta=1)

        logger.info("Testing save/load...")
        self.ensemble.save(join(rootdir, "tmp", "forecast_ensemble"))
        ensemble = ForecasterEnsemble.load(join(rootdir, "tmp", "forecast_ensemble"))
        loaded_yhat = ensemble.forecast(self.vals_test.time_stamps)[0]
        loaded_yhat = loaded_yhat.univariates[loaded_yhat.names[0]].np_values
        self.assertSequenceEqual(list(yhat), list(loaded_yhat))

        # serialize and deserialize
        obj = self.ensemble.to_bytes()
        ensemble = ModelFactory.load_bytes(obj)
        loaded_yhat = ensemble.forecast(self.vals_test.time_stamps)[0]
        loaded_yhat = loaded_yhat.univariates[loaded_yhat.names[0]].np_values
        self.assertSequenceEqual(list(yhat), list(loaded_yhat))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
