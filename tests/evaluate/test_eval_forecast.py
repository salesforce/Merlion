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


from merlion.evaluate.forecast import ForecastEvaluator, ForecastEvaluatorConfig, ForecastMetric
from merlion.models.ensemble.combine import MetricWeightedMean
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.models.forecast.arima import ArimaConfig, Arima
from merlion.transform.base import Identity
from merlion.utils.time_series import UnivariateTimeSeries, ts_csv_load


logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestEvaluateForecast(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_single_model(self):
        print("-" * 80)
        logger.info("test_single_model\n" + "-" * 80 + "\n")

        # Create a piecewise linear time series
        values = [i for i in range(60)]
        values += [values[-1] + 2 * (i + 1) for i in range(30)]
        values += [values[-1] + 3 * (i + 1) for i in range(30)]
        values += [values[-1] + 1 * (i + 1) for i in range(30)]
        ts = UnivariateTimeSeries(time_stamps=None, values=values, freq="1d").to_ts()

        # Get train & test split
        self.train_data = ts[:30]
        self.test_data = ts[30:]

        # Set up a simple ARIMA model that can learn a linear relationship
        # We will be training the model on 30 day chunks (which are linear) and
        # having it forecast on 30 days chunks (which may have a different
        # slope than the model expects)
        self.model = Arima(ArimaConfig(order=(1, 1, 0), max_forecast_steps=30))

        logger.info("Training model using an evaluator...")
        evaluator = ForecastEvaluator(
            model=self.model, config=ForecastEvaluatorConfig(retrain_freq="30d", train_window="30d")
        )

        # Get pred
        _, pred = evaluator.get_predict(train_vals=self.train_data, test_vals=self.test_data)

        # Calculate evaluation metric
        smape = evaluator.evaluate(ground_truth=self.test_data, predict=pred, metric=ForecastMetric.sMAPE)
        self.assertAlmostEqual(smape, 9.823, delta=0.001)

    def test_ensemble(self):
        print("-" * 80)
        logger.info("test_ensemble\n" + "-" * 80 + "\n")

        csv_name = join(rootdir, "data", "example.csv")
        ts = ts_csv_load(csv_name, ms=True, n_vars=1).align(granularity="1h")
        n_test = len(ts) // 5
        train, test = ts[:-n_test], ts[-n_test:]

        # Construct ensemble to forecast up to 120hr in the future
        n = 120
        kwargs = dict(max_forecast_steps=n, transform=Identity())
        model0 = Arima(ArimaConfig(order=(4, 1, 2), **kwargs))
        model1 = Arima(ArimaConfig(order=(20, 0, 0), **kwargs))
        model2 = Arima(ArimaConfig(order=(6, 2, 1), **kwargs))
        ensemble = ForecasterEnsemble(
            config=ForecasterEnsembleConfig(combiner=MetricWeightedMean(metric=ForecastMetric.sMAPE)),
            models=[model0, model1, model2],
        )

        # Set up evaluator & run it on the data
        evaluator = ForecastEvaluator(
            model=ensemble,
            config=ForecastEvaluatorConfig(retrain_freq="7d", horizon="5d", cadence=0, train_window=None),
        )

        _, pred = evaluator.get_predict(train_vals=train, test_vals=test)
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred), len(test))

        # Compute ensemble's sMAPE
        smape = evaluator.evaluate(ground_truth=test, predict=pred, metric=ForecastMetric.sMAPE)
        logger.info(f"Ensemble sMAPE: {smape}")

        # Do a quick test of save/load
        ensemble.save("tmp/eval/forecast_ensemble")
        ForecasterEnsemble.load("tmp/eval/forecast_ensemble")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
