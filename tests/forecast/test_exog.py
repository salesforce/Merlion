#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import logging
import sys
import unittest

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.automl.autoprophet import AutoProphet, AutoProphetConfig
from merlion.models.automl.autosarima import AutoSarima, AutoSarimaConfig
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.ensemble.combine import ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.utils.time_series import TimeSeries
from ts_datasets.forecast import CustomDataset

logger = logging.getLogger(__name__)
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExog(unittest.TestCase):
    def test_exog_ensemble(self):
        self._test_exog_ensemble(automl=False)

    def test_exog_automl_ensemble(self):
        self._test_exog_ensemble(automl=True)

    def _test_exog_ensemble(self, automl: bool):
        print("-" * 80)
        logger.info(f"TestExog.test_exog{'_automl' if automl else ''}_ensemble\n" + "-" * 80)
        # Get train, test, and exogenous data
        csv = os.path.join(rootdir, "data", "walmart", "walmart_mini.csv")
        index_cols = ["Store", "Dept"]
        target = ["Weekly_Sales"]
        ts, md = CustomDataset(rootdir=csv, test_frac=0.25, index_cols=index_cols)[0]
        train = TimeSeries.from_pd(ts.loc[md.trainval, target])
        test = TimeSeries.from_pd(ts.loc[~md.trainval, target])
        exog = TimeSeries.from_pd(ts[[c for c in ts.columns if "MarkDown" in c or "Holiday" in c]])

        if automl:
            models = [AutoProphet(AutoProphetConfig()), AutoSarima(AutoSarimaConfig(maxiter=10))]
        else:
            models = [Prophet(ProphetConfig()), Arima(ArimaConfig(order=(4, 1, 2)))]

        for ex in [None, exog]:
            # Train models & get prediction
            logger.info("With exogenous data..." if ex is not None else "No exogenous data...")
            model = ForecasterEnsemble(
                config=ForecasterEnsembleConfig(combiner=ModelSelector(metric=ForecastMetric.RMSE), models=models)
            )
            model.train(train_data=train, exog_data=ex)
            val_results = [(type(m).__name__, v) for m, v in zip(model.models, model.combiner.metric_values)]
            logger.info(f"Validation {model.combiner.metric.name}: {', '.join(f'{m}={v:.2f}' for m, v in val_results)}")
            pred, _ = model.forecast(time_stamps=test.time_stamps, exog_data=ex)

            # Evaluate model
            smape = ForecastMetric.sMAPE.value(test, pred)
            logger.info(f"Ensemble test sMAPE = {smape:.2f}\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
