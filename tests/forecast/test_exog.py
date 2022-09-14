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

import pandas as pd
import numpy as np

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.utils.time_series import TimeSeries
from ts_datasets.forecast import CustomDataset

logger = logging.getLogger(__name__)
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExog(unittest.TestCase):
    def test_exog_ensemble(self):
        print("-" * 80)
        logger.info("TestExog.test_exog_ensemble\n" + "-" * 80)
        # Get train, test, and exogenous data
        csv = os.path.join(rootdir, "data", "walmart", "walmart_mini.csv")
        index_cols = ["Store", "Dept"]
        target = ["Weekly_Sales"]
        ts, md = CustomDataset(rootdir=csv, test_frac=0.25, index_cols=index_cols)[0]
        train = TimeSeries.from_pd(ts.loc[md.trainval, target])
        test = TimeSeries.from_pd(ts.loc[~md.trainval, target])
        exog = TimeSeries.from_pd(ts[[c for c in ts.columns if "MarkDown" in c or "Holiday" in c]])

        # Train model & get prediction
        model = ForecasterEnsemble(
            config=ForecasterEnsembleConfig(combiner=dict(name="ModelSelector", metric=ForecastMetric.RMSE)),
            models=[Prophet(ProphetConfig()), Arima(ArimaConfig(order=(4, 1, 2)))],
        )
        model.train(train_data=train, exog_data=exog)
        val_results = [(type(m).__name__, v) for m, v in zip(model.models, model.combiner.metric_values)]
        logger.info(f"Validation {model.combiner.metric.name}: {', '.join(f'{m}={v:.2f}' for m, v in val_results)}")
        pred, _ = model.forecast(time_stamps=test.time_stamps, exog_data=exog)

        # Evaluate model
        smape = ForecastMetric.sMAPE.value(test, pred)
        logger.info(f"Ensemble test sMAPE = {smape:.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
