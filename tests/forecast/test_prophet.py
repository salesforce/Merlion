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
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.utils.resample import to_timestamp
from merlion.utils.time_series import TimeSeries
from ts_datasets.forecast import CustomDataset

logger = logging.getLogger(__name__)
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProphet(unittest.TestCase):
    def test_resample_time_stamps(self):
        # arrange
        config = ProphetConfig()
        prophet = Prophet(config)
        prophet.last_train_time = pd._libs.tslibs.timestamps.Timestamp(year=2022, month=1, day=1)
        prophet.timedelta = pd._libs.tslibs.timedeltas.Timedelta(days=1)
        target = np.array([to_timestamp(pd._libs.tslibs.timestamps.Timestamp(year=2022, month=1, day=2))])

        # act
        output = prophet.resample_time_stamps(time_stamps=1)

        # assert
        assert output == target

    def test_exog(self):
        print("-" * 80)
        logger.info("TestProphet.test_exog\n" + "-" * 80)
        # Get train, test, and exogenous data
        csv = os.path.join(rootdir, "data", "walmart", "walmart_mini.csv")
        index_cols = ["Store", "Dept"]
        target = ["Weekly_Sales"]
        ts, md = CustomDataset(rootdir=csv, test_frac=0.25, index_cols=index_cols)[0]
        train = TimeSeries.from_pd(ts.loc[md.trainval, target])
        test = TimeSeries.from_pd(ts.loc[~md.trainval, target])
        exog = TimeSeries.from_pd(ts[[c for c in ts.columns if "MarkDown" in c or "Holiday" in c]])

        # Train model & get prediction
        model = Prophet(ProphetConfig())
        model.train(train_data=train)
        pred, _ = model.forecast(time_stamps=test.time_stamps)
        exog_model = Prophet(ProphetConfig())
        exog_model.train(train_data=train, exog_data=exog)
        exog_pred, _ = exog_model.forecast(time_stamps=test.time_stamps, exog_data=exog)

        # Evaluate model
        smape = ForecastMetric.sMAPE.value(test, pred)
        exog_smape = ForecastMetric.sMAPE.value(test, exog_pred)
        logger.info(f"sMAPE = {smape:.2f} (no exog)")
        logger.info(f"sMAPE = {exog_smape:.2f} (with exog)")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
