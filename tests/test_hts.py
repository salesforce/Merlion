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

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.factory import ModelFactory
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.transform.moving_average import DifferenceTransform
from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.utils.data_io import df_to_time_series
from merlion.utils.hts import minT_reconciliation
from ts_datasets.forecast import SeattleTrail

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(abspath(__file__)))


class TestHTS(unittest.TestCase):
    """
    we test data loading, model instantiation, forecasting consistency, in particular
    (1) load a testing data
    (2) transform data
    (3) instantiate the model and train
    (4) forecast, and the forecasting result agrees with the reference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_forecast_steps = 2
        self.maxlags = 6
        self.i = 0

        dataset = "seattle_trail"
        d, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", dataset))[0]
        t = int(d[md["trainval"]].index[-1].to_pydatetime().timestamp())
        data = TimeSeries.from_pd(d)
        cleanup_transform = TransformSequence(
            [TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300), DifferenceTransform()]
        )
        cleanup_transform.train(data)
        data = cleanup_transform(data)

        train_data, test_data = data.bisect(t)

        h = 100
        minmax_transform = MinMaxNormalize()
        minmax_transform.train(train_data)
        self.train_data_norm = minmax_transform(train_data[-2000:])
        self.test_data_norm = minmax_transform(test_data[:h])
        self.train_data_agg = UnivariateTimeSeries.from_pd(self.train_data_norm.to_pd().sum(axis=1), name="val").to_ts()
        self.test_data_agg = UnivariateTimeSeries.from_pd(self.test_data_norm.to_pd().sum(axis=1), name="val").to_ts()

        self.models = [ModelFactory.create("AutoETS", target_seq_index=i) for i in range(test_data.dim)]
        self.agg_model = ModelFactory.create("LGBMForecaster", max_forecast_steps=h, maxlags=100)

    def test_minT(self):
        print("=" * 80)
        logger.info("test_minT" + "\n" + "=" * 80)
        logger.info("Training models...")
        forecasts, errs = [], []
        models = [self.agg_model, *self.models]
        train_data = [self.train_data_agg] + [self.train_data_norm] * len(self.models)
        test_data = [self.test_data_agg] + [self.test_data_norm] * len(self.models)
        for model, train, test in zip(models, train_data, test_data):
            model.train(train)
            forecast, err = model.forecast(test.time_stamps)
            forecasts.append(forecast)
            errs.append(None if len(errs) == 1 else err)

        logger.info("Applying reconciliation...")
        sum_matrix = np.concatenate([np.ones((1, len(self.models))), np.eye(len(self.models))])
        reconciled = minT_reconciliation(forecasts, errs, sum_matrix=sum_matrix, n_leaves=len(self.models))

        naive_sum = np.sum([f.to_pd().values.flatten() for f in forecasts[1:]])
        naive_sum = UnivariateTimeSeries(time_stamps=self.test_data_agg.time_stamps, values=naive_sum).to_ts()
        naive = ForecastMetric.RMSE.value(predict=naive_sum, ground_truth=self.test_data_agg)
        direct = ForecastMetric.RMSE.value(predict=forecasts[0], ground_truth=self.test_data_agg)
        minT = ForecastMetric.RMSE.value(predict=reconciled[0], ground_truth=self.test_data_agg)

        logger.info(f"Naive summation     RMSE: {naive:.2f}")
        logger.info(f"Direct prediction   RMSE: {direct:.4f}")
        logger.info(f"minT reconciliation RMSE: {minT:.4f}")
        self.assertLess(direct, naive)
        self.assertLess(minT, direct)

    def test_data_io(self):
        print("=" * 80)
        logger.info("test_data_io" + "\n" + "=" * 80)
        df = self.train_data_norm.to_pd()
        df_hierarchical = pd.concat(
            [pd.DataFrame({"name": c, "val": df[c].values}, index=df.index) for c in df.columns]
        )
        df_hierarchical.index.name = None
        ts_agg = df_to_time_series(df_hierarchical, index_cols=["name"])
        max_delta = (ts_agg.to_pd() - self.train_data_agg.to_pd()).abs().max().item()
        self.assertLessEqual(max_delta, 1e-8)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
