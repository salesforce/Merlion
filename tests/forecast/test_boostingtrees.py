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

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.trees import LGBMForecaster, LGBMForecasterConfig
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.transform.normalize import MeanVarNormalize
from merlion.utils import TimeSeries
from ts_datasets.forecast import SeattleTrail

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestLGBMForecaster(unittest.TestCase):
    """
    we test data loading, model instantiation, forecasting consistency, in particular
    (1) load a testing data
    (2) transform data
    (3) instantiate the model and train
    (4) forecast, and the forecasting result agrees with the reference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run_test(self, univariate: bool, use_exog: bool, autoregressive: bool):
        # Get the data and clean it up
        df, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", "seattle_trail"))[0]
        t = int(df[md["trainval"]].index[-1].to_pydatetime().timestamp())
        target = "BGT North of NE 70th Total"
        data = TimeSeries.from_pd(df.iloc[:, [1, 0, 2, 3, 4]])
        cleanup = TransformSequence([TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300)])
        cleanup.train(data)
        data = cleanup(data)

        # Pretend the last 2 non-target univariates are exogenous regressors if using exogenous data.
        # Test the benefit of adding them, so don't include them in the endogenous data by default.
        exog_cols = [c for c in data.names if c != target][-3:]
        data_cols = [target] if univariate else [c for c in data.names if c not in exog_cols]
        exog_data = None if not use_exog else TimeSeries([data.univariates[c] for c in exog_cols])
        data = TimeSeries([data.univariates[c] for c in data_cols])
        train_data, test_data = data.bisect(t)

        # Set up the model
        maxlags = 15
        max_forecast_steps = 20
        target_seq_idx = {k: i for i, k in enumerate(train_data.names)}[target]
        model = LGBMForecaster(
            LGBMForecasterConfig(
                max_forecast_steps=None if autoregressive else max_forecast_steps,
                maxlags=maxlags,
                target_seq_index=target_seq_idx,
                prediction_stride=5 if univariate and autoregressive else 1 if autoregressive else max_forecast_steps,
                n_estimators=20,
                max_depth=7,
                n_jobs=1,
                transform=MeanVarNormalize(),
                exog_transform=MeanVarNormalize(),
                invert_transform=True,
            )
        )

        # Train model
        yhat, _ = model.train(train_data, exog_data=exog_data)

        # Check RMSE with forecast transform inversion
        forecast, _ = model.forecast(max_forecast_steps, exog_data=exog_data)
        rmse = ForecastMetric.RMSE.value(test_data, forecast, target_seq_index=target_seq_idx)
        logger.info(f"Immediate forecast RMSE:  {rmse:.2f}")

        # Check look-ahead RMSE using time_series_prev
        dataset = RollingWindowDataset(test_data, target_seq_idx, maxlags, max_forecast_steps, ts_index=True)
        test_prev, test = dataset[0]
        pred, _ = model.forecast(test.time_stamps, time_series_prev=test_prev, exog_data=exog_data)
        lookahead_rmse = ForecastMetric.RMSE.value(test, pred)
        logger.info(f"Look-ahead RMSE with time_series_prev:  {lookahead_rmse:.2f}")

        # save and load
        model.save(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_model = LGBMForecaster.load(dirname=join(rootdir, "tmp", "lgbmforecaster"))
        loaded_pred, _ = loaded_model.forecast(test.time_stamps, time_series_prev=test_prev, exog_data=exog_data)
        self.assertEqual(len(loaded_pred), len(pred))
        self.assertAlmostEqual((pred.to_pd() - loaded_pred.to_pd()).abs().max().item(), 0, places=5)

    def test_forecast_multi_autoregression(self):
        print("-" * 80)
        logger.info("test_forecast_multi_autoregression" + "\n" + "-" * 80)
        self._run_test(univariate=False, autoregressive=True, use_exog=False)

    def test_forecast_multi_seq2seq(self):
        print("-" * 80)
        logger.info("test_forecast_multi_seq2seq" + "\n" + "-" * 80)
        self._run_test(univariate=False, autoregressive=False, use_exog=False)

    def test_forecast_uni(self):
        print("-" * 80)
        logger.info("test_forecast_uni" + "\n" + "-" * 80)
        self._run_test(univariate=True, autoregressive=True, use_exog=False)

    def test_forecast_multi_autoregression_exog(self):
        print("-" * 80)
        logger.info("test_forecast_multi_autoregression_exog" + "\n" + "-" * 80)
        self._run_test(univariate=False, autoregressive=True, use_exog=True)

    def test_forecast_multi_seq2seq_exog(self):
        print("-" * 80)
        logger.info("test_forecast_multi_seq2seq_exog" + "\n" + "-" * 80)
        self._run_test(univariate=False, autoregressive=False, use_exog=True)

    def test_forecast_uni_exog(self):
        print("-" * 80)
        logger.info("test_forecast_uni_exog" + "\n" + "-" * 80)
        self._run_test(univariate=True, autoregressive=True, use_exog=True)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
