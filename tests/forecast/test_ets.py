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

import pandas as pd
import numpy as np

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.forecast.ets import ETSConfig, ETS
from merlion.utils.time_series import TimeSeries, to_pd_datetime

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestETS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = [
            30.05251300,
            19.14849600,
            25.31769200,
            27.59143700,
            32.07645600,
            23.48796100,
            28.47594000,
            35.12375300,
            36.83848500,
            25.00701700,
            30.72223000,
            28.69375900,
            36.64098600,
            23.82460900,
            29.31168300,
            31.77030900,
            35.17787700,
            19.77524400,
            29.60175000,
            34.53884200,
            41.27359900,
            26.65586200,
            28.27985900,
            35.19115300,
            42.20566386,
            24.64917133,
            32.66733514,
            37.25735401,
            45.24246027,
            29.35048127,
            36.34420728,
            41.78208136,
            49.27659843,
            31.27540139,
            37.85062549,
            38.83704413,
            51.23690034,
            31.83855162,
            41.32342126,
            42.79900337,
            55.70835836,
            33.40714492,
            42.31663797,
            45.15712257,
            59.57607996,
            34.83733016,
            44.84168072,
            46.97124960,
            60.01903094,
            38.37117851,
            46.97586413,
            50.73379646,
            61.64687319,
            39.29956937,
            52.67120908,
            54.33231689,
            66.83435838,
            40.87118847,
            51.82853579,
            57.49190993,
            65.25146985,
            43.06120822,
            54.76075713,
            59.83447494,
            73.25702747,
            47.69662373,
            61.09776802,
            66.05576122,
        ]
        data = TimeSeries.from_pd(pd.Series(data))
        idx = int(0.7 * len(data))
        self.train_data = data[:idx]
        self.test_data = data[idx:]
        self.data = data
        self.max_forecast_steps = len(self.test_data)
        self.model = ETS(ETSConfig(error="add", trend="mul", seasonal="add", damped_trend=True, seasonal_periods=4, pred_interval_strategy="simulated"))

    def _multi_setup(self):
        x = self.data.to_pd()
        self.data = TimeSeries.from_pd(
            pd.DataFrame(np.concatenate((x.values, x.values * 2), axis=1), columns=["A", "B"], index=x.index)
        )
        self.train_data = self.data[: len(self.train_data)]
        self.test_data = self.data[len(self.train_data) :]
        self.model.config.target_seq_index = 0

    def test_univariate(self):
        self._test_forecast()

    def test_multivariate(self):
        self._multi_setup()
        self._test_forecast()

    def _test_forecast(self):
        # batch forecasting RMSE = 6.5612
        _, _ = self.model.train(self.train_data)
        forecast, lb, ub = self.model.forecast(self.max_forecast_steps, return_iqr=True)
        rmse = ForecastMetric.RMSE.value(self.test_data, forecast, target_seq_index=0)
        logger.info(f"RMSE = {rmse:.4f} for {self.max_forecast_steps} step forecasting")

        rmspe = ForecastMetric.RMSPE.value(self.test_data, forecast, target_seq_index=0)
        logger.info(f"RMPSE = {rmspe:.4f} for {self.max_forecast_steps} step forecasting")
        smape = ForecastMetric.sMAPE.value(self.test_data, forecast, target_seq_index=0)
        logger.info(f"sMAPE = {smape:.4f} for {self.max_forecast_steps} step forecasting")
        msis = ForecastMetric.MSIS.value(
            ground_truth=self.test_data,
            predict=forecast,
            insample=self.train_data,
            periodicity=4,
            ub=ub,
            lb=lb,
            target_seq_index=0,
        )
        logger.info(f"MSIS = {msis:.4f}")


        # make sure save/load model gets same predictions
        logger.info("Test save/load...")
        savedir = join(rootdir, "tmp", "ets")
        self.model.save(dirname=savedir)
        loaded = ETS.load(dirname=savedir)

        loaded_pred, loaded_lb, loaded_ub = loaded.forecast(self.max_forecast_steps, return_iqr=True)
        self.assertSequenceEqual(list(loaded_pred), list(forecast))


        # streaming forecasting
        test_t = self.test_data.np_time_stamps
        t, tf = to_pd_datetime([test_t[0], test_t[-1]])
        forecast_results = None
        while t < tf:
            cur_train, cur_test = self.data.bisect(t, t_in_left=False)
            cur_test = cur_test.window(t, t + self.model.timedelta)
            forecast, err = self.model.forecast(cur_test.time_stamps, cur_train)
            if forecast_results is None:
                forecast_results = forecast
            forecast_results += forecast
            t += self.model.timedelta
        rmse_onestep = ForecastMetric.RMSE.value(self.test_data, forecast_results, target_seq_index=0)
        logger.info(f"Streaming RMSE = {rmse_onestep:.4f} for {self.max_forecast_steps} step forecasting")


        # streaming forecasting performs better than batch forecasting
        self.assertLessEqual(rmse_onestep, rmse)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
