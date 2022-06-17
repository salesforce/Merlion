#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
A variant of ARIMA with a user-specified Seasonality.
"""

import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as sm_Sarima

from merlion.models.automl.seasonality import SeasonalityModel
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries, to_pd_datetime, to_timestamp

logger = logging.getLogger(__name__)


class SarimaConfig(ForecasterConfig):
    """
    Config class for `Sarima` (Seasonal AutoRegressive Integrated Moving Average).
    """

    _default_transform = TemporalResample(granularity=None)

    def __init__(self, order=(4, 1, 2), seasonal_order=(2, 0, 1, 24), **kwargs):
        """
        :param order: Order is (p, d, q) for an ARIMA(p, d, q) process. d must
            be an integer indicating the integration order of the process, while
            p and q must be integers indicating the AR and MA orders (so that
            all lags up to those orders are included).
        :param seasonal_order: Seasonal order is (P, D, Q, S) for seasonal ARIMA
            process, where s is the length of the seasonality cycle (e.g. s=24
            for 24 hours on hourly granularity). P, D, Q are as for ARIMA.
        """
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order


class Sarima(ForecasterBase, SeasonalityModel):
    """
    Implementation of the classic statistical model SARIMA (Seasonal
    AutoRegressive Integrated Moving Average) for forecasting.
    """

    config_class = SarimaConfig

    def __init__(self, config: SarimaConfig):
        super().__init__(config)
        self.model = None
        self._last_val = None

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def order(self) -> Tuple[int, int, int]:
        """
        :return: the order (p, d, q) of the model, where p is the AR order,
            d is the integration order, and q is the MA order.
        """
        return self.config.order

    @property
    def seasonal_order(self) -> Tuple[int, int, int, int]:
        """
        :return: the seasonal order (P, D, Q, S) for the seasonal ARIMA
            process, where p is the AR order, D is the integration order,
            Q is the MA order, and S is the length of the seasonality cycle.
        """
        return self.config.seasonal_order

    @property
    def _max_lookback(self) -> int:
        if self.model is None:
            return 0
        orders = self.model.model_orders
        if orders["reduced_ma"] > 0:
            return 0
        return 2 * orders["reduced_ar"] + 1

    def _train(self, train_data: pd.DataFrame, train_config=None):
        # train model
        name = self.target_name
        train_data = train_data[name]
        times = train_data.index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = sm_Sarima(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(method_kwargs={"disp": 0})

        # FORECASTING: forecast for next n steps using Sarima model
        self._last_val = train_data[-1]
        yhat = (train_data.values - self.model.resid).tolist()
        err = [np.sqrt(self.model.params["sigma2"])] * len(train_data)
        return pd.DataFrame(yhat, index=times, columns=[name]), pd.DataFrame(err, index=times, columns=[f"{name}_err"])

    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Directly obtain forecast if no time_series_prev
        if time_series_prev is None:
            forecast_result = self.model.get_forecast(len(time_stamps))
            forecast = np.asarray(forecast_result.predicted_mean)
            err = np.asarray(forecast_result.se_mean)
            last_val = self._last_val

        # If there is a time_series_prev, use it to set the SARIMA model's state, and then obtain its forecast
        else:
            time_series_prev = time_series_prev.iloc[:, self.target_seq_index]
            t_prev = to_timestamp(time_series_prev.index)
            val_prev = time_series_prev.values[-self._max_lookback :]
            last_val = val_prev[-1]

            try:
                new_state = self.model.apply(val_prev, validate_specification=False)
                forecast_result = new_state.get_forecast(len(time_stamps))
                forecast = np.asarray(forecast_result.predicted_mean)
                err = np.asarray(forecast_result.se_mean)
                assert len(forecast) == len(time_stamps), (
                    f"Expected SARIMA model to return forecast of length {len(time_stamps)}, but got "
                    f"{len(forecast)} instead."
                )
            except Exception as e:
                logger.warning(f"Caught {type(e).__name__}: {str(e)}")
                forecast = np.full(len(time_stamps), last_val)
                err = np.zeros(len(time_stamps))

            if return_prev:
                m = len(time_series_prev) - len(val_prev)
                params = dict(zip(new_state.param_names, new_state.params))
                err_prev = np.concatenate((np.zeros(m), np.full(len(val_prev), np.sqrt(params["sigma2"]))))
                forecast = np.concatenate((time_series_prev.values[:m], val_prev - new_state.resid, forecast))
                err = np.concatenate((err_prev, err))
                time_stamps = np.concatenate((t_prev, time_stamps))

        # Check for NaN's
        if any(np.isnan(forecast)):
            logger.warning(
                "Trained SARIMA model is producing NaN forecast. Use the last training point as the prediction."
            )
            forecast[np.isnan(forecast)] = last_val
        if any(np.isnan(err)):
            err[np.isnan(err)] = 0

        # Return the forecast & its standard error
        name = self.target_name
        forecast = pd.DataFrame(forecast, index=to_pd_datetime(time_stamps), columns=[name])
        err = pd.DataFrame(err, index=to_pd_datetime(time_stamps), columns=[f"{name}_err"])
        return forecast, err

    def set_seasonality(self, theta, train_data: UnivariateTimeSeries):
        # Make sure seasonality is a positive int, and set it to 1 if the train data is constant
        theta = 1 if np.max(train_data) == np.min(train_data) else max(1, int(theta))
        self.config.seasonal_order = self.seasonal_order[:-1] + (theta,)
