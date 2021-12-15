#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
A variant of ARIMA with a user-specified Seasonality.
"""

import logging
import warnings
from typing import List, Tuple, Union

import numpy as np
from merlion.utils import autosarima_utils
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA as sm_Sarima

from merlion.models.automl.seasonality import SeasonalityModel
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries

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
        self.last_val = None

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

    def train(self, train_data: TimeSeries, train_config=None):
        # Train the transform & transform the training data
        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)

        # train model
        name = self.target_name
        train_data = train_data.univariates[name].to_pd()
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
        forecast_result = self.model.get_forecast(self.max_forecast_steps)
        self.last_val = train_data[-1]

        yhat = (train_data.values - self.model.resid).tolist()
        err = [np.sqrt(self.model.params["sigma2"])] * len(train_data)
        return (
            UnivariateTimeSeries(times, yhat, name).to_ts(),
            UnivariateTimeSeries(times, err, f"{name}_err").to_ts(),
        )

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr=False,
        return_prev=False,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        # Make sure the timestamps are valid (spaced at the right timedelta)
        # If time_series_prev is None, i0 is the first index of the pre-computed
        # forecast, which we'd like to start returning a forecast from
        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)

        # transform time_series_prev if relevant (before making the prediction)
        if time_series_prev is not None:
            time_series_prev = self.transform(time_series_prev)

        if time_series_prev is None:
            forecast_result = self.model.get_forecast(len(time_stamps))
            forecast = forecast_result.predicted_mean
            err = forecast_result.se_mean
            if any(np.isnan(forecast)):
                logger.warning(
                    "Trained SARIMA model is producing NaN forecast.Use the last "
                    "point in the training data as the prediction."
                )
                forecast[np.isnan(forecast)] = self.last_val
            if any(np.isnan(err)):
                err[np.isnan(err)] = 0

        # If there is a time_series_prev, use it to set the ARIMA model's state,
        # and then obtain its forecast (and standard error of that forecast)
        else:
            k = time_series_prev.names[self.target_seq_index]
            time_series_prev = time_series_prev.univariates[k]
            t_prev = time_series_prev.time_stamps
            val_prev = time_series_prev.np_values[-self._max_lookback :]

            try:
                new_state = self.model.apply(val_prev, validate_specification=False)
                forecast_result = new_state.get_forecast(len(time_stamps))
                forecast = forecast_result.predicted_mean
                err = forecast_result.se_mean
                assert len(forecast) == len(time_stamps), (
                    f"Expected SARIMA model to return forecast of length {len(time_stamps)}, but got "
                    f"{len(forecast)} instead."
                )
            except Exception as e:
                logger.warning(f"Caught {type(e).__name__}: {str(e)}")
                forecast = np.full(len(time_stamps), val_prev[-1])
                err = np.zeros(len(time_stamps))

            if return_prev:
                n_prev = len(time_series_prev)
                params = dict(zip(new_state.param_names, new_state.params))
                err_prev = np.sqrt(params["sigma2"])
                forecast = np.concatenate((val_prev - new_state.resid, forecast))
                err = np.concatenate((err_prev * np.ones(n_prev), err))
                time_stamps = t_prev + time_stamps
                orig_t = None if orig_t is None else t_prev + orig_t

        # Return the IQR (25%ile & 75%ile) along with the forecast if desired
        name = self.target_name
        if return_iqr:
            lb = (
                UnivariateTimeSeries(
                    name=f"{name}_lower", time_stamps=time_stamps, values=forecast + norm.ppf(0.25) * err
                )
                .to_ts()
                .align(reference=orig_t)
            )
            ub = (
                UnivariateTimeSeries(
                    name=f"{name}_upper", time_stamps=time_stamps, values=forecast + norm.ppf(0.75) * err
                )
                .to_ts()
                .align(reference=orig_t)
            )
            forecast = (
                UnivariateTimeSeries(name=name, time_stamps=time_stamps, values=forecast)
                .to_ts()
                .align(reference=orig_t)
            )
            return forecast, lb, ub

        # Otherwise, just return the forecast & its standard error
        else:
            forecast = (
                UnivariateTimeSeries(name=name, time_stamps=time_stamps, values=forecast)
                .to_ts()
                .align(reference=orig_t)
            )
            err = (
                UnivariateTimeSeries(name=f"{name}_err", time_stamps=time_stamps, values=err)
                .to_ts()
                .align(reference=orig_t)
            )
            return forecast, err

    def set_seasonality(self, theta, train_data: UnivariateTimeSeries):
        # Make sure seasonality is a positive int, and set it to 1 if the train data is constant
        theta = 1 if np.max(train_data) == np.min(train_data) else max(1, int(theta))
        self.config.seasonal_order = self.seasonal_order[:-1] + (theta,)
