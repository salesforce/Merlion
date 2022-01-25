#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
ETS (Error, Trend, Seasonal) forecasting model.
"""

import logging
from typing import List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from merlion.models.automl.seasonality import SeasonalityModel
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class ETSConfig(ForecasterConfig):
    """
    Configuration class for :py:class:`ETS` model. ETS model is an underlying state space
    model consisting of an error term (E), a trend component (T), a seasonal
    component (S), and a level component. Each component is flexible with
    different traits with additive ('add') or multiplicative ('mul') formulation.
    Refer to https://otexts.com/fpp2/taxonomy.html for more information
    about ETS model.
    """

    _default_transform = TemporalResample(granularity=None)

    def __init__(
        self,
        max_forecast_steps=None,
        target_seq_index=None,
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=None,
        **kwargs,
    ):
        """
        :param max_forecast_steps: Number of steps we would like to forecast for.
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param error: The error term. "add" or "mul".
        :param trend: The trend component. "add", "mul" or None.
        :param damped_trend: Whether or not an included trend component is damped.
        :param seasonal: The seasonal component. "add", "mul" or None.
        :param seasonal_periods: The length of the seasonality cycle. ``None`` by default.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods


class ETS(SeasonalityModel, ForecasterBase):
    """
    Implementation of the classic local statistical model ETS (Error, Trend, Seasonal) for forecasting.
    """

    config_class = ETSConfig

    def __init__(self, config: ETSConfig):
        super().__init__(config)
        self.model = None
        self.last_train_window = None
        self._last_val = None
        self._n_train = None

    @property
    def error(self):
        return self.config.error

    @property
    def trend(self):
        return self.config.trend

    @property
    def damped_trend(self):
        return self.config.damped_trend

    @property
    def seasonal(self):
        return self.config.seasonal

    @property
    def seasonal_periods(self):
        return self.config.seasonal_periods

    def set_seasonality(self, theta, train_data: UnivariateTimeSeries):
        if theta > 1:
            self.config.seasonal_periods = int(theta)
        else:
            self.config.seasonal = None
            self.config.seasonal_periods = None

    def train(self, train_data: TimeSeries, train_config=None):
        # Train the transform & transform the training data
        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)

        # train model
        name = self.target_name
        train_data = train_data.univariates[name].to_pd()
        times = train_data.index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ETSModel(
                train_data,
                error=self.error,
                trend=self.trend,
                seasonal=None if self.seasonal_periods is None else self.seasonal,
                damped_trend=self.damped_trend,
                seasonal_periods=self.seasonal_periods,
            ).fit(disp=False)

        # to match the minimum data size requirement when refitting new data
        last_train_window_size = 10
        if self.seasonal_periods is not None:
            last_train_window_size = max(10, 10 + 2 * (self.seasonal_periods // 2), 2 * self.seasonal_periods)
            last_train_window_size = min(last_train_window_size, train_data.shape[0])
        self.last_train_window = train_data[-last_train_window_size:]

        # FORECASTING: forecast for next n steps using ETS model
        self._n_train = train_data.shape[0]
        self._last_val = train_data[-1]

        yhat = self.model.fittedvalues.values.tolist()
        err = self.model.standardized_forecasts_error.tolist()
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
        refit=True,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        # Make sure the timestamps are valid (spaced at the right timedelta)
        # If time_series_prev is None, i0 is the first index of the pre-computed
        # forecast, which we'd like to start returning a forecast from
        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)

        # transform time_series_prev if relevant (before making the prediction)
        if time_series_prev is not None:
            time_series_prev = self.transform(time_series_prev)
            _, new_data = time_series_prev.bisect(self.last_train_time + self.timedelta, t_in_left=False)
            # if time_series_prev does not contain new data w.r.t. train_data, we skip it
            if new_data.is_empty():
                time_series_prev = None

        if time_series_prev is None:
            forecast_result = self.model.get_prediction(
                start=self._n_train, end=self._n_train + len(time_stamps) - 1, method="exact"
            )
            forecast = forecast_result.predicted_mean
            err = np.sqrt(forecast_result.var_pred_mean)
            if any(np.isnan(forecast)):
                logger.warning(
                    "Trained ETS model is producing NaN forecast. Use the last "
                    "point in the training data as the prediction."
                )
                forecast[np.isnan(forecast)] = self._last_val
            if any(np.isnan(err)):
                err[np.isnan(err)] = 0

        # If there is a time_series_prev, use it to smooth ETS model,
        # and then obtain its forecast (and standard error of that forecast)
        else:
            k = time_series_prev.names[self.target_seq_index]
            time_series_prev = time_series_prev.univariates[k]
            last_train_window_size = self.last_train_window.shape[0]
            time_series_prev_pd = time_series_prev.to_pd()

            # truncate time series window for smooth or refit
            if len(time_series_prev) >= last_train_window_size:
                mask = time_series_prev_pd.index > self.last_train_window.index[-1]
                if sum(mask) <= last_train_window_size:
                    val_prev = time_series_prev_pd[-last_train_window_size:]
                else:
                    val_prev = time_series_prev_pd[-sum(mask) :]
                self.last_train_window = time_series_prev_pd[-last_train_window_size:]
            else:
                mask = self.last_train_window.index < time_series_prev_pd.index[0]
                val_prev = pd.concat([self.last_train_window[mask], time_series_prev_pd])[-last_train_window_size:]
                self.last_train_window = val_prev
            new_model = ETSModel(
                val_prev,
                error=self.error,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend,
                seasonal_periods=self.seasonal_periods,
            )

            # the default setting of refit=False is fast and conduct exponential smoothing with given parameters,
            # while the setting of refit=True is slow and refit the model with a selected training set from
            # time_series_prev and self.last_train_window
            if refit:
                self.model = new_model.fit(start_params=self.model.params, disp=False)
            else:
                self.model = new_model.smooth(params=self.model.params)
            forecast_result = self.model.get_prediction(
                start=val_prev.shape[0], end=val_prev.shape[0] + len(time_stamps) - 1, method="simulated"
            )
            forecast = forecast_result.predicted_mean
            err = forecast_result._results.simulation_results.std(axis=1)

            # if return_prev is Ture, it will return the forecast and error of last train window
            # instead of time_series_prev
            if return_prev:
                yhat_prev = self.model.fittedvalues.values.tolist()
                err_prev = self.model.standardized_forecasts_error.tolist()
                forecast = np.concatenate((yhat_prev, forecast))
                err = np.concatenate((err_prev, err))
                t_prev = self.last_train_window.index.values.astype("datetime64[s]").astype(np.int64).tolist()
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
