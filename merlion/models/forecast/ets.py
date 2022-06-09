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
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from merlion.models.automl.seasonality import SeasonalityModel
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries, UnivariateTimeSeries, to_pd_datetime

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
        refit=True,
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
        :param refit: if ``True``, refit the full ETS model when ``time_series_prev`` is given to the forecast method
            (slower). If ``False``, simply perform exponential smoothing (faster).
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.refit = refit


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
    def require_even_sampling(self) -> bool:
        return True

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

    def _train(self, train_data: pd.DataFrame, train_config=None):
        # train model
        name = self.target_name
        train_data = train_data[name]
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

        yhat = pd.DataFrame(self.model.fittedvalues.values.tolist(), index=times, columns=[name])
        err = pd.DataFrame(self.model.standardized_forecasts_error.tolist(), index=times, columns=[f"{name}_err"])
        return yhat, err

    def _forecast(
        self, time_stamps: Union[int, List[int]], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Ignore time_series_prev if it comes after the training data
        if time_series_prev is not None and time_series_prev.index[-1] < self.last_train_time + self.timedelta:
            time_series_prev = None

        # Basic forecasting without time_series_prev
        if time_series_prev is None:
            forecast_result = self.model.get_prediction(
                start=self._n_train, end=self._n_train + len(time_stamps) - 1, method="exact"
            )
            forecast = np.asarray(forecast_result.predicted_mean)
            err = np.sqrt(np.asarray(forecast_result.var_pred_mean))

        # If there is a time_series_prev, use it to smooth/refit ETS model,
        # and then obtain its forecast (and standard error of that forecast)
        else:
            time_series_prev = time_series_prev.iloc[:, self.target_seq_index]
            last_train_window_size = len(self.last_train_window)

            # truncate time series window for smooth or refit
            if len(time_series_prev) >= last_train_window_size:
                mask = time_series_prev.index > self.last_train_window.index[-1]
                if sum(mask) <= last_train_window_size:
                    val_prev = time_series_prev[-last_train_window_size:]
                else:
                    val_prev = time_series_prev[-sum(mask) :]
                self.last_train_window = time_series_prev[-last_train_window_size:]
            else:
                mask = self.last_train_window.index < time_series_prev.index[0]
                val_prev = pd.concat([self.last_train_window[mask], time_series_prev])[-last_train_window_size:]
                self.last_train_window = val_prev
            self._last_val = val_prev[-1]
            new_model = ETSModel(
                val_prev,
                error=self.error,
                trend=self.trend,
                seasonal=None if self.seasonal_periods is None else self.seasonal,
                damped_trend=self.damped_trend,
                seasonal_periods=self.seasonal_periods,
            )

            # the default setting of refit=False is fast and conduct exponential smoothing with given parameters,
            # while the setting of refit=True is slow and refit the model with a selected training set from
            # time_series_prev and self.last_train_window
            if self.config.refit:
                self.model = new_model.fit(start_params=self.model.params, disp=False)
            else:
                self.model = new_model.smooth(params=self.model.params)
            forecast_result = self.model.get_prediction(
                start=val_prev.shape[0], end=val_prev.shape[0] + len(time_stamps) - 1, method="simulated"
            )
            forecast = np.asarray(forecast_result.predicted_mean)
            err = np.asarray(forecast_result._results.simulation_results.std(axis=1))

            # if return_prev is Ture, it will return the forecast and error of last train window
            # instead of time_series_prev
            if return_prev:
                yhat_prev = self.model.fittedvalues.values.tolist()
                err_prev = self.model.standardized_forecasts_error.tolist()
                forecast = np.concatenate((yhat_prev, forecast))
                err = np.concatenate((err_prev, err))
                t_prev = self.last_train_window.index.values.astype("datetime64[s]").astype(np.int64).tolist()
                time_stamps = t_prev + time_stamps

        # Check for NaN's
        if any(np.isnan(forecast)):
            logger.warning(
                "Trained ETS model is producing NaN forecast. Use the last training point as the prediction."
            )
            forecast[np.isnan(forecast)] = self._last_val
        if any(np.isnan(err)):
            err[np.isnan(err)] = 0

        # Return the forecast & its standard error
        name = self.target_name
        forecast = pd.DataFrame(forecast, index=to_pd_datetime(time_stamps), columns=[name])
        err = pd.DataFrame(err, index=to_pd_datetime(time_stamps), columns=[f"{name}_err"])
        return forecast, err
