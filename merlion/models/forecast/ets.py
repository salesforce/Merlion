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
from merlion.utils import UnivariateTimeSeries, to_pd_datetime

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
        pred_interval_strategy="exact",
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
        :param pred_interval_strategy: Strategy to compute prediction intervals. "exact" or "simulated".
        Note that "simulated" setting supports more variants of ETS model.
        :param refit: if ``True``, refit the full ETS model when ``time_series_prev`` is given to the forecast method
            (slower). If ``False``, simply perform exponential smoothing (faster).
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.pred_interval_strategy = pred_interval_strategy
        self.refit = refit


class ETS(SeasonalityModel, ForecasterBase):
    """
    Implementation of the classic local statistical model ETS (Error, Trend, Seasonal) for forecasting.
    """

    config_class = ETSConfig

    def __init__(self, config: ETSConfig):
        super().__init__(config)
        self.model = None
        self._last_val = None

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

    @property
    def _max_lookback(self):
        if self.seasonal_periods is None:
            return 10
        return max(10, 10 + 2 * (self.seasonal_periods // 2), 2 * self.seasonal_periods)

    def _instantiate_model(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ETSModel(
                data,
                error=self.error,
                trend=self.trend,
                seasonal=None if self.seasonal_periods is None else self.seasonal,
                damped_trend=self.damped_trend,
                seasonal_periods=self.seasonal_periods,
            )

    def _train(self, train_data: pd.DataFrame, train_config=None):
        # train model
        name = self.target_name
        train_data = train_data[name]
        times = train_data.index
        self.model = self._instantiate_model(train_data).fit(disp=False)

        # get forecast for the training data
        self._last_val = train_data[-1]
        yhat = pd.DataFrame(self.model.fittedvalues.values.tolist(), index=times, columns=[name])
        err = pd.DataFrame(self.model.standardized_forecasts_error.tolist(), index=times, columns=[f"{name}_err"])
        return yhat, err

    def _forecast(
        self, time_stamps: Union[int, List[int]], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # If there is a time_series_prev, use it to set the ETS model's state, and then obtain its forecast
        time_stamps = to_pd_datetime(time_stamps)
        if time_series_prev is None:
            last_val = self._last_val
            model = self.model
        else:
            time_series_prev = time_series_prev.iloc[:, self.target_seq_index]
            val_prev = time_series_prev[-self._max_lookback :]
            last_val = val_prev[-1]

            # the default setting of refit=False is fast and conducts exponential smoothing with given parameters,
            # while the setting of refit=True is slow and refits the model on time_series_prev.
            model = self._instantiate_model(val_prev)
            if self.config.refit and len(time_series_prev) > self._max_lookback:
                model = model.fit(start_params=self.model.params, disp=False)
            else:
                model = model.smooth(params=self.model.params)

        # Run forecasting. Some variants of ETS model does not support prediction interval when
        # pred_interval_strategy="exact". In this case we use point forecasting and set prediction_interval as None.
        try:
            forecast_result = model.get_prediction(
                start=time_stamps[0], end=time_stamps[-1], method=self.config.pred_interval_strategy
            )
            forecast = np.asarray(forecast_result.predicted_mean)
            err = np.sqrt(np.asarray(forecast_result.var_pred_mean))
        except (NotImplementedError, AttributeError):
            forecast_result = model.predict(start=time_stamps[0], end=time_stamps[-1])
            forecast = np.asarray(forecast_result)
            err = None

        # If return_prev is True, it will return the forecast and error of last train window
        # instead of time_series_prev
        if time_series_prev is not None and return_prev:
            m = len(time_series_prev) - len(val_prev)
            params = dict(zip(model.param_names, model.params))
            err_prev = np.concatenate((np.zeros(m), model.standardized_forecasts_error.values))
            forecast = np.concatenate((time_series_prev.values[:m], model.fittedvalues.values, forecast))
            err = np.concatenate((err_prev, err))
            time_stamps = to_pd_datetime(np.concatenate((time_series_prev.index, time_stamps)))

        # Check for NaN's
        if any(np.isnan(forecast)):
            logger.warning("Trained ETS is producing NaN forecast. Use the last training point as the prediction.")
            forecast[np.isnan(forecast)] = last_val
        if err is not None and any(np.isnan(err)):
            err[np.isnan(err)] = 0

        # Return the forecast & its standard error
        name = self.target_name
        forecast = pd.DataFrame(forecast, index=time_stamps, columns=[name])
        if err is not None:
            err = pd.DataFrame(err, index=time_stamps, columns=[f"{name}_err"])
        return forecast, err
