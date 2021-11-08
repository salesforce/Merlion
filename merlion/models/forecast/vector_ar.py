#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Vector AutoRegressive model for multivariate time series forecasting.
"""
import logging
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.api import VAR as sm_VAR
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA

from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class VectorARConfig(ForecasterConfig):
    """
    Config object for `VectorAR` forecaster.
    """

    _default_transform = TemporalResample()

    """
    Configuration class for Vector AutoRegressive model.
    """

    def __init__(self, max_forecast_steps: int, maxlags: int, target_seq_index: int = None, **kwargs):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param maxlags: Max # of lags for AR
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param maxlags: Max # of lags for AR
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.maxlags = maxlags


class VectorAR(ForecasterBase):
    """
    Vector AutoRegressive model for multivariate time series forecasting.
    """

    config_class = VectorARConfig

    def __init__(self, config: VectorARConfig):
        super().__init__(config)
        self.model = None
        self._forecast = np.zeros(self.max_forecast_steps)
        self._forecast_err = np.ones(self.max_forecast_steps)
        self._input_already_transformed = False

    @property
    def maxlags(self) -> int:
        return self.config.maxlags

    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, TimeSeries]:
        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)

        i = self.target_seq_index
        times = train_data.univariates[self.target_name].time_stamps

        # train model
        df = train_data.to_pd()
        if self.dim == 1:
            df = df.iloc[:, 0]
            self.model = sm_ARIMA(
                df,
                order=(self.maxlags, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                validate_specification=False,
            )
            self.model = self.model.fit(method="yule_walker", cov_type="oim")
        else:
            self.model = sm_VAR(df).fit(self.maxlags)

        # FORECASTING: forecast for next n steps using VAR model
        n = self.max_forecast_steps
        resid = self.model.resid
        pred = df - resid if self.dim == 1 else (df - resid).iloc[:, i]
        if self.dim == 1:
            forecast_result = self.model.get_forecast(steps=n)
            forecast = forecast_result.predicted_mean
            err = forecast_result.se_mean
            pred_err = [np.sqrt(self.model.params["sigma2"])] * len(pred)
        else:
            forecast = self.model.forecast(df[-self.maxlags :].values, steps=n)[:, i]
            err = np.sqrt(self.model.forecast_cov(n)[:, i, i])
            pred_err = [self.model.cov_ybar()[i, i]] * len(pred)
        self._forecast = forecast
        self._forecast_err = err

        return (
            UnivariateTimeSeries(times, pred, self.target_name).to_ts(),
            UnivariateTimeSeries(times, pred_err, f"{self.target_name}_err").to_ts(),
        )

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr=False,
        return_prev=False,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, TimeSeries]]:

        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        t = self.resample_time_stamps(time_stamps, time_series_prev)

        if time_series_prev is not None:
            if not self._input_already_transformed:
                time_series_prev = self.transform(time_series_prev)
            # make sure training data agree with prediction data in the shape
            assert time_series_prev.dim == self.dim, (
                f"time_series_prev has multivariate dimension of "
                f"{time_series_prev.dim} that is different from "
                f"training data dimension of {self.dim} for the model"
            )
            assert len(time_series_prev) >= self.maxlags, (
                f"time_series_prev has a data length of "
                f"{len(time_series_prev)} that is shorter than the maxlags "
                f"for the model"
            )
            assert not return_prev, "VectorAR.forecast() does not support return_prev=True"

        if time_series_prev is None:
            yhat = self._forecast[: len(t)]
            err = self._forecast_err[: len(t)]

        else:
            df = time_series_prev.to_pd()
            if self.dim == 1:
                new_state = self.model.apply(df.iloc[-self.maxlags :, 0].values, validate_specification=False)
                forecast_result = new_state.get_forecast(steps=len(t))
                yhat = forecast_result.predicted_mean
                err = forecast_result.se_mean
            else:
                yhat = self.model.forecast(df[-self.maxlags :].values, steps=len(t))
                yhat = yhat[:, self.target_seq_index]
                # Compute forecast covariance matrices for desired number of steps,
                # here we return the diagonal elements, i.e., variance (more rigorous math here?)
                cov = self.model.forecast_cov(len(t))
                err = np.sqrt(cov[:, self.target_seq_index, self.target_seq_index])

        # Return the IQR (25%ile & 75%ile) along with the forecast if desired
        name = self.target_name
        if return_iqr:
            lb = (
                UnivariateTimeSeries(time_stamps=t, values=yhat + norm.ppf(0.25) * err, name=f"{name}_lower")
                .to_ts()
                .align(reference=orig_t)
            )
            ub = (
                UnivariateTimeSeries(time_stamps=t, values=yhat + norm.ppf(0.75) * err, name=f"{name}_upper")
                .to_ts()
                .align(reference=orig_t)
            )
            yhat = UnivariateTimeSeries(t, yhat, name).to_ts().align(reference=orig_t)
            return yhat, lb, ub

        # Otherwise, just return the forecast & its standard error
        else:
            forecast = UnivariateTimeSeries(name=name, time_stamps=t, values=yhat).to_ts().align(reference=orig_t)
            err = UnivariateTimeSeries(name=f"{name}_err", time_stamps=t, values=err).to_ts().align(reference=orig_t)
            return forecast, err

    def set_data_already_transformed(self):
        self._input_already_transformed = True

    def reset_data_already_transformed(self):
        self._input_already_transformed = False
