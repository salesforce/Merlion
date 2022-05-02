#
# Copyright (c) 2022 salesforce.com, inc.
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
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.api import VAR as sm_VAR
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA

from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import to_pd_datetime, TimeSeries, UnivariateTimeSeries

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
        self._np_train_data = None

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def maxlags(self) -> int:
        return self.config.maxlags

    def _train(self, train_data: pd.DataFrame, train_config=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # train model
        if self.dim == 1:
            train_data = train_data.iloc[:, 0]
            self.model = sm_ARIMA(
                train_data,
                order=(self.maxlags, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                validate_specification=False,
            )
            self.model = self.model.fit(method="yule_walker", cov_type="oim")
        else:
            self.model = sm_VAR(train_data).fit(self.maxlags)

        # FORECASTING: forecast for next n steps using VAR model
        i = self.target_seq_index
        resid = self.model.resid
        pred = train_data - resid if self.dim == 1 else (train_data - resid).iloc[:, i]
        nanpred = pred.isna()
        if nanpred.any():
            pred[nanpred] = train_data.loc[nanpred, self.target_name]
        if self.dim == 1:
            pred_err = [np.sqrt(self.model.params["sigma2"]).item()] * len(pred)
        else:
            self._np_train_data = train_data.values[-self.maxlags :]
            pred_err = [self.model.cov_ybar()[i, i].item()] * len(pred)

        pred = pd.DataFrame(pred, index=train_data.index, columns=[self.target_name])
        pred_err = pd.DataFrame(pred_err, index=train_data.index, columns=[f"{self.target_name}_err"])
        return pred, pred_err

    def _forecast(
        self, time_stamps: Union[int, List[int]], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if time_series_prev is not None:
            assert len(time_series_prev) >= self.maxlags, (
                f"time_series_prev has a data length of "
                f"{len(time_series_prev)} that is shorter than the maxlags "
                f"for the model"
            )
            assert not return_prev, "VectorAR.forecast() does not support return_prev=True"

        n = len(time_stamps)
        if time_series_prev is None:
            if self.dim == 1:
                forecast_result = self.model.get_forecast(steps=n)
                yhat = forecast_result.predicted_mean
                err = forecast_result.se_mean
            else:
                i = self.target_seq_index
                yhat = self.model.forecast(self._np_train_data, steps=n)[:, i]
                err = np.sqrt(self.model.forecast_cov(n)[:, i, i])

        else:
            if self.dim == 1:
                new_state = self.model.apply(time_series_prev.values[-self.maxlags :, 0], validate_specification=False)
                forecast_result = new_state.get_forecast(steps=n)
                yhat = forecast_result.predicted_mean
                err = forecast_result.se_mean
            else:
                yhat = self.model.forecast(time_series_prev.values[-self.maxlags :], steps=n)
                yhat = yhat[:, self.target_seq_index]
                # Compute forecast covariance matrices for desired number of steps,
                # here we return the diagonal elements, i.e., variance (more rigorous math here?)
                err = np.sqrt(self.model.forecast_cov(n)[:, self.target_seq_index, self.target_seq_index])

        # Return the IQR (25%ile & 75%ile) along with the forecast if desired
        name = self.target_name
        forecast = pd.DataFrame(np.asarray(yhat), index=to_pd_datetime(time_stamps), columns=[name])
        err = pd.DataFrame(np.asarray(err), index=to_pd_datetime(time_stamps), columns=[f"{name}_err"])
        return forecast, err
