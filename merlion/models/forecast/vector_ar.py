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
            self.model = sm_ARIMA(train_data, order=(self.maxlags, 0, 0)).fit(method="yule_walker", cov_type="oim")
        else:
            self.model = sm_VAR(train_data).fit(self.maxlags)

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
        i = self.target_seq_index
        if self.dim == 1:
            if time_series_prev is None:
                model = self.model
            else:
                model = self.model.apply(time_series_prev.values[-self.maxlags :, 0], validate_specification=False)
            forecast_result = model.get_forecast(steps=n)
            yhat = forecast_result.predicted_mean
            err = forecast_result.se_mean
        else:
            prev = self._np_train_data if time_series_prev is None else time_series_prev.values[-self.maxlags :]
            yhat = self.model.forecast(prev, steps=n)[:, i]
            err = np.sqrt(self.model.forecast_cov(n)[:, i, i])

        name = self.target_name
        forecast = pd.DataFrame(np.asarray(yhat), index=to_pd_datetime(time_stamps), columns=[name])
        err = pd.DataFrame(np.asarray(err), index=to_pd_datetime(time_stamps), columns=[f"{name}_err"])
        return forecast, err
