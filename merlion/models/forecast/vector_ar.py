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
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as sm_VAR
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA

from merlion.models.forecast.base import ForecasterExogBase, ForecasterExogConfig
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import to_pd_datetime

logger = logging.getLogger(__name__)


class VectorARConfig(ForecasterExogConfig):
    """
    Config object for `VectorAR` forecaster.
    """

    _default_transform = TemporalResample()

    """
    Configuration class for Vector AutoRegressive model.
    """

    def __init__(self, maxlags: int = None, target_seq_index: int = None, **kwargs):
        """
        :param maxlags: Max # of lags for AR
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param maxlags: Max # of lags for AR
        """
        super().__init__(target_seq_index=target_seq_index, **kwargs)
        self.maxlags = maxlags


class VectorAR(ForecasterExogBase):
    """
    Vector AutoRegressive model for multivariate time series forecasting.
    """

    config_class = VectorARConfig

    def __init__(self, config: VectorARConfig):
        super().__init__(config)
        self.model = None
        self._pd_train_data = None

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def maxlags(self) -> int:
        return self.config.maxlags

    def _train_with_exog(
        self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.maxlags is None:
            self.config.maxlags = max(min(20, len(train_data) // 10), self.max_forecast_steps or 1)

        # train model
        if self.dim == 1:
            train_data = train_data.iloc[:, 0]
            self.model = sm_ARIMA(train_data, exog=exog_data, order=(self.maxlags, 0, 0))
            self.model = self.model.fit(method="yule_walker", cov_type="oim")
        else:
            self.model = sm_VAR(train_data, exog=exog_data).fit(self.maxlags)

        i = self.target_seq_index
        resid = self.model.resid
        pred = train_data - resid if self.dim == 1 else (train_data - resid).iloc[:, i]
        nanpred = pred.isna()
        if nanpred.any():
            pred[nanpred] = train_data.loc[nanpred, self.target_name]
        if self.dim == 1:
            pred_err = [np.sqrt(self.model.params["sigma2"]).item()] * len(pred)
        else:
            self._pd_train_data = train_data.iloc[-self.maxlags :]
            pred_err = [self.model.cov_ybar()[i, i].item()] * len(pred)

        pred = pd.DataFrame(pred, index=train_data.index, columns=[self.target_name])
        pred_err = pd.DataFrame(pred_err, index=train_data.index, columns=[f"{self.target_name}_err"])
        return pred, pred_err

    def _forecast_with_exog(
        self,
        time_stamps: List[int],
        time_series_prev: pd.DataFrame = None,
        return_prev=False,
        exog_data: pd.DataFrame = None,
        exog_data_prev: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if time_series_prev is not None:
            assert (
                len(time_series_prev) >= self.maxlags
            ), f"time_series_prev has length of {len(time_series_prev)}, which is shorter than the model's maxlags"
            assert not return_prev, "VectorAR.forecast() does not support return_prev=True"

        n = len(time_stamps)
        prev = self._pd_train_data if time_series_prev is None else time_series_prev.iloc[-self.maxlags :]
        exog_data_prev = None if exog_data_prev is None else exog_data_prev.loc[prev.index]
        if self.dim == 1:
            if time_series_prev is None:
                model = self.model
            else:
                model = self.model.apply(prev, exog=exog_data_prev, validate_specification=False)
            forecast_result = model.get_forecast(steps=n, exog=exog_data)
            yhat = forecast_result.predicted_mean
            err = forecast_result.se_mean
        else:
            old_exog = self.model.exog
            exog = None if exog_data is None else exog_data.values
            self.model.exog = old_exog if exog_data_prev is None else exog_data_prev.values
            yhat = self.model.forecast(prev.values, exog_future=exog, steps=n)[:, self.target_seq_index]
            err = np.sqrt(self.model.forecast_cov(n)[:, self.target_seq_index, self.target_seq_index])
            self.model.exog = old_exog

        name = self.target_name
        forecast = pd.DataFrame(np.asarray(yhat), index=to_pd_datetime(time_stamps), columns=[name])
        err = pd.DataFrame(np.asarray(err), index=forecast.index, columns=[f"{name}_err"])
        return forecast, err
