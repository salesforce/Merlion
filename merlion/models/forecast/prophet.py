#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Wrapper around Facebook's popular Prophet model for time series forecasting.
"""
import logging
from typing import List, Tuple, Union

import fbprophet
import numpy as np
import pandas as pd

from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.utils import TimeSeries, UnivariateTimeSeries, to_pd_datetime, autosarima_utils

logger = logging.getLogger(__name__)


class ProphetConfig(ForecasterConfig):
    def __init__(
        self,
        max_forecast_steps: int = None,
        target_seq_index: int = None,
        yearly_seasonality: Union[bool, int] = "auto",
        weekly_seasonality: Union[bool, int] = "auto",
        daily_seasonality: Union[bool, int] = "auto",
        add_seasonality="auto",
        uncertainty_samples: int = 100,
        **kwargs,
    ):
        """
        Configuration class for Facebook's Prophet model, as described in this
        `paper <https://peerj.com/preprints/3190/>`_.

        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param yearly_seasonality: If bool, whether to enable yearly seasonality.
            By default, it is activated if there are >= 2 years of history, but
            deactivated otherwise. If int, this is the number of Fourier series
            components used to model the seasonality (default = 10).
        :param weekly_seasonality: If bool, whether to enable weekly seasonality.
            By default, it is activated if there are >= 2 weeks of history, but
            deactivated otherwise. If int, this is the number of Fourier series
            components used to model the seasonality (default = 3).
        :param daily_seasonality: If bool, whether to enable daily seasonality.
            By default, it is activated if there are >= 2 days of history, but
            deactivated otherwise. If int, this is the number of Fourier series
            components used to model the seasonality (default = 4).
        :param add_seasonality: 'auto' indicates automatically adding extra
            seasonaltiy by detection methods (default = None).
        :param uncertainty_samples: The number of posterior samples to draw in
            order to calibrate the anomaly scores.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.add_seasonality = add_seasonality
        self.uncertainty_samples = uncertainty_samples


class Prophet(ForecasterBase):
    """
    Facebook's model for time series forecasting. See docs for `ProphetConfig`
    and the `paper <https://peerj.com/preprints/3190/>`_ for more details.
    """

    config_class = ProphetConfig

    def __init__(self, config: ProphetConfig):
        super().__init__(config)
        self.model = fbprophet.Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            uncertainty_samples=self.uncertainty_samples,
        )
        self.last_forecast_time_stamps_full = None
        self.last_forecast_time_stamps = None
        self.resid_samples = None

    def __getstate__(self):
        stan_backend = self.model.stan_backend
        if hasattr(stan_backend, "logger"):
            model_logger = self.model.stan_backend.logger
            self.model.stan_backend.logger = None
        state_dict = super().__getstate__()
        if hasattr(stan_backend, "logger"):
            self.model.stan_backend.logger = model_logger
        return state_dict

    @property
    def yearly_seasonality(self):
        return self.config.yearly_seasonality

    @property
    def weekly_seasonality(self):
        return self.config.weekly_seasonality

    @property
    def daily_seasonality(self):
        return self.config.daily_seasonality

    @property
    def add_seasonality(self):
        return self.config.add_seasonality

    @property
    def uncertainty_samples(self):
        return self.config.uncertainty_samples

    def train(self, train_data: TimeSeries, train_config=None):
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)
        series = train_data.univariates[self.target_name]
        df = pd.DataFrame({"ds": series.index, "y": series.np_values})

        if self.add_seasonality == "auto":
            periods = autosarima_utils.multiperiodicity_detection(series.np_values)
            if len(periods) > 0:
                max_periodicty = periods[-1]
            else:
                max_periodicty = 0
            if max_periodicty > 1:
                logger.info(f"Add seasonality {str(max_periodicty)}")
                self.model.add_seasonality(
                    name="extra_season", period=max_periodicty * self.timedelta / 86400, fourier_order=max_periodicty
                )

        self.model.fit(df)

        # Get & return prediction & errors for train data
        self.model.uncertainty_samples = 0
        forecast = self.model.predict(df)["yhat"].values.tolist()
        self.model.uncertainty_samples = self.uncertainty_samples
        samples = self.model.predictive_samples(df)["yhat"]
        samples = samples - np.expand_dims(forecast, -1)

        yhat = UnivariateTimeSeries(df.ds, forecast, self.target_name).to_ts()
        err = UnivariateTimeSeries(df.ds, np.std(samples, axis=-1), f"{self.target_name}_err").to_ts()
        return yhat, err

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr=False,
        return_prev=False,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        if isinstance(time_stamps, (int, float)):
            time_stamps = [self.last_train_time + (i + 1) * self.timedelta for i in range(int(time_stamps))]
        times = to_pd_datetime(time_stamps)

        # Construct data frame for fbprophet
        df = pd.DataFrame()
        if time_series_prev is not None:
            series = self.transform(time_series_prev)
            series = series.univariates[series.names[self.target_seq_index]]
            df = pd.DataFrame({"ds": series.index, "y": series.np_values})
        df = df.append(pd.DataFrame({"ds": times}))

        # Get MAP estimate from fbprophet
        self.model.uncertainty_samples = 0
        yhat = self.model.predict(df)["yhat"].values
        self.model.uncertainty_samples = self.uncertainty_samples

        # Use posterior sampling get the uncertainty for this forecast
        if time_series_prev is not None:
            time_stamps_full = time_series_prev.time_stamps + time_stamps
        else:
            time_stamps_full = time_stamps

        if self.last_forecast_time_stamps_full != time_stamps_full:
            samples = self.model.predictive_samples(df)["yhat"]
            self.last_forecast_time_stamps_full = time_stamps_full
            if self.last_forecast_time_stamps != time_stamps:
                self.resid_samples = samples - np.expand_dims(yhat, -1)
                self.last_forecast_time_stamps = time_stamps
            else:
                n = len(time_stamps)
                prev = samples[:-n] - np.expand_dims(yhat[:-n], -1)
                self.resid_samples = np.concatenate((prev, self.resid_samples))

        if not return_prev:
            yhat = yhat[-len(time_stamps) :]
            t = time_stamps
        else:
            t = time_stamps_full

        t = t[-len(yhat) :]
        samples = self.resid_samples[-len(yhat) :]
        name = self.target_name
        if return_iqr:
            lb = UnivariateTimeSeries(
                name=f"{name}_lower", time_stamps=t, values=yhat + np.percentile(samples, 25, axis=-1)
            ).to_ts()
            ub = UnivariateTimeSeries(
                name=f"{name}_upper", time_stamps=t, values=yhat + np.percentile(samples, 75, axis=-1)
            ).to_ts()
            yhat = UnivariateTimeSeries(t, yhat, name).to_ts()
            return yhat, ub, lb
        else:
            yhat = UnivariateTimeSeries(t, yhat, name).to_ts()
            err = UnivariateTimeSeries(t, np.std(samples, axis=-1), f"{name}_err").to_ts()
            return yhat, err
