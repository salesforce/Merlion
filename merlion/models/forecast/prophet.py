#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Wrapper around Facebook's popular Prophet model for time series forecasting.
"""
import copy
import logging
import os
from typing import Iterable, List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import prophet
import prophet.serialize

from merlion.models.automl.seasonality import SeasonalityModel
from merlion.models.forecast.base import ForecasterExogBase, ForecasterExogConfig
from merlion.utils import TimeSeries, UnivariateTimeSeries, to_pd_datetime, to_timestamp

logger = logging.getLogger(__name__)


class _suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    Source: https://github.com/facebook/prophet/issues/223#issuecomment-326455744
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class ProphetConfig(ForecasterExogConfig):
    """
    Configuration class for Facebook's `Prophet` model, as described by
    `Taylor & Letham, 2017 <https://peerj.com/preprints/3190/>`__.
    """

    def __init__(
        self,
        max_forecast_steps: int = None,
        target_seq_index: int = None,
        yearly_seasonality: Union[bool, int] = "auto",
        weekly_seasonality: Union[bool, int] = "auto",
        daily_seasonality: Union[bool, int] = "auto",
        seasonality_mode="additive",
        holidays=None,
        uncertainty_samples: int = 100,
        **kwargs,
    ):
        """
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
        :param seasonality_mode: 'additive' (default) or 'multiplicative'.
        :param holidays: pd.DataFrame with columns holiday (string) and ds (date type)
            and optionally columns lower_window and upper_window which specify a
            range of days around the date to be included as holidays.
            lower_window=-2 will include 2 days prior to the date as holidays. Also
            optionally can have a column prior_scale specifying the prior scale for
            that holiday. Can also be a dict corresponding to the desired pd.DataFrame.
        :param uncertainty_samples: The number of posterior samples to draw in
            order to calibrate the anomaly scores.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.uncertainty_samples = uncertainty_samples
        self.holidays = holidays


class Prophet(ForecasterExogBase, SeasonalityModel):
    """
    Facebook's model for time series forecasting. See docs for `ProphetConfig`
    and `Taylor & Letham, 2017 <https://peerj.com/preprints/3190/>`__ for more details.
    """

    config_class = ProphetConfig

    def __init__(self, config: ProphetConfig):
        super().__init__(config)
        self.model = prophet.Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            uncertainty_samples=self.uncertainty_samples,
            holidays=None if self.holidays is None else pd.DataFrame(self.holidays),
        )

    @property
    def require_even_sampling(self) -> bool:
        return False

    def __getstate__(self):
        try:
            model = prophet.serialize.model_to_json(self.model)
        except ValueError:  # prophet.serialize only works for fitted models, so deepcopy as a backup
            model = copy.deepcopy(self.model)
        return {k: model if k == "model" else copy.deepcopy(v) for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        if "model" in state:
            model = state["model"]
            if isinstance(model, str):
                state = copy.copy(state)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state["model"] = prophet.serialize.model_from_json(model)
        super().__setstate__(state)

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
    def seasonality_mode(self):
        return self.config.seasonality_mode

    @property
    def holidays(self):
        return self.config.holidays

    @property
    def uncertainty_samples(self):
        return self.config.uncertainty_samples

    def set_seasonality(self, theta, train_data: UnivariateTimeSeries):
        theta = [theta] if not isinstance(theta, Iterable) else theta
        dt = train_data.index[1] - train_data.index[0]
        for p in theta:
            if p > 1:
                period = p * dt.total_seconds() / 86400
                logger.debug(f"Add seasonality {str(p)} ({p * dt})")
                self.model.add_seasonality(name=f"extra_season_{p}", period=period, fourier_order=p)

    def _add_exog_data(self, data: pd.DataFrame, exog_data: pd.DataFrame):
        df = pd.DataFrame(data[self.target_name].rename("y"))
        if exog_data is not None:
            df = df.join(exog_data, how="outer")
        df.index.rename("ds", inplace=True)
        df.reset_index(inplace=True)
        return df

    def _train_with_exog(
        self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if exog_data is not None:
            for col in exog_data.columns:
                self.model.add_regressor(col)

        df = self._add_exog_data(train_data, exog_data)
        with _suppress_stdout_stderr():
            self.model.fit(df)

        # Get & return prediction & errors for train data.
        # sigma computation based on https://github.com/facebook/prophet/issues/549#issuecomment-435482584
        self.model.uncertainty_samples = 0
        forecast = self.model.predict(df)["yhat"].values.tolist()
        sigma = (self.model.params["sigma_obs"] * self.model.y_scale).item()
        self.model.uncertainty_samples = self.uncertainty_samples
        yhat = pd.DataFrame(forecast, index=df.ds, columns=[self.target_name])
        err = pd.DataFrame(sigma, index=df.ds, columns=[f"{self.target_name}_err"])
        return yhat, err

    def _forecast_with_exog(
        self,
        time_stamps: List[int],
        time_series_prev: pd.DataFrame = None,
        return_prev=False,
        exog_data: pd.DataFrame = None,
        exog_data_prev: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Construct data frame for prophet
        time_stamps = to_pd_datetime(time_stamps)
        df = self._add_exog_data(data=pd.DataFrame({self.target_name: np.nan}, index=time_stamps), exog_data=exog_data)
        if time_series_prev is not None:
            past = self._add_exog_data(time_series_prev, exog_data_prev)
            df = pd.concat((past, df))

        # Determine the right set of timestamps to use
        if return_prev and time_series_prev is not None:
            time_stamps = df["ds"]

        # Get MAP estimate from prophet
        self.model.uncertainty_samples = 0
        yhat = self.model.predict(df)["yhat"].values
        self.model.uncertainty_samples = self.uncertainty_samples

        # Get posterior samples for uncertainty estimation
        resid_samples = self.model.predictive_samples(df)["yhat"] - np.expand_dims(yhat, -1)

        # Return the MAP estimate & stderr
        yhat = yhat[-len(time_stamps) :]
        resid_samples = resid_samples[-len(time_stamps) :]
        name = self.target_name
        yhat = pd.DataFrame(yhat, index=time_stamps, columns=[name])
        err = pd.DataFrame(np.std(resid_samples, axis=-1), index=time_stamps, columns=[f"{name}_err"])
        return yhat, err
