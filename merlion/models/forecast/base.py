#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for forecasting models.
"""
from abc import abstractmethod
import logging
from typing import List, Optional, Tuple, Union

import pandas as pd

from merlion.models.base import Config, ModelBase
from merlion.plot import Figure
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries

logger = logging.getLogger(__name__)


class ForecasterConfig(Config):
    """
    Config object used to define a forecaster model.
    """

    max_forecast_steps: Optional[int] = None
    target_seq_index: Optional[int] = None

    def __init__(self, max_forecast_steps: int = None, target_seq_index: int = None, **kwargs):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.
            Required for some models like `MSES` and `LGBMForecaster`.
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        """
        super().__init__(**kwargs)
        self.max_forecast_steps = max_forecast_steps
        self.target_seq_index = target_seq_index


class ForecasterBase(ModelBase):
    """
    Base class for a forecaster model.

    .. note::

        If your model depends on an evenly spaced time series, make sure to

        1. Call `ForecasterBase.train_pre_process` in `ForecasterBase.train`
        2. Call `ForecasterBase.resample_time_stamps` at the start of
           `ForecasterBase.forecast` to get a set of resampled time stamps, and
           call ``time_series.align(reference=time_stamps)`` to align the forecast
           with the original time stamps.
    """

    config_class = ForecasterConfig
    target_name = None
    """
    The name of the target univariate to forecast.
    """

    def __init__(self, config: ForecasterConfig):
        super().__init__(config)
        self.target_name = None

    @property
    def max_forecast_steps(self):
        return self.config.max_forecast_steps

    @property
    def target_seq_index(self) -> int:
        """
        :return: the index of the univariate (amongst all univariates in a
            general multivariate time series) whose value we would like to forecast.
        """
        return self.config.target_seq_index

    def resample_time_stamps(self, time_stamps: Union[int, List[int]], time_series_prev: TimeSeries = None):
        assert self.timedelta is not None and self.last_train_time is not None, (
            "train() must be called before you can call forecast(). "
            "If you have already called train(), make sure it sets "
            "self.timedelta and self.last_train_time appropriately."
        )

        # Determine timedelta & initial time of forecast
        dt = self.timedelta
        if time_series_prev is not None and not time_series_prev.is_empty():
            t0 = to_pd_datetime(time_series_prev.tf)
        else:
            t0 = self.last_train_time

        # Handle the case where time_stamps is an integer
        if isinstance(time_stamps, (int, float)):
            n = int(time_stamps)
            assert self.max_forecast_steps is None or n <= self.max_forecast_steps
            resampled = pd.date_range(start=t0, periods=n + 1, freq=dt)[1:]
            tf = resampled[-1]
            time_stamps = to_timestamp(resampled)

        # Handle the cases where we don't have a max_forecast_steps
        elif self.max_forecast_steps is None:
            tf = to_pd_datetime(time_stamps[-1])
            resampled = pd.date_range(start=t0, end=tf, freq=dt)[1:]
            if resampled[-1] < tf:
                extra = pd.date_range(start=resampled[-1], periods=2, freq=dt)[1:]
                resampled = resampled.union(extra)

        # Handle the case where we do have a max_forecast_steps
        else:
            resampled = pd.date_range(start=t0, periods=self.max_forecast_steps + 1, freq=dt)[1:]
            tf = resampled[-1]
            n = sum(t < to_pd_datetime(time_stamps[-1]) for t in resampled)
            resampled = resampled[: n + 1]

        assert to_pd_datetime(time_stamps[0]) >= t0 and to_pd_datetime(time_stamps[-1]) <= tf, (
            f"Expected `time_stamps` to be between {t0} and {tf}, but `time_stamps` ranges "
            f"from {to_pd_datetime(time_stamps[0])} to {to_pd_datetime(time_stamps[-1])}"
        )

        return to_timestamp(resampled).tolist()

    def train_pre_process(
        self, train_data: TimeSeries, require_even_sampling: bool, require_univariate: bool
    ) -> TimeSeries:
        train_data = super().train_pre_process(train_data, require_even_sampling, require_univariate)
        if self.dim == 1:
            self.config.target_seq_index = 0
        elif self.target_seq_index is None:
            raise RuntimeError(
                f"Attempting to use a forecaster on a {train_data.dim}-variable "
                f"time series, but didn't specify a `target_seq_index` "
                f"indicating which univariate is the target."
            )
        assert 0 <= self.target_seq_index < train_data.dim, (
            f"Expected `target_seq_index` to be between 0 and {train_data.dim} "
            f"(the dimension of the transformed data), but got {self.target_seq_index}"
        )
        self.target_name = train_data.names[self.target_seq_index]

        return train_data

    @abstractmethod
    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        """
        Trains the forecaster on the input time series.

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param train_config: Additional training configs, if needed. Only
            required for some models.

        :return: the model's prediction on ``train_data``, in the same format as
            if you called `ForecasterBase.forecast` on the time stamps of
            ``train_data``
        """
        raise NotImplementedError

    @abstractmethod
    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Union[Tuple[TimeSeries, Optional[TimeSeries]], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        """
        Returns the model's forecast on the timestamps given. Note that if
        ``self.transform`` is specified in the config, the forecast is a forecast
        of transformed values! It is up to you to manually invert the transform
        if desired.

        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for,
            or the number of steps (``int``) we wish to forecast for.
        :param time_series_prev: a list of (timestamp, value) pairs immediately
            preceding ``time_series``. If given, we use it to initialize the time
            series model. Otherwise, we assume that ``time_series`` immediately
            follows the training data.
        :param return_iqr: whether to return the inter-quartile range for the
            forecast. Note that not all models support this option.
        :param return_prev: whether to return the forecast for
            ``time_series_prev`` (and its stderr or IQR if relevant), in addition
            to the forecast for ``time_stamps``. Only used if ``time_series_prev``
            is provided.
        :return: ``(forecast, forecast_stderr)`` if ``return_iqr`` is false,
            ``(forecast, forecast_lb, forecast_ub)`` otherwise.

            - ``forecast``: the forecast for the timestamps given
            - ``forecast_stderr``: the standard error of each forecast value.
                May be ``None``.
            - ``forecast_lb``: 25th percentile of forecast values for each timestamp
            - ``forecast_ub``: 75th percentile of forecast values for each timestamp
        """
        raise NotImplementedError

    def batch_forecast(
        self,
        time_stamps_list: List[List[int]],
        time_series_prev_list: List[TimeSeries],
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Tuple[
        Union[
            Tuple[List[TimeSeries], List[Optional[TimeSeries]]],
            Tuple[List[TimeSeries], List[TimeSeries], List[TimeSeries]],
        ]
    ]:
        """
        Returns the model's forecast on a batch of timestamps given. Note that if
        ``self.transform`` is specified in the config, the forecast is a forecast
        of transformed values! It is up to you to manually invert the transform
        if desired.

        :param time_stamps_list: a list of lists of timestamps we wish to forecast for
        :param time_series_prev_list: a list of TimeSeries immediately preceding the time stamps in time_stamps_list
        :param return_iqr: whether to return the inter-quartile range for the
            forecast. Note that not all models support this option.
        :param return_prev: whether to return the forecast for
            ``time_series_prev`` (and its stderr or IQR if relevant), in addition
            to the forecast for ``time_stamps``. Only used if ``time_series_prev``
            is provided.
        :return: ``(forecast, forecast_stderr)`` if ``return_iqr`` is false,
            ``(forecast, forecast_lb, forecast_ub)`` otherwise.

            - ``forecast``: the forecast for the timestamps given
            - ``forecast_stderr``: the standard error of each forecast value.
                May be ``None``.
            - ``forecast_lb``: 25th percentile of forecast values for each timestamp
            - ``forecast_ub``: 75th percentile of forecast values for each timestamp
        """
        out_list = []
        if time_series_prev_list is None:
            time_series_prev_list = [None for _ in range(len(time_stamps_list))]
        for time_stamps, time_series_prev in zip(time_stamps_list, time_series_prev_list):
            out = self.forecast(time_stamps, time_series_prev, return_iqr, return_prev)
            out_list.append(out)
        return tuple(zip(*out_list))

    def invert_transform(self, forecast: TimeSeries, time_series_prev: TimeSeries = None):
        if time_series_prev is None:
            time_series_prev = self.transform(self.train_data)
        t = forecast.np_time_stamps
        return self.transform.invert(time_series_prev + forecast).align(reference=t)

    def get_figure(
        self,
        *,
        time_series: TimeSeries = None,
        time_stamps: List[int] = None,
        time_series_prev: TimeSeries = None,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
    ) -> Figure:
        """
        :param time_series: the time series over whose timestamps we wish to
            make a forecast. Exactly one of ``time_series`` or ``time_stamps``
            should be provided.
        :param time_stamps: a list of timestamps we wish to forecast for. Exactly
            one of ``time_series`` or ``time_stamps`` should be provided.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_stamps``. If given, we use it to initialize the time series
            model. Otherwise, we assume that ``time_stamps`` immediately follows
            the training data.
        :param plot_forecast_uncertainty: whether to plot uncertainty estimates (the
            inter-quartile range) for forecast values. Not supported for all
            models.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :return: a `Figure` of the model's forecast.
        """
        assert not (
            time_series is None and time_stamps is None
        ), "Must provide at least one of time_series or time_stamps"
        if time_stamps is None:
            transformed_ts = self.transform(time_series)
            time_stamps = [t for t, y in transformed_ts]
            y = transformed_ts.univariates[transformed_ts.names[self.target_seq_index]]
        else:
            y = None

        # Get forecast + bounds if plotting uncertainty
        if plot_forecast_uncertainty:
            yhat, lb, ub = self.forecast(
                time_stamps, time_series_prev, return_iqr=True, return_prev=plot_time_series_prev
            )
            yhat, lb, ub = [x.univariates[x.names[0]] for x in [yhat, lb, ub]]

        # Just get the forecast otherwise
        else:
            lb, ub = None, None
            yhat, err = self.forecast(
                time_stamps, time_series_prev, return_iqr=False, return_prev=plot_time_series_prev
            )
            yhat = yhat.univariates[yhat.names[0]]

        # Set up all the parameters needed to make a figure
        if time_series_prev is not None and plot_time_series_prev:
            time_series_prev = self.transform(time_series_prev)
            assert time_series_prev.dim == 1, (
                f"Plotting only supported for univariate time series, but got a"
                f"time series of dimension {time_series_prev.dim}"
            )
            time_series_prev = time_series_prev.univariates[time_series_prev.names[0]]

            n_prev = len(time_series_prev)
            yhat_prev, yhat = yhat[:n_prev], yhat[n_prev:]
            if lb is not None and ub is not None:
                lb_prev, lb = lb[:n_prev], lb[n_prev:]
                ub_prev, ub = ub[:n_prev], ub[n_prev:]
            else:
                lb_prev = ub_prev = None
        else:
            time_series_prev = None
            yhat_prev = lb_prev = ub_prev = None

        # Create the figure
        return Figure(
            y=y,
            yhat=yhat,
            yhat_lb=lb,
            yhat_ub=ub,
            y_prev=time_series_prev,
            yhat_prev=yhat_prev,
            yhat_prev_lb=lb_prev,
            yhat_prev_ub=ub_prev,
        )

    def plot_forecast(
        self,
        *,
        time_series: TimeSeries = None,
        time_stamps: List[int] = None,
        time_series_prev: TimeSeries = None,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
        figsize=(1000, 600),
        ax=None,
    ):
        """
        Plots the forecast for the time series in matplotlib, optionally also
        plotting the uncertainty of the forecast, as well as the past values
        (both true and predicted) of the time series.

        :param time_series: the time series over whose timestamps we wish to
            make a forecast. Exactly one of ``time_series`` or ``time_stamps``
            should be provided.
        :param time_stamps: a list of timestamps we wish to forecast for. Exactly
            one of ``time_series`` or ``time_stamps`` should be provided.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_stamps``. If given, we use it to initialize the time series
            model. Otherwise, we assume that ``time_stamps`` immediately follows
            the training data.
        :param plot_forecast_uncertainty: whether to plot uncertainty estimates (the
            inter-quartile range) for forecast values. Not supported for all
            models.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param figsize: figure size in pixels
        :param ax: matplotlib axis to add this plot to

        :return: (fig, ax): matplotlib figure & axes the figure was plotted on
        """
        fig = self.get_figure(
            time_series=time_series,
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
        )
        title = f"{type(self).__name__}: Forecast of {self.target_name}"
        return fig.plot(title=title, metric_name=self.target_name, figsize=figsize, ax=ax)

    def plot_forecast_plotly(
        self,
        *,
        time_series: TimeSeries = None,
        time_stamps: List[int] = None,
        time_series_prev: TimeSeries = None,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
        figsize=(1000, 600),
    ):
        """
        Plots the forecast for the time series in plotly, optionally also
        plotting the uncertainty of the forecast, as well as the past values
        (both true and predicted) of the time series.

        :param time_series: the time series over whose timestamps we wish to
            make a forecast. Exactly one of ``time_series`` or ``time_stamps``
            should be provided.
        :param time_stamps: a list of timestamps we wish to forecast for. Exactly
            one of ``time_series`` or ``time_stamps`` should be provided.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_stamps``. If given, we use it to initialize the time series
            model. Otherwise, we assume that ``time_stamps`` immediately follows
            the training data.
        :param plot_forecast_uncertainty: whether to plot uncertainty estimates (the
            inter-quartile range) for forecast values. Not supported for all
            models.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param figsize: figure size in pixels
        """
        fig = self.get_figure(
            time_series=time_series,
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
        )
        title = f"{type(self).__name__}: Forecast of {self.target_name}"
        return fig.plot_plotly(title=title, metric_name=self.target_name, figsize=figsize)
