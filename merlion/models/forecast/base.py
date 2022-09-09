#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for forecasting models.
"""
from abc import abstractmethod
import copy
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from merlion.models.base import Config, ModelBase
from merlion.plot import Figure
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, MissingValuePolicy

logger = logging.getLogger(__name__)


class ForecasterConfig(Config):
    """
    Config object used to define a forecaster model.
    """

    max_forecast_steps: Optional[int] = None
    target_seq_index: Optional[int] = None
    invert_transform: bool = False

    def __init__(self, max_forecast_steps: int = None, target_seq_index: int = None, invert_transform=False, **kwargs):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.  Required for some models like `MSES`.
        :param target_seq_index: The index of the univariate (amongst all univariates in a general multivariate time
            series) whose value we would like to forecast.
        :param invert_transform: Whether to automatically invert the ``transform`` before returning a forecast.
        """
        super().__init__(**kwargs)
        self.max_forecast_steps = max_forecast_steps
        self.target_seq_index = target_seq_index
        self.invert_transform = invert_transform


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

    @property
    def invert_transform(self):
        """
        :return: Whether to automatically invert the ``transform`` before returning a forecast.
        """
        return self.config.invert_transform

    @property
    def require_univariate(self) -> bool:
        """
        All forecasters can work on multivariate data, since they only forecast a single target univariate.
        """
        return False

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

        elif not self.require_even_sampling:
            resampled = to_pd_datetime(time_stamps)
            tf = resampled[-1]

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

    def train_pre_process(self, train_data: TimeSeries) -> TimeSeries:
        train_data = super().train_pre_process(train_data)
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

    def train(
        self, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None
    ) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        """
        Trains the forecaster on the input time series.

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param train_config: Additional training configs, if needed. Only required for some models.
        :param exog_data: A time series of exogenous variables, sampled at the same time stamps as ``train_data``.
            Exogenous variables are known a priori, and they are independent of the variable being forecasted.
            Only supported for models which inherit from `ForecasterWithExog`.

        :return: the model's prediction on ``train_data``, in the same format as
            if you called `ForecasterBase.forecast` on the time stamps of ``train_data``
        """
        if train_config is None:
            train_config = copy.deepcopy(self._default_train_config)
        train_data = self.train_pre_process(train_data).to_pd()
        exog_data, _ = self.resample_exog_data(exog_data=exog_data, time_stamps=train_data.index)
        if exog_data is None:
            train_result = self._train(train_data, train_config=train_config)
        else:
            train_result = self._train_with_exog(train_data, train_config=train_config, exog_data=exog_data.to_pd())
        return self.train_post_process(train_result)

    def train_post_process(
        self, train_result: Tuple[Union[TimeSeries, pd.DataFrame], Optional[Union[TimeSeries, pd.DataFrame]]]
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Converts the train result (forecast & stderr for training data) into TimeSeries objects, and inverts the
        model's transform if desired.
        """
        train_pred, train_stderr = [TimeSeries.from_pd(df) for df in train_result]
        if self.invert_transform:
            train_pred, train_stderr = self._apply_inverse_transform(train_pred, train_stderr)
        return train_pred, train_stderr

    @abstractmethod
    def _train(self, train_data: pd.DataFrame, train_config=None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        raise NotImplementedError

    def _train_with_exog(self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None):
        raise NotImplementedError

    def resample_exog_data(
        self,
        exog_data: TimeSeries,
        time_stamps: Union[List[int], pd.DatetimeIndex],
        time_series_prev: TimeSeries = None,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, None], Tuple[None, None]]:
        if exog_data is not None:
            logger.warning(f"Exogenous regressors are not supported for model {type(self).__name__}")
        return None, None

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        exog_data: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Union[Tuple[TimeSeries, Optional[TimeSeries]], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        """
        Returns the model's forecast on the timestamps given. If ``self.transform`` is specified in the config, the
        forecast is a forecast of transformed values by default. To invert the transform and forecast the actual
        values of the time series, specify ``invert_transform = True`` when specifying the config.

        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for, or the number of steps (``int``)
            we wish to forecast for.
        :param time_series_prev: a time series immediately preceding ``time_series``. If given, we use it to initialize
            the forecaster's state. Otherwise, we assume that ``time_series`` immediately follows the training data.
        :param exog_data: A time series of exogenous variables. Exogenous variables are known a priori, and they are
            independent of the variable being forecasted. ``exog_data`` must include data for all of ``time_stamps``;
            if ``time_series_prev`` is given, it must include data for all of ``time_series_prev.time_stamps`` as well.
            Optional. Only supported for models which inherit from `ForecasterWithExog`.
        :param return_iqr: whether to return the inter-quartile range for the forecast.
            Only supported for models which return error bars.
        :param return_prev: whether to return the forecast for ``time_series_prev`` (and its stderr or IQR if relevant),
            in addition to the forecast for ``time_stamps``. Only used if ``time_series_prev`` is provided.
        :return: ``(forecast, stderr)`` if ``return_iqr`` is false, ``(forecast, lb, ub)`` otherwise.

            - ``forecast``: the forecast for the timestamps given
            - ``stderr``: the standard error of each forecast value. May be ``None``.
            - ``lb``: 25th percentile of forecast values for each timestamp
            - ``ub``: 75th percentile of forecast values for each timestamp
        """
        # determine the time stamps to forecast for, and resample them if needed
        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)
        if return_prev and time_series_prev is not None:
            if orig_t is None:
                orig_t = time_series_prev.time_stamps + time_stamps
            else:
                orig_t = time_series_prev.time_stamps + to_timestamp(orig_t).tolist()

        # transform time_series_prev if relevant (before making the prediction)
        old_inversion_state = self.transform.inversion_state
        if time_series_prev is None:
            time_series_prev_df = None
        else:
            time_series_prev = self.transform(time_series_prev)
            assert time_series_prev.dim == self.dim, (
                f"time_series_prev has dimension of {time_series_prev.dim} that is different from "
                f"training data dimension of {self.dim} for the model"
            )
            time_series_prev_df = time_series_prev.to_pd()

        # Make the prediction
        exog_data, exog_data_prev = self.resample_exog_data(
            exog_data, time_stamps=time_stamps, time_series_prev=time_series_prev
        )
        if exog_data is None:
            forecast, err = self._forecast(
                time_stamps=time_stamps, time_series_prev=time_series_prev_df, return_prev=return_prev
            )
        else:
            forecast, err = self._forecast_with_exog(
                time_stamps=time_stamps,
                time_series_prev=time_series_prev_df,
                return_prev=return_prev,
                exog_data=exog_data.to_pd(),
                exog_data_prev=None if exog_data_prev is None else exog_data_prev.to_pd(),
            )

        # Format the return value(s)
        if self.invert_transform and time_series_prev is None:
            time_series_prev = self.transform(self.train_data)
        if time_series_prev is not None:
            time_series_prev = time_series_prev.univariates[time_series_prev.names[self.target_seq_index]].to_ts()

        # Handle the case where we want to return the IQR. If applying the inverse transform, we just apply
        # the inverse transform directly to the upper/lower bounds.
        if return_iqr:
            # Compute positive & negative deviations. Case 1 is where we return distinct upper & lower errors.
            if err is None:
                logger.warning("Model returned err = None, so returning IQR = (None, None)")
                d_neg, d_pos = None, None
            elif isinstance(err, tuple) and len(err) == 2:
                d_neg, d_pos = err[0].values * norm.ppf(0.25), err[1].values * norm.ppf(0.75)
            else:
                d_neg, d_pos = err.values * norm.ppf(0.25), err.values * norm.ppf(0.75)

            # Concatenate time_series_prev to the forecast & upper/lower bounds if inverting the transform
            if self.invert_transform:
                time_series_prev_df = time_series_prev.to_pd()
                if d_neg is not None and d_pos is not None:
                    d_neg = np.concatenate((np.zeros((len(time_series_prev_df), d_neg.shape[1])), d_neg))
                    d_pos = np.concatenate((np.zeros((len(time_series_prev_df), d_neg.shape[1])), d_pos))
                forecast = pd.concat((time_series_prev_df, forecast))

            # Convert to time series & invert the transform if desired
            if d_neg is None or d_pos is None:
                lb, ub = None, None
            else:
                lb = TimeSeries.from_pd((forecast + d_neg).rename(columns=lambda c: f"{c}_lower"))
                ub = TimeSeries.from_pd((forecast + d_pos).rename(columns=lambda c: f"{c}_upper"))
            forecast = TimeSeries.from_pd(forecast)
            if self.invert_transform:
                forecast = self.transform.invert(forecast, retain_inversion_state=True)
                if lb is not None and ub is not None:
                    lb = self.transform.invert(lb, retain_inversion_state=True)
                    ub = self.transform.invert(ub, retain_inversion_state=True)
            ret = forecast, lb, ub

        # Handle the case where we directly return the forecast and its standard error.
        # If applying the inverse transform, we compute an upper/lower bound, apply the inverse transform to those
        # bounds, and use the difference of those bounds as the stderr.
        else:
            if isinstance(err, tuple) and len(err) == 2:
                err = (err[0].abs().values + err[1].abs().values) / 2
                err = pd.DataFrame(err, index=forecast.index, columns=[f"{c}_err" for c in forecast.columns])
            forecast = TimeSeries.from_pd(forecast)
            err = None if err is None else TimeSeries.from_pd(err)
            ret = forecast, err
            if self.invert_transform:
                ret = self._apply_inverse_transform(forecast, err, None if return_prev else time_series_prev)

        self.transform.inversion_state = old_inversion_state
        return tuple(None if x is None else x.align(reference=orig_t) for x in ret)

    def _apply_inverse_transform(self, forecast, err, time_series_prev=None):
        forecast = forecast if time_series_prev is None else time_series_prev + forecast

        if err is not None:
            forecast_df, err_df = forecast.to_pd(), err.to_pd()
            n = len(time_series_prev) if time_series_prev is not None else 0
            if n > 0:
                zeros = pd.DataFrame(np.zeros((n, err.dim)), index=forecast_df.index[:n], columns=err_df.columns)
                err_df = pd.concat((zeros, err_df))
            lb = TimeSeries.from_pd(forecast_df.values - err_df)
            ub = TimeSeries.from_pd(forecast_df.values + err_df)
            lb = self.transform.invert(lb, retain_inversion_state=True)
            ub = self.transform.invert(ub, retain_inversion_state=True)
            err = TimeSeries.from_pd((ub.to_pd() - lb.to_pd()).abs() / 2)

        forecast = self.transform.invert(forecast, retain_inversion_state=True)
        return forecast, err

    @abstractmethod
    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, Union[None, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]:
        raise NotImplementedError

    def _forecast_with_exog(
        self,
        time_stamps: List[int],
        time_series_prev: pd.DataFrame = None,
        return_prev=False,
        exog_data: pd.DataFrame = None,
        exog_data_prev: pd.DataFrame = None,
    ):
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
            out = self.forecast(
                time_stamps=time_stamps,
                time_series_prev=time_series_prev,
                return_iqr=return_iqr,
                return_prev=return_prev,
            )
            out_list.append(out)
        return tuple(zip(*out_list))

    def get_figure(
        self,
        *,
        time_series: TimeSeries = None,
        time_stamps: List[int] = None,
        time_series_prev: TimeSeries = None,
        exog_data: TimeSeries = None,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
    ) -> Figure:
        """
        :param time_series: the time series over whose timestamps we wish to make a forecast. Exactly one of
            ``time_series`` or ``time_stamps`` should be provided.
        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for, or the number of steps (``int``)
            we wish to forecast for. Exactly one of ``time_series`` or ``time_stamps`` should be provided.
        :param time_series_prev: a time series immediately preceding ``time_series``. If given, we use it to initialize
            the forecaster's state. Otherwise, we assume that ``time_series`` immediately follows the training data.
        :param exog_data: A time series of exogenous variables. Exogenous variables are known a priori, and they are
            independent of the variable being forecasted. ``exog_data`` must include data for all of ``time_stamps``;
            if ``time_series_prev`` is given, it must include data for all of ``time_series_prev.time_stamps`` as well.
            Optional. Only supported for models which inherit from `ForecasterWithExog`.
        :param plot_forecast_uncertainty: whether to plot uncertainty estimates (the inter-quartile range) for forecast
            values. Not supported for all  models.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and  the model's fit for it).
            Only used if ``time_series_prev`` is given.
        :return: a `Figure` of the model's forecast.
        """
        assert not (
            time_series is None and time_stamps is None
        ), "Must provide at least one of time_series or time_stamps"
        if time_stamps is None:
            if self.invert_transform:
                time_stamps = time_series.time_stamps
                y = time_series.univariates[time_series.names[self.target_seq_index]]
            else:
                transformed_ts = self.transform(time_series)
                time_stamps = transformed_ts.time_stamps
                y = transformed_ts.univariates[transformed_ts.names[self.target_seq_index]]
        else:
            y = None

        # Get forecast + bounds if plotting uncertainty
        if plot_forecast_uncertainty:
            yhat, lb, ub = self.forecast(
                time_stamps, time_series_prev, exog_data=exog_data, return_iqr=True, return_prev=plot_time_series_prev
            )
            yhat, lb, ub = [None if x is None else x.univariates[x.names[0]] for x in [yhat, lb, ub]]

        # Just get the forecast otherwise
        else:
            lb, ub = None, None
            yhat, err = self.forecast(
                time_stamps, time_series_prev, exog_data=exog_data, return_iqr=False, return_prev=plot_time_series_prev
            )
            yhat = yhat.univariates[yhat.names[0]]

        # Set up all the parameters needed to make a figure
        if time_series_prev is not None and plot_time_series_prev:
            if not self.invert_transform:
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
        exog_data: TimeSeries = None,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
        figsize=(1000, 600),
        ax=None,
    ):
        """
        Plots the forecast for the time series in matplotlib, optionally also
        plotting the uncertainty of the forecast, as well as the past values
        (both true and predicted) of the time series.

        :param time_series: the time series over whose timestamps we wish to make a forecast. Exactly one of
            ``time_series`` or ``time_stamps`` should be provided.
        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for, or the number of steps (``int``)
            we wish to forecast for. Exactly one of ``time_series`` or ``time_stamps`` should be provided.
        :param time_series_prev: a time series immediately preceding ``time_series``. If given, we use it to initialize
            the forecaster's state. Otherwise, we assume that ``time_series`` immediately follows the training data.
        :param exog_data: A time series of exogenous variables. Exogenous variables are known a priori, and they are
            independent of the variable being forecasted. ``exog_data`` must include data for all of ``time_stamps``;
            if ``time_series_prev`` is given, it must include data for all of ``time_series_prev.time_stamps`` as well.
            Optional. Only supported for models which inherit from `ForecasterWithExog`.
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
            exog_data=exog_data,
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
        exog_data: TimeSeries = None,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
        figsize=(1000, 600),
    ):
        """
        Plots the forecast for the time series in plotly, optionally also
        plotting the uncertainty of the forecast, as well as the past values
        (both true and predicted) of the time series.

        :param time_series: the time series over whose timestamps we wish to make a forecast. Exactly one of
            ``time_series`` or ``time_stamps`` should be provided.
        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for, or the number of steps (``int``)
            we wish to forecast for. Exactly one of ``time_series`` or ``time_stamps`` should be provided.
        :param time_series_prev: a time series immediately preceding ``time_series``. If given, we use it to initialize
            the forecaster's state. Otherwise, we assume that ``time_series`` immediately follows the training data.
        :param exog_data: A time series of exogenous variables. Exogenous variables are known a priori, and they are
            independent of the variable being forecasted. ``exog_data`` must include data for all of ``time_stamps``;
            if ``time_series_prev`` is given, it must include data for all of ``time_series_prev.time_stamps`` as well.
            Optional. Only supported for models which inherit from `ForecasterWithExog`.
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
            exog_data=exog_data,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
        )
        title = f"{type(self).__name__}: Forecast of {self.target_name}"
        return fig.plot_plotly(title=title, metric_name=self.target_name, figsize=figsize)


class ForecasterWithExogConfig(ForecasterConfig):
    def __init__(self, exog_missing_value_policy: Union[MissingValuePolicy, str] = "ZFill", **kwargs):
        """
        :param exog_missing_value_policy: The policy to use for imputing missing values in exogenous data.
        """
        super().__init__(**kwargs)
        self.exog_missing_value_policy = exog_missing_value_policy

    @property
    def exog_missing_value_policy(self):
        return self._exog_missing_value_policy

    @exog_missing_value_policy.setter
    def exog_missing_value_policy(self, mv: Union[MissingValuePolicy, str]):
        if isinstance(mv, str):
            valid = set(MissingValuePolicy.__members__.keys())
            if mv not in valid:
                raise KeyError(f"{mv} is not a valid change kind. Valid missing value policies are: {valid}")
            mv = MissingValuePolicy[mv]
        self._exog_missing_value_policy = mv


class ForecasterWithExogBase(ForecasterBase):
    """
    Base class for a forecaster model which supports exogenous regressors.
    """

    def __init__(self, config):
        super().__init__(config)
        self.exog_dim = None

    @property
    def exog_missing_value_policy(self):
        return self.config.exog_missing_value_policy

    def train(
        self, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None
    ) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        self.exog_dim = exog_data.dim if exog_data is not None else None
        return super().train(train_data=train_data, train_config=train_config, exog_data=exog_data)

    def resample_exog_data(
        self,
        exog_data: TimeSeries,
        time_stamps: Union[List[int], pd.DatetimeIndex],
        time_series_prev: TimeSeries = None,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, None], Tuple[None, None]]:
        """
        Resamples exogenous data and splits it into two subsets: one with the same timestamps as ``time_series_prev``
        (``None`` if ``time_series_prev`` is ``None``), and one with the timestamps ``time_stamps``.

        :param exog_data: The exogenous data of interest.
        :param time_stamps: The timestamps of interest (either the timestamps of data, or the timestamps at which
            we want to obtain a forecast)
        :param time_series_prev: The timestamps of a time series preceding ``time_stamps`` as context. Optional.
        :return: ``(exog_data, exog_data_prev)``, where ``exog_data`` has been resampled to match the ``time_stamps``
            and ``exog_data_prev` has been resampled to match ``time_series_prev.time_stamps``.
        """
        if exog_data is None:
            if self.exog_dim is not None:
                raise ValueError(f"Trained with {self.exog_dim}-dim exogenous data, but received none.")
            return None, None
        if self.exog_dim is None:
            raise ValueError("Trained without exogenous data, but received exogenous data.")
        if self.exog_dim != exog_data.dim:
            raise ValueError(f"Trained with {self.exog_dim}-dim exogenous data, but received {exog_data.dim}-dim.")

        if time_series_prev is not None:
            t = time_series_prev.time_stamps + to_timestamp(time_stamps).tolist()
            exog_data = exog_data.align(reference=t, missing_value_policy=self.exog_missing_value_policy)
            exog_data_prev, exog_data = exog_data.bisect(time_stamps[0], t_in_left=False)
        else:
            exog_data_prev = None
            exog_data = exog_data.align(reference=time_stamps, missing_value_policy=self.exog_missing_value_policy)
        return exog_data, exog_data_prev

    @abstractmethod
    def _train_with_exog(self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None):
        raise NotImplementedError

    def _train(self, train_data: pd.DataFrame, train_config=None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self._train_with_exog(train_data=train_data, train_config=train_config, exog_data=None)

    @abstractmethod
    def _forecast_with_exog(
        self,
        time_stamps: List[int],
        time_series_prev: pd.DataFrame = None,
        return_prev=False,
        exog_data: pd.DataFrame = None,
        exog_data_prev: pd.DataFrame = None,
    ):
        raise NotImplementedError

    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, Union[None, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]:
        return self._forecast_with_exog(
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            return_prev=return_prev,
            exog_data=None,
            exog_data_prev=None,
        )
