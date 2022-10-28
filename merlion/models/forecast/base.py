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
from merlion.transform.base import TransformBase
from merlion.transform.factory import TransformFactory
from merlion.transform.normalize import MeanVarNormalize
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy

logger = logging.getLogger(__name__)


class ForecasterConfig(Config):
    """
    Config object used to define a forecaster model.
    """

    max_forecast_steps: Optional[int] = None
    target_seq_index: Optional[int] = None
    invert_transform: bool = None

    def __init__(self, max_forecast_steps: int = None, target_seq_index: int = None, invert_transform=None, **kwargs):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.  Required for some models like `MSES`.
        :param target_seq_index: The index of the univariate (amongst all univariates in a general multivariate time
            series) whose value we would like to forecast.
        :param invert_transform: Whether to automatically invert the ``transform`` before returning a forecast.
            By default, we will invert the transform for all base forecasters if it supports a proper inversion, but
            we will not invert it for forecaster-based anomaly detectors or transforms without proper inversions.
        """
        from merlion.models.anomaly.base import DetectorConfig

        super().__init__(**kwargs)
        if invert_transform is None:
            invert_transform = self.transform.proper_inversion and not isinstance(self, DetectorConfig)
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
        if self.supports_exog:
            assert isinstance(config, ForecasterExogConfig)
        super().__init__(config)
        self.target_name = None
        self.exog_dim = None

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

    @property
    def supports_exog(self):
        """
        Whether this forecaster supports exogenous data.
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

    def train_pre_process(
        self, train_data: TimeSeries, exog_data: TimeSeries = None, return_exog=None
    ) -> Union[TimeSeries, Tuple[TimeSeries, Union[TimeSeries, None]]]:
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

        # Handle exogenous data
        if return_exog is None:
            return_exog = exog_data is not None
        if not self.supports_exog:
            if exog_data is not None:
                exog_data = None
                logger.warning(f"Exogenous regressors are not supported for model {type(self).__name__}")
        if exog_data is not None:
            self.exog_dim = exog_data.dim
            self.config.exog_transform.train(exog_data)
        else:
            self.exog_dim = None
        if return_exog and exog_data is not None:
            exog_data, _ = self.transform_exog_data(exog_data=exog_data, time_stamps=train_data.time_stamps)
        return (train_data, exog_data) if return_exog else train_data

    def train(
        self, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None
    ) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        """
        Trains the forecaster on the input time series.

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param train_config: Additional training configs, if needed. Only required for some models.
        :param exog_data: A time series of exogenous variables, sampled at the same time stamps as ``train_data``.
            Exogenous variables are known a priori, and they are independent of the variable being forecasted.
            Only supported for models which inherit from `ForecasterExogBase`.

        :return: the model's prediction on ``train_data``, in the same format as
            if you called `ForecasterBase.forecast` on the time stamps of ``train_data``
        """
        if train_config is None:
            train_config = copy.deepcopy(self._default_train_config)
        train_data, exog_data = self.train_pre_process(train_data, exog_data=exog_data, return_exog=True)
        if self._pandas_train:
            train_data = train_data.to_pd()
            exog_data = None if exog_data is None else exog_data.to_pd()
        if exog_data is None:
            train_result = self._train(train_data=train_data, train_config=train_config)
        else:
            train_result = self._train_with_exog(train_data=train_data, train_config=train_config, exog_data=exog_data)
        return self.train_post_process(train_result)

    def train_post_process(
        self, train_result: Tuple[Union[TimeSeries, pd.DataFrame], Optional[Union[TimeSeries, pd.DataFrame]]]
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Converts the train result (forecast & stderr for training data) into TimeSeries objects, and inverts the
        model's transform if desired.
        """
        return self._process_forecast(*train_result)

    def transform_exog_data(
        self,
        exog_data: TimeSeries,
        time_stamps: Union[List[int], pd.DatetimeIndex],
        time_series_prev: TimeSeries = None,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, None], Tuple[None, None]]:
        if exog_data is not None:
            logger.warning(f"Exogenous regressors are not supported for model {type(self).__name__}")
        return None, None

    @abstractmethod
    def _train(self, train_data: pd.DataFrame, train_config=None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        raise NotImplementedError

    def _train_with_exog(
        self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self._train(train_data=train_data, train_config=train_config)

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
            Optional. Only supported for models which inherit from `ForecasterExogBase`.
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
        # Determine the time stamps to forecast for, and resample them if needed
        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)
        if return_prev and time_series_prev is not None:
            if orig_t is None:
                orig_t = time_series_prev.time_stamps + time_stamps
            else:
                orig_t = time_series_prev.time_stamps + to_timestamp(orig_t).tolist()

        # Transform time_series_prev if it is given
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
        exog_data, exog_data_prev = self.transform_exog_data(
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

        # Format the return values and reset the transform's inversion state
        if self.invert_transform and time_series_prev is None:
            time_series_prev = self.transform(self.train_data)
        if time_series_prev is not None:
            time_series_prev = pd.DataFrame(time_series_prev.univariates[time_series_prev.names[self.target_seq_index]])
        ret = self._process_forecast(forecast, err, time_series_prev, return_prev=return_prev, return_iqr=return_iqr)
        self.transform.inversion_state = old_inversion_state
        return tuple(None if x is None else x.align(reference=orig_t) for x in ret)

    def _process_forecast(self, forecast, err, time_series_prev=None, return_prev=False, return_iqr=False):
        forecast = forecast.to_pd() if isinstance(forecast, TimeSeries) else forecast
        if return_prev and time_series_prev is not None:
            forecast = pd.concat((time_series_prev, forecast))

        # Obtain negative & positive error bars which are appropriately padded
        if err is not None:
            err = (err,) if not isinstance(err, tuple) else err
            assert isinstance(err, tuple) and len(err) in (1, 2)
            assert all(isinstance(e, (pd.DataFrame, TimeSeries)) for e in err)
            new_err = []
            for e in err:
                e = e.to_pd() if isinstance(e, TimeSeries) else e
                n, d = len(forecast) - len(e), e.shape[1]
                if n > 0:
                    e = pd.concat((pd.DataFrame(np.zeros((n, d)), index=forecast.index[:n], columns=e.columns), e))
                e.columns = [f"{c}_err" for c in forecast.columns]
                new_err.append(e.abs())
            e_neg, e_pos = new_err if len(new_err) == 2 else (new_err[0], new_err[0])
        else:
            e_neg = e_pos = None

        # Compute upper/lower bounds for the (potentially inverted) forecast.
        # Only do this if returning the IQR or inverting the transform.
        invert_transform = self.invert_transform and not self.transform.identity_inversion
        if (return_iqr or invert_transform) and e_neg is not None and e_pos is not None:
            lb = TimeSeries.from_pd((forecast + e_neg.values * (norm.ppf(0.25) if return_iqr else -1)))
            ub = TimeSeries.from_pd((forecast + e_pos.values * (norm.ppf(0.75) if return_iqr else 1)))
            if invert_transform:
                lb = self.transform.invert(lb, retain_inversion_state=True)
                ub = self.transform.invert(ub, retain_inversion_state=True)
        else:
            lb = ub = None

        # Convert the forecast to TimeSeries and invert the transform on it if desired
        forecast = TimeSeries.from_pd(forecast)
        if invert_transform:
            forecast = self.transform.invert(forecast, retain_inversion_state=True)

        # Return the IQR if desired
        if return_iqr:
            if lb is None or ub is None:
                logger.warning("Model returned err = None, so returning IQR = (None, None)")
            else:
                lb, ub = lb.rename(lambda c: f"{c}_lower"), ub.rename(lambda c: f"{c}_upper")
            return forecast, lb, ub

        # Otherwise, either compute the stderr from the upper/lower bounds (if relevant), or just use the error
        if lb is not None and ub is not None:
            err = TimeSeries.from_pd((ub.to_pd() - lb.to_pd().values).rename(columns=lambda c: f"{c}_err").abs() / 2)
        elif e_neg is not None and e_pos is not None:
            err = TimeSeries.from_pd(e_pos if e_neg is e_pos else (e_neg + e_pos) / 2)
        else:
            err = None
        return forecast, err

    @abstractmethod
    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        raise NotImplementedError

    def _forecast_with_exog(
        self,
        time_stamps: List[int],
        time_series_prev: pd.DataFrame = None,
        return_prev=False,
        exog_data: pd.DataFrame = None,
        exog_data_prev: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self._forecast(time_stamps=time_stamps, time_series_prev=time_series_prev, return_prev=return_prev)

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
        Returns the model's forecast on a batch of timestamps given.

        :param time_stamps_list: a list of lists of timestamps we wish to forecast for
        :param time_series_prev_list: a list of TimeSeries immediately preceding the time stamps in time_stamps_list
        :param return_iqr: whether to return the inter-quartile range for the forecast.
            Only supported by models which can return error bars.
        :param return_prev: whether to return the forecast for ``time_series_prev`` (and its stderr or IQR if relevant),
            in addition to the forecast for ``time_stamps``. Only used if ``time_series_prev`` is provided.
        :return: ``(forecast, forecast_stderr)`` if ``return_iqr`` is false,
            ``(forecast, forecast_lb, forecast_ub)`` otherwise.

            - ``forecast``: the forecast for the timestamps given
            - ``forecast_stderr``: the standard error of each forecast value. May be ``None``.
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
            Optional. Only supported for models which inherit from `ForecasterExogBase`.
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
            time_series_prev = time_series_prev.univariates[time_series_prev.names[self.target_seq_index]]

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
            Optional. Only supported for models which inherit from `ForecasterExogBase`.
        :param plot_forecast_uncertainty: whether to plot uncertainty estimates (the inter-quartile range) for forecast
            values. Not supported for all models.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and the model's fit for it). Only used if
            ``time_series_prev`` is given.
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
            Optional. Only supported for models which inherit from `ForecasterExogBase`.
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


class ForecasterExogConfig(ForecasterConfig):
    _default_exog_transform = MeanVarNormalize()
    exog_transform: TransformBase = None

    def __init__(
        self,
        exog_transform: TransformBase = None,
        exog_aggregation_policy: Union[AggregationPolicy, str] = "Mean",
        exog_missing_value_policy: Union[MissingValuePolicy, str] = "ZFill",
        **kwargs,
    ):
        """
        :param exog_transform: The pre-processing transform for exogenous data. Note: resampling is handled separately.
        :param exog_aggregation_policy: The policy to use for aggregating values in exogenous data,
            to ensure it is sampled at the same timestamps as the endogenous data.
        :param exog_missing_value_policy: The policy to use for imputing missing values in exogenous data,
            to ensure it is sampled at the same timestamps as the endogenous data.
        """
        super().__init__(**kwargs)
        if exog_transform is None:
            self.exog_transform = copy.deepcopy(self._default_exog_transform)
        elif isinstance(exog_transform, dict):
            self.exog_transform = TransformFactory.create(**exog_transform)
        else:
            self.exog_transform = exog_transform
        self.exog_aggregation_policy = exog_aggregation_policy
        self.exog_missing_value_policy = exog_missing_value_policy

    @property
    def exog_aggregation_policy(self):
        return self._exog_aggregation_policy

    @exog_aggregation_policy.setter
    def exog_aggregation_policy(self, agg):
        if isinstance(agg, str):
            valid = set(AggregationPolicy.__members__.keys())
            if agg not in valid:
                raise KeyError(f"{agg} is not a aggregation policy. Valid aggregation policies are: {valid}")
            agg = AggregationPolicy[agg]
        self._exog_aggregation_policy = agg

    @property
    def exog_missing_value_policy(self):
        return self._exog_missing_value_policy

    @exog_missing_value_policy.setter
    def exog_missing_value_policy(self, mv: Union[MissingValuePolicy, str]):
        if isinstance(mv, str):
            valid = set(MissingValuePolicy.__members__.keys())
            if mv not in valid:
                raise KeyError(f"{mv} is not a valid missing value policy. Valid missing value policies are: {valid}")
            mv = MissingValuePolicy[mv]
        self._exog_missing_value_policy = mv


class ForecasterExogBase(ForecasterBase):
    """
    Base class for a forecaster model which supports exogenous variables. Exogenous variables are known a priori, and
    they are independent of the variable being forecasted.
    """

    @property
    def supports_exog(self):
        return True

    @property
    def exog_transform(self):
        return self.config.exog_transform

    @property
    def exog_aggregation_policy(self):
        return self.config.exog_aggregation_policy

    @property
    def exog_missing_value_policy(self):
        return self.config.exog_missing_value_policy

    def transform_exog_data(
        self,
        exog_data: TimeSeries,
        time_stamps: Union[List[int], pd.DatetimeIndex],
        time_series_prev: TimeSeries = None,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, None], Tuple[None, None]]:
        """
        Transforms & resamples exogenous data and splits it into two subsets:
        one with the same timestamps as ``time_series_prev`` (``None`` if ``time_series_prev`` is ``None``),
        and one with the timestamps ``time_stamps``.

        :param exog_data: The exogenous data of interest.
        :param time_stamps: The timestamps of interest (either the timestamps of data, or the timestamps at which
            we want to obtain a forecast)
        :param time_series_prev: The timestamps of a time series preceding ``time_stamps`` as context. Optional.
        :return: ``(exog_data, exog_data_prev)``, where ``exog_data`` has been resampled to match the ``time_stamps``
            and ``exog_data_prev` has been resampled to match ``time_series_prev.time_stamps``.
        """
        # Check validity
        if exog_data is None:
            if self.exog_dim is not None:
                raise ValueError(f"Trained with {self.exog_dim}-dim exogenous data, but received none.")
            return None, None
        if self.exog_dim is None:
            raise ValueError("Trained without exogenous data, but received exogenous data.")
        if self.exog_dim != exog_data.dim:
            raise ValueError(f"Trained with {self.exog_dim}-dim exogenous data, but received {exog_data.dim}-dim.")

        # Transform & resample
        exog_data = self.exog_transform(exog_data)
        if time_series_prev is not None:
            t = time_series_prev.time_stamps + to_timestamp(time_stamps).tolist()
            exog_data = exog_data.align(
                reference=t,
                aggregation_policy=self.exog_aggregation_policy,
                missing_value_policy=self.exog_missing_value_policy,
            )
            exog_data_prev, exog_data = exog_data.bisect(time_stamps[0], t_in_left=False)
        else:
            exog_data_prev = None
            exog_data = exog_data.align(
                reference=time_stamps,
                aggregation_policy=self.exog_aggregation_policy,
                missing_value_policy=self.exog_missing_value_policy,
            )
        return exog_data, exog_data_prev

    @abstractmethod
    def _train_with_exog(
        self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        raise NotImplementedError

    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self._forecast_with_exog(
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            return_prev=return_prev,
            exog_data=None,
            exog_data_prev=None,
        )
