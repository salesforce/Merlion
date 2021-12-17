#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for anomaly detectors based on forecasting models.
"""
import logging
from typing import List, Optional

import numpy as np

from merlion.models.anomaly.base import DetectorBase
from merlion.models.forecast.base import ForecasterBase
from merlion.plot import Figure
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


class ForecastingDetectorBase(ForecasterBase, DetectorBase, metaclass=AutodocABCMeta):
    """
    Base class for a forecast-based anomaly detector.
    """

    @property
    def _default_post_rule_train_config(self):
        from merlion.evaluate.anomaly import TSADMetric

        return dict(metric=TSADMetric.F1, unsup_quantile=None)

    def forecast_to_anom_score(
        self, time_series: TimeSeries, forecast: TimeSeries, stderr: Optional[TimeSeries]
    ) -> TimeSeries:
        """
        Compare a model's forecast to a ground truth time series, in order to
        compute anomaly scores. By default, we compute a z-score if model
        uncertainty (``stderr``) is given, or the residuals if there is no
        model uncertainty.

        :param time_series: the ground truth time series.
        :param forecast: the model's forecasted values for the time series
        :param stderr: the standard errors of the model's forecast

        :return: Anomaly scores based on the difference between the ground truth
            values of the time series, and the model's forecast.
        """
        if len(forecast) == 0:
            return UnivariateTimeSeries.empty(name="anom_score").to_ts()
        i = self.target_seq_index
        time_series = time_series.univariates[time_series.names[i]]
        if len(time_series) > len(forecast):
            time_series = time_series[-len(forecast) :]
        times = time_series.index
        y = time_series.np_values
        yhat = forecast.univariates[forecast.names[0]].np_values
        if stderr is None:
            return UnivariateTimeSeries(time_stamps=times, values=(y - yhat), name="anom_score").to_ts()
        else:
            sigma = stderr.univariates[stderr.names[0]].np_values
            if np.isnan(sigma).all():
                sigma = 1
            else:
                sigma[np.isnan(sigma)] = np.mean(sigma)
            return UnivariateTimeSeries(
                time_stamps=times, values=(y - yhat) / (sigma + 1e-8), name="anom_score"
            ).to_ts()

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        """
        Trains the underlying forecaster (unsupervised) on the training data.
        Converts the  forecast into anomaly scores, and and then trains the
        post-rule for filtering anomaly scores (supervised, if labels are given)
        on the input time series.

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param anomaly_labels: a `TimeSeries` indicating which timestamps are
            anomalous. Optional.
        :param train_config: Additional training configs, if needed. Only
            required for some models.
        :param post_rule_train_config: The config to use for training the
            model's post-rule. The model's default post-rule train config is
            used if none is supplied here.
        :return: A `TimeSeries` of the model's anomaly scores on the training
            data.
        """
        forecast, err = super().train(train_data, train_config)
        anomaly_scores = self.forecast_to_anom_score(self.transform(train_data), forecast, err)

        self.train_post_rule(
            anomaly_scores=anomaly_scores, anomaly_labels=anomaly_labels, post_rule_train_config=post_rule_train_config
        )
        return anomaly_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        time_series, _ = self.transform_time_series(time_series, time_series_prev)
        name = time_series.names[self.target_seq_index]
        time_stamps = time_series.univariates[name].time_stamps
        forecast, err = self.forecast(time_stamps, time_series_prev, return_iqr=False, return_prev=False)

        assert err is None or len(forecast) == len(err), (
            f"Expected forecast & standard error of forecast to have the same "
            f"length, but len(forecast) = {len(forecast)}, len(err) = {len(err)}"
        )
        assert len(forecast) == len(
            time_series
        ), f"forecast() returned a forecast with length {len(forecast)}, but expected length {len(time_series)}"

        return self.forecast_to_anom_score(time_series, forecast, err)

    def get_figure(
        self,
        *,
        time_series: TimeSeries = None,
        time_stamps: List[int] = None,
        time_series_prev: TimeSeries = None,
        plot_anomaly=True,
        filter_scores=True,
        plot_forecast=False,
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
        :param plot_anomaly: Whether to plot the model's predicted anomaly scores.
        :param filter_scores: whether to filter the anomaly scores by the
            post-rule before plotting them.
        :param plot_forecast: Whether to plot the model's forecasted values.
        :param plot_forecast_uncertainty: whether to plot uncertainty estimates (the
            inter-quartile range) for forecast values. Not supported for all
            models.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :return: a `Figure` of the model's anomaly score predictions and/or forecast.
        """
        assert not (
            time_series is None and time_stamps is None
        ), "Must provide at least one of time_series or time_stamps"
        fig = None
        plot_forecast = plot_forecast or not plot_anomaly
        if plot_forecast or time_series is None:
            fig = ForecasterBase.get_figure(
                self,
                time_series=time_series,
                time_stamps=time_stamps,
                time_series_prev=time_series_prev,
                plot_forecast_uncertainty=plot_forecast_uncertainty,
                plot_time_series_prev=plot_time_series_prev,
            )
        if time_series is None or not plot_anomaly:
            return fig
        return DetectorBase.get_figure(
            self,
            time_series=time_series,
            time_series_prev=time_series_prev,
            plot_time_series_prev=plot_time_series_prev,
            filter_scores=filter_scores,
            fig=fig,
        )

    def plot_anomaly(
        self,
        time_series: TimeSeries,
        time_series_prev: TimeSeries = None,
        *,
        filter_scores=True,
        plot_forecast=False,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
        figsize=(1000, 600),
        ax=None,
    ):
        """
        Plots the time series in matplotlib as a line graph, with points in the
        series overlaid as points color-coded to indicate their severity as
        anomalies. Optionally allows you to overlay the model's forecast & the
        model's uncertainty in its forecast (if applicable).

        :param time_series: The time series we wish to plot, with color-coding
            to indicate anomalies.
        :param time_series_prev: A time series immediately preceding
            ``time_series``, which is used to initialize the time series model.
            Otherwise, we assume ``time_series`` immediately follows the training
            data.
        :param filter_scores: whether to filter the anomaly scores by the
            post-rule before plotting them.
        :param plot_forecast: Whether to plot the model's forecast, in addition
            to the anomaly scores.
        :param plot_forecast_uncertainty: Whether to plot the model's
            uncertainty in its own forecast, in addition to the forecast and
            anomaly scores. Only used if ``plot_forecast`` is ``True``.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param figsize: figure size in pixels
        :param ax: matplotlib axis to add this plot to

        :return: matplotlib figure & axes
        """
        metric_name = time_series.names[0]
        fig = self.get_figure(
            time_series=time_series,
            time_series_prev=time_series_prev,
            filter_scores=filter_scores,
            plot_anomaly=True,
            plot_forecast=plot_forecast,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
        )

        title = f"{type(self).__name__}: Anomalies in {metric_name}"
        if plot_forecast:
            title += " (Forecast Overlaid)"
        return fig.plot(title=title, figsize=figsize, ax=ax)

    def plot_anomaly_plotly(
        self,
        time_series: TimeSeries,
        time_series_prev: TimeSeries = None,
        *,
        filter_scores=True,
        plot_forecast=False,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
        figsize=(1000, 600),
    ):
        """
        Plots the time series in matplotlib as a line graph, with points in the
        series overlaid as points color-coded to indicate their severity as
        anomalies. Optionally allows you to overlay the model's forecast & the
        model's uncertainty in its forecast (if applicable).

        :param time_series: The time series we wish to plot, with color-coding
            to indicate anomalies.
        :param time_series_prev: A time series immediately preceding
            ``time_series``, which is used to initialize the time series model.
            Otherwise, we assume ``time_series`` immediately follows the training
            data.
        :param filter_scores: whether to filter the anomaly scores by the
            post-rule before plotting them.
        :param plot_forecast: Whether to plot the model's forecast, in addition
            to the anomaly scores.
        :param plot_forecast_uncertainty: Whether to plot the model's
            uncertainty in its own forecast, in addition to the forecast and
            anomaly scores. Only used if ``plot_forecast`` is ``True``.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param figsize: figure size in pixels
        :return: plotly figure
        """
        metric_name = time_series.names[0]
        fig = self.get_figure(
            time_series=time_series,
            time_series_prev=time_series_prev,
            filter_scores=filter_scores,
            plot_forecast=plot_forecast,
            plot_anomaly=True,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
        )
        title = f"{type(self).__name__}: Anomalies in {metric_name}"
        if plot_forecast:
            title += " (Forecast Overlaid)"
        return fig.plot_plotly(title=title, metric_name=metric_name, figsize=figsize)

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
        fig = self.get_figure(
            time_series=time_series,
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
            plot_anomaly=False,
            plot_forecast=True,
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
        fig = self.get_figure(
            time_series=time_series,
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
            plot_anomaly=False,
            plot_forecast=True,
        )
        title = f"{type(self).__name__}: Forecast of {self.target_name}"
        return fig.plot_plotly(title=title, metric_name=self.target_name, figsize=figsize)
