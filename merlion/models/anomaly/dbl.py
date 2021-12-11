#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Dynamic Baseline anomaly detection model for time series with daily, weekly or monthly trends.
"""
import logging
from itertools import chain, product


from merlion.plot import Figure
from typing import Any, Dict, List, Tuple
from enum import Enum, auto
from operator import xor
from math import ceil

import numpy as np
import pandas as pd

from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.utils.istat import Mean, Variance
from merlion.utils.resample import granularity_str_to_seconds, to_pd_datetime

logger = logging.getLogger(__name__)


class DynamicBaselineConfig(DetectorConfig):
    """
    Configuration class for `DynamicBaseline`.
    """

    _default_trends = ["weekly", "daily"]

    def __init__(
        self,
        fixed_period: Tuple[str, str] = None,
        train_window: str = None,
        wind_sz: str = "1h",
        trends: List[str] = None,
        **kwargs,
    ):
        """
        :param fixed_period: ``(t0, tf)``; Train the model on all datapoints
            occurring between ``t0`` and ``tf`` (inclusive).
        :param train_window: A string representing a duration of time to serve as
            the scope for a rolling dynamic baseline model.
        :param wind_sz: The window size in minutes to bucket times of day. This
            parameter only applied if a daily trend is one of the trends used.
        :param trends: The list of trends to use. Supported trends are "daily",
            "weekly" and "monthly".
        """
        super().__init__(**kwargs)
        self.trends = self._default_trends if trends is None else trends
        self.wind_sz = wind_sz
        if not xor(fixed_period is None, train_window is None):
            fixed_period = None
            train_window = train_window if train_window is not None else self.determine_train_window()
        self.fixed_period = fixed_period
        self.train_window = train_window

    @property
    def fixed_period(self):
        return self._fixed_period

    @fixed_period.setter
    def fixed_period(self, period: Tuple[str, str]):
        if period is not None:
            assert len(period) == 2
            period = tuple(to_pd_datetime(t) for t in period)
        self._fixed_period = period

    @property
    def trends(self):
        return self._trends

    @trends.setter
    def trends(self, trends: List[str]):
        assert all(
            t.lower() in Trend.__members__ for t in trends
        ), f"Encountered a trend that is unsupported. Supported trend types include: {Trend.__members__.keys()}"
        self._trends = [Trend[t.lower()] for t in trends]

    def determine_train_window(self):
        assert self.trends is not None, "cannot determine `train_window` without trends"
        if Trend.monthly in self.trends:
            return "14w"
        elif Trend.weekly in self.trends:
            return "4w"
        return "2w"

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"trends"}))
        if "trends" not in _skipped_keys:
            config_dict["trends"] = [t.name for t in self.trends]
        return config_dict


class DynamicBaseline(DetectorBase):
    """
    Dynamic baseline-based anomaly detector.

    Detects anomalies by comparing data to historical data that has occurred in
    the same window of time, as defined by any combination of time of day,
    day of week, or day of month.

    A DBL model can have a fixed period or a dynamic rolling period. A fixed
    period model trains its baselines exclusively on datapoints occurring in the
    fixed period, while a rolling period model trains continually on the most
    recent datapoints within its ``train-window``.
    """

    config_class = DynamicBaselineConfig
    _default_post_rule_train_config = dict(metric=TSADMetric.F1, unsup_quantile=None)

    def __init__(self, config: DynamicBaselineConfig):
        super().__init__(config)
        self.segmenter = Segmenter(trends=config.trends, wind_sz=config.wind_sz)
        self._data = UnivariateTimeSeries.empty()

    @property
    def train_window(self):
        return pd.to_timedelta(self.config.train_window)

    @property
    def fixed_period(self):
        return self.config.fixed_period

    @property
    def has_fixed_period(self):
        return self.fixed_period is not None

    @property
    def data(self) -> UnivariateTimeSeries:
        return self._data

    @data.setter
    def data(self, data: UnivariateTimeSeries):
        t0, tf = (
            self.fixed_period
            if self.has_fixed_period
            else (self.last_train_time - self.train_window, self.last_train_time)
        )
        self._data = data.window(t0=t0, tf=tf, include_tf=True)

    def get_relevant(self, data: UnivariateTimeSeries):
        """
        Returns the subset of the data that should be used for training
        or updating.
        """
        if self.has_fixed_period:
            t0 = max(self.last_train_time + pd.to_timedelta("1ms"), self.fixed_period[0])
            tf = self.fixed_period[1]
        else:
            t0 = max(self.last_train_time + pd.to_timedelta("1ms"), to_pd_datetime(data.tf) - self.train_window)
            tf = data.tf

        return data.window(t0=t0, tf=tf, include_tf=True)

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        """
        :param train_data: train_data[t] = (timestamp_t, value_t)
        :param anomaly_labels: anomaly_labels[i] = (timestamp_i, is_anom(timestamp_i))
        :param train_config: unused
        :param post_rule_train_config: config to train the post rule

        :return: anomaly scores of training data
        """
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=True)

        self.last_train_time = pd.Timestamp.min
        train_data = train_data.squeeze()
        rel_train_data = self.get_relevant(train_data)

        if rel_train_data.is_empty():
            logger.warning("relevant `train_data` is empty.")
            return

        self.last_train_time = rel_train_data.tf
        for t, x in rel_train_data:
            self.segmenter.add(t, x)

        # only keep data for rolling case
        self.data = UnivariateTimeSeries.empty() if self.has_fixed_period else rel_train_data
        train_scores = self.get_anomaly_score(train_data.to_ts())
        self.train_post_rule(train_scores, anomaly_labels, post_rule_train_config)
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        """
        :param time_series: a list of (timestamps, score) pairs
        :param time_series_prev: ignored
        """
        time_series, _ = self.transform_time_series(time_series, time_series_prev)
        self.check_dim(time_series)

        scores = [self.segmenter.score(t, x) for t, (x,) in time_series]
        return UnivariateTimeSeries(time_series.time_stamps, scores, "anom_score").to_ts()

    def get_baseline(self, time_stamps: List[float]) -> Tuple[UnivariateTimeSeries, UnivariateTimeSeries]:
        """
        Returns the dynamic baselines corresponding to the time stamps
        :param time_stamps: a list of timestamps
        """
        baselines, err = np.array([self.segmenter.get_baseline(t) for t in time_stamps]).T
        return (
            UnivariateTimeSeries(time_stamps, baselines.tolist(), "baseline").to_ts(),
            UnivariateTimeSeries(time_stamps, err.tolist(), "err").to_ts(),
        )

    def check_dim(self, time_series):
        assert time_series.dim == 1, (
            f"{type(self).__name__} model only accepts univariate time "
            f"series, but time series (after transform {self.transform}) "
            f"has dimension {time_series.dim}"
        )

    def update(self, new_data: TimeSeries):
        assert self.last_train_time is not None, "The model must be initially trained before it can be updated"

        self.check_dim(new_data)
        new_data = self.transform(new_data).squeeze()
        new_data = self.get_relevant(new_data)

        if new_data.is_empty():
            logger.warning("relevant `new_data` is empty.")
            return

        # add new points
        self.last_train_time = new_data.tf
        for t, x in new_data:
            self.segmenter.add(t, x)

        if not self.has_fixed_period:
            # drop points outside rolling scope
            if pd.to_timedelta(new_data.tf - new_data.t0, unit="s") >= self.train_window:
                self.segmenter.reset()
            else:
                old_data, _ = self.data.bisect(to_pd_datetime(new_data.tf) - self.train_window)
                for t, x in old_data:
                    self.segmenter.drop(t, x)
            # update data
            self.data = self.data.concat(new_data)

    def get_baseline_figure(
        self,
        time_series: TimeSeries,
        time_series_prev: TimeSeries = None,
        *,
        filter_scores=True,
        plot_time_series_prev=False,
        fig: Figure = None,
        jitter_time_stamps=True,
    ) -> Figure:
        time_stamps = time_series.time_stamps
        if jitter_time_stamps:
            time_stamps = list(chain.from_iterable([t - 0.001, t, t + 0.001] for t in time_stamps))[1:-1]
        # get baselines and errors
        baseline, err = self.get_baseline(time_stamps)
        ub = UnivariateTimeSeries(
            time_stamps=err.time_stamps, values=baseline.squeeze().np_values + 2 * err.squeeze().np_values
        )
        lb = UnivariateTimeSeries(
            time_stamps=err.time_stamps, values=baseline.squeeze().np_values - 2 * err.squeeze().np_values
        )
        baseline = baseline.squeeze()

        assert time_series.dim == 1, (
            f"Plotting only supported for univariate time series, but got a"
            f"time series of dimension {time_series.dim}"
        )
        time_series = time_series.univariates[time_series.names[0]]

        if fig is None:
            if time_series_prev is not None and plot_time_series_prev:
                k = time_series_prev.names[0]
                time_series_prev = time_series_prev.univariates[k]
            elif not plot_time_series_prev:
                time_series_prev = None
            fig = Figure(y=time_series, y_prev=time_series_prev, yhat=baseline, yhat_lb=lb, yhat_ub=ub)
        else:
            if fig.y is None:
                fig.y = time_series
            fig.yhat = baseline
            fig.yhat_iqr = TimeSeries({"lb": lb, "ub": ub}).align()
        return fig


class Trend(Enum):
    """
    Enumeration of the supported trends.
    """

    daily = auto()
    weekly = auto()
    monthly = auto()


class Segment:
    """
    Class representing a segment. The class maintains a mean (baseline)
    along with a variance so that a z-score can be computed.
    """

    def __init__(self, key):
        self.key = key
        self.mean = Mean()
        self.var = Variance()

    def add(self, x: float):
        self.mean.add(x)
        self.var.add(x)

    def drop(self, x: float):
        self.mean.drop(x)
        self.var.drop(x)

    def score(self, x: float):
        mu, sd = self.mean.value, self.var.sd
        # if segment is empty, assume data is not anomalous
        if mu is None or sd is None or sd == np.inf:
            return 0
        return (x - mu) / (sd + 1e-8)


class Segmenter:
    """
    Class for managing the segments that belong to a `DynamicBaseline` model.
    """

    day_delta = pd.Timedelta("1d")
    hour_delta = pd.Timedelta("1h")
    min_delta = pd.Timedelta("1min")
    zero_delta = pd.Timedelta("0min")

    def __init__(self, trends: List[Trend], wind_sz: str):
        """
        :param trends: A list of trend types to create segments based on.
        :param wind_sz: The window size in minutes to bucket times of day.
            Only used if a daily trend is one of the trends used.
        """
        self.wind_sz = wind_sz
        self.trend = trends

    def reset(self):
        self.__init__(self.trends, self.wind_sz)

    @property
    def wind_delta(self):
        return pd.Timedelta(self.wind_sz)

    @property
    def trends(self):
        return self._trends

    @trends.setter
    def trend(self, trends: List[Trend]):
        self._trends = trends
        # update trend types
        self.has_daily, self.has_weekly, self.has_monthly = [
            trend in trends for trend in (Trend.daily, Trend.weekly, Trend.monthly)
        ]
        # update segments
        windows = range(ceil(self.day_delta / self.wind_delta)) if self.has_daily else [None]
        weekdays = range(7) if self.has_weekly else [None]
        days = range(1, 32) if self.has_monthly else [None]
        self.segments = {k: Segment(k) for k in product(days, weekdays, windows)}

    def window_key(self, t: pd.Timestamp):
        if not self.has_daily:
            return None
        day_elapsed_delta = t.hour * self.hour_delta + t.minute * self.min_delta
        return day_elapsed_delta // self.wind_delta

    def weekday_key(self, t: pd.Timestamp):
        return t.dayofweek if self.has_weekly else None

    def day_key(self, t: pd.Timestamp):
        return t.day if self.has_monthly else None

    def segment_key(self, timestamp: float):
        t = to_pd_datetime(timestamp)
        return self.day_key(t), self.weekday_key(t), self.window_key(t)

    def add(self, t: float, x: float):
        key = self.segment_key(t)
        self.segments[key].add(x)

    def drop(self, t: float, x: float):
        key = self.segment_key(t)
        self.segments[key].drop(x)

    def score(self, t: float, x: float):
        key = self.segment_key(t)
        return self.segments[key].score(x)

    def get_baseline(self, t: float) -> Tuple[float, float]:
        segment = self.segments[self.segment_key(t)]
        return segment.mean.value, segment.var.sd
