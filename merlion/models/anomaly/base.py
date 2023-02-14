#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for anomaly detectors.
"""
from abc import abstractmethod
import copy
import inspect
import logging
from typing import Any, Dict, List, Union

import pandas as pd
from scipy.stats import norm

from merlion.models.base import Config, ModelBase, MultipleTimeseriesModelMixin
from merlion.plot import Figure, MTSFigure
from merlion.post_process.calibrate import AnomScoreCalibrator
from merlion.post_process.factory import PostRuleFactory
from merlion.post_process.sequence import PostRuleSequence
from merlion.post_process.threshold import AggregateAlarms, Threshold
from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.utils.misc import call_with_accepted_kwargs

logger = logging.getLogger(__name__)


class DetectorConfig(Config):
    """
    Config object used to define an anomaly detection model.
    """

    _default_threshold = AggregateAlarms(alm_threshold=3.0)
    calibrator: AnomScoreCalibrator = None
    threshold: Threshold = None
    enable_calibrator: bool = True
    enable_threshold: bool = True

    def __init__(
        self, max_score: float = 1000, threshold=None, enable_calibrator=True, enable_threshold=True, **kwargs
    ):
        """
        Base class of the object used to configure an anomaly detection model.

        :param max_score: maximum possible uncalibrated anomaly score
        :param threshold: the rule to use for thresholding anomaly scores
        :param enable_threshold: whether to enable the thresholding rule
            when post-processing anomaly scores
        :param enable_calibrator: whether to enable a calibrator which
            automatically transforms all raw anomaly scores to be z-scores
            (i.e. distributed as N(0, 1)).
        """
        super().__init__(**kwargs)
        self.enable_threshold = enable_threshold
        self.enable_calibrator = enable_calibrator
        self.calibrator = AnomScoreCalibrator(max_score=max_score)
        if threshold is None:
            self.threshold = copy.deepcopy(self._default_threshold)
        elif isinstance(threshold, dict):
            self.threshold = PostRuleFactory.create(**threshold)
        else:
            self.threshold = threshold

    @property
    def post_rule(self):
        """
        :return: The full post-processing rule. Includes calibration if
            ``enable_calibrator`` is ``True``, followed by thresholding if
            ``enable_threshold`` is ``True``.
        """
        rules = []
        if self.enable_calibrator and self.calibrator is not None:
            rules.append(self.calibrator)
        if self.enable_threshold and self.threshold is not None:
            rules.append(self.threshold)
        return PostRuleSequence(rules)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], return_unused_kwargs=False, calibrator=None, **kwargs):
        # Get the calibrator, but we will set it manually after the constructor by putting it in kwargs
        calibrator = config_dict.pop("calibrator", calibrator)
        config, kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)
        if calibrator is not None:
            calibrator = PostRuleFactory.create(**calibrator)
            config.calibrator = calibrator

        if len(kwargs) > 0 and not return_unused_kwargs:
            logger.warning(f"Unused kwargs: {kwargs}", stack_info=True)
        elif return_unused_kwargs:
            return config, kwargs
        return config


class NoCalibrationDetectorConfig(DetectorConfig):
    """
    Abstract config object for an anomaly detection model that should never
    perform anomaly score calibration.
    """

    def __init__(self, enable_calibrator=False, **kwargs):
        """
        :param enable_calibrator: ``False`` because this config assumes calibrated outputs from the model.
        """
        super().__init__(enable_calibrator=enable_calibrator, **kwargs)

    @property
    def calibrator(self):
        """
        :return: ``None``
        """
        return None

    @calibrator.setter
    def calibrator(self, calibrator):
        # no-op
        pass

    @property
    def enable_calibrator(self):
        """
        :return: ``False``
        """
        return False

    @enable_calibrator.setter
    def enable_calibrator(self, e):
        if e is not False:
            logger.warning(f"Tried to set enable_calibrator={e}, but only False supported for {type(self).__name__}.")


class DetectorBase(ModelBase):
    """
    Base class for an anomaly detection model.
    """

    config_class = DetectorConfig

    def __init__(self, config: DetectorConfig):
        """
        :param config: model configuration
        """
        super().__init__(config)

    @property
    def _default_post_rule_train_config(self):
        """
        :return: the default config to use when training the post-rule.
        """
        from merlion.evaluate.anomaly import TSADMetric

        t = self.config._default_threshold.alm_threshold
        # self.calibrator is only None if calibration has been manually disabled
        # and the anomaly scores are expected to be calibrated by get_anomaly_score(). If
        # self.config.enable_calibrator, the model will return a calibrated score.
        if self.calibrator is None or self.config.enable_calibrator or t == 0:
            q = None
        # otherwise, choose the quantile corresponding to the given threshold
        else:
            q = 2 * norm.cdf(t) - 1
        return dict(metric=TSADMetric.F1, unsup_quantile=q)

    @property
    def threshold(self):
        return self.config.threshold

    @threshold.setter
    def threshold(self, threshold):
        self.config.threshold = threshold

    @property
    def calibrator(self):
        return self.config.calibrator

    @property
    def post_rule(self):
        return self.config.post_rule

    def train(
        self, train_data: TimeSeries, train_config=None, anomaly_labels: TimeSeries = None, post_rule_train_config=None
    ) -> TimeSeries:
        """
        Trains the anomaly detector (unsupervised) and its post-rule (supervised, if labels are given) on train data.

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param train_config: Additional training configs, if needed. Only required for some models.
        :param anomaly_labels: a `TimeSeries` indicating which timestamps are anomalous. Optional.
        :param post_rule_train_config: The config to use for training the model's post-rule. The model's default
            post-rule train config is used if none is supplied here.

        :return: A `TimeSeries` of the model's anomaly scores on the training data.
        """
        if train_config is None:
            train_config = copy.deepcopy(self._default_train_config)
        train_data = self.train_pre_process(train_data)
        train_data = train_data.to_pd() if self._pandas_train else train_data
        train_result = call_with_accepted_kwargs(  # For ensembles
            self._train, train_data=train_data, train_config=train_config, anomaly_labels=anomaly_labels
        )
        return self.train_post_process(
            train_result=train_result, anomaly_labels=anomaly_labels, post_rule_train_config=post_rule_train_config
        )

    def train_post_process(
        self, train_result: Union[TimeSeries, pd.DataFrame], anomaly_labels=None, post_rule_train_config=None
    ) -> TimeSeries:
        """
        Converts the train result (anom scores on train data) into a TimeSeries object and trains the post-rule.

        :param train_result: Raw anomaly scores on the training data.
        :param anomaly_labels: a `TimeSeries` indicating which timestamps are anomalous. Optional.
        :param post_rule_train_config: The config to use for training the model's post-rule. The model's default
            post-rule train config is used if none is supplied here.
        """
        anomaly_scores = UnivariateTimeSeries.from_pd(train_result, name="anom_score").to_ts()
        if self.post_rule is not None:
            kwargs = copy.copy(self._default_post_rule_train_config)
            if post_rule_train_config is not None:
                kwargs.update(post_rule_train_config)
            kwargs.update(anomaly_scores=anomaly_scores, anomaly_labels=anomaly_labels)
            call_with_accepted_kwargs(self.post_rule.train, **kwargs)
        return anomaly_scores

    @abstractmethod
    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        raise NotImplementedError

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        """
        Returns the model's predicted sequence of anomaly scores.

        :param time_series: the `TimeSeries` we wish to predict anomaly scores
            for.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_series``. If given, we use it to initialize the time series
            anomaly detection model. Otherwise, we assume that ``time_series``
            immediately follows the training data.
        :return: a univariate `TimeSeries` of anomaly scores
        """
        # Ensure the dimensions are correct
        assert (
            time_series.dim == self.dim
        ), f"Expected time_series to have dimension {self.dim}, but got {time_series.dim}."
        if time_series_prev is not None:
            assert (
                time_series_prev.dim == self.dim
            ), f"Expected time_series_prev to have dimension {self.dim}, but got {time_series_prev.dim}."

        # Transform the time series
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
        if self.require_univariate:
            assert time_series.dim == 1, (
                f"{type(self).__name__} model only accepts univariate time series, but time series "
                f"(after transform {self.transform}) has dimension {time_series.dim}."
            )

        time_series = time_series.to_pd()
        if time_series_prev is not None:
            time_series_prev = time_series_prev.to_pd()

        # Get the anomaly scores & ensure the dimensions are correct
        anom_scores = self._get_anomaly_score(time_series, time_series_prev)
        assert anom_scores.shape[1] == 1, f"Expected anomaly scores returned by {type(self)} to be univariate."
        return UnivariateTimeSeries.from_pd(anom_scores, name="anom_score").to_ts()

    def get_anomaly_label(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        """
        Returns the model's predicted sequence of anomaly scores, processed
        by any relevant post-rules (calibration and/or thresholding).

        :param time_series: the `TimeSeries` we wish to predict anomaly scores
            for.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_series``. If given, we use it to initialize the time series
            anomaly detection model. Otherwise, we assume that ``time_series``
            immediately follows the training data.
        :return: a univariate `TimeSeries` of anomaly scores, filtered by the
            model's post-rule
        """
        scores = self.get_anomaly_score(time_series, time_series_prev)
        return self.post_rule(scores) if self.post_rule is not None else scores

    def get_figure(
        self,
        time_series: TimeSeries,
        time_series_prev: TimeSeries = None,
        *,
        filter_scores=True,
        plot_time_series_prev=False,
        fig: Figure = None,
        **kwargs,
    ) -> Figure:
        """
        :param time_series: The `TimeSeries` we wish to plot & predict anomaly scores for.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_stamps``. If given, we use it to initialize the time series
            model. Otherwise, we assume that ``time_stamps`` immediately follows
            the training data.
        :param filter_scores: whether to filter the anomaly scores by the
            post-rule before plotting them.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param fig: a `Figure` we might want to add anomaly scores onto.
        :return: a `Figure` of the model's anomaly score predictions.
        """
        f = self.get_anomaly_label if filter_scores else self.get_anomaly_score
        scores = f(time_series, time_series_prev=time_series_prev, **kwargs)
        scores = scores.univariates[scores.names[0]]

        # Get the severity level associated with each value & convert things to
        # numpy arrays as well
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
            fig = Figure(y=time_series, y_prev=time_series_prev, anom=scores)
        else:
            if fig.y is None:
                fig.y = time_series
            fig.anom = scores
        return fig

    def plot_anomaly(
        self,
        time_series: TimeSeries,
        time_series_prev: TimeSeries = None,
        *,
        filter_scores=True,
        plot_time_series_prev=False,
        figsize=(1000, 600),
        ax=None,
    ):
        """
        Plots the time series in matplotlib as a line graph, with points in the
        series overlaid as points color-coded to indicate their severity as
        anomalies.

        :param time_series: The `TimeSeries` we wish to plot & predict anomaly scores for.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_series``. Plotted as context if given.
        :param filter_scores: whether to filter the anomaly scores by the
            post-rule before plotting them.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param figsize: figure size in pixels
        :param ax: matplotlib axes to add this plot to
        :return: matplotlib figure & axes
        """
        metric_name = time_series.names[0]
        title = f"{type(self).__name__}: Anomalies in {metric_name}"
        fig = self.get_figure(
            time_series=time_series,
            time_series_prev=time_series_prev,
            filter_scores=filter_scores,
            plot_time_series_prev=plot_time_series_prev,
        )
        return fig.plot(title=title, figsize=figsize, ax=ax)

    def plot_anomaly_plotly(
        self,
        time_series: TimeSeries,
        time_series_prev: TimeSeries = None,
        *,
        filter_scores=True,
        plot_time_series_prev=False,
        figsize=None,
    ):
        """
        Plots the time series in plotly as a line graph, with points in the
        series overlaid as points color-coded to indicate their severity as
        anomalies.

        :param time_series: The `TimeSeries` we wish to plot & predict anomaly scores for.
        :param time_series_prev: a `TimeSeries` immediately preceding
            ``time_series``. Plotted as context if given.
        :param filter_scores: whether to filter the anomaly scores by the
            post-rule before plotting them.
        :param plot_time_series_prev: whether to plot ``time_series_prev`` (and
            the model's fit for it). Only used if ``time_series_prev`` is given.
        :param figsize: figure size in pixels
        :return: plotly figure
        """
        title = f"{type(self).__name__}: Anomalies in Time Series"
        f = self.get_anomaly_label if filter_scores else self.get_anomaly_score
        scores = f(time_series, time_series_prev=time_series_prev)
        fig = MTSFigure(y=time_series, y_prev=time_series_prev, anom=scores)
        return fig.plot_plotly(title=title, figsize=figsize)


class MultipleTimeseriesDetectorMixin(MultipleTimeseriesModelMixin):
    """
    Abstract mixin for anomaly detectors supporting training on multiple time series.
    """

    @abstractmethod
    def train_multiple(
        self,
        multiple_train_data: List[TimeSeries],
        train_config=None,
        anomaly_labels: List[TimeSeries] = None,
        post_rule_train_config=None,
    ) -> List[TimeSeries]:
        """
        Trains the anomaly detector (unsupervised) and its post-rule
        (supervised, if labels are given) on the input multiple time series.

        :param multiple_train_data: a list of `TimeSeries` of metric values to train the model.
        :param anomaly_labels: a list of `TimeSeries` indicating which timestamps are anomalous. Optional.
        :param train_config: Additional training configs, if needed. Only required for some models.
        :param post_rule_train_config: The config to use for training the
            model's post-rule. The model's default post-rule train config is
            used if none is supplied here.

        :return: A list of `TimeSeries` of the model's anomaly scores on the training
            data with each element corresponds to time series from ``multiple_train_data``.
        """
        raise NotImplementedError
