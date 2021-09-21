#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Rules that use a threshold to sparsify a sequence of anomaly scores.
"""
import bisect
import logging

import numpy as np

from merlion.evaluate.anomaly import TSADMetric
from merlion.post_process.base import PostRuleBase
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


class Threshold(PostRuleBase):
    """
    Zeroes all anomaly scores whose absolute value is less than the threshold.
    """

    from merlion.evaluate.anomaly import TSADMetric

    def __init__(self, alm_threshold: float = None, abs_score=True):
        """
        :param alm_threshold: Float describing the anomaly threshold.
        :param abs_score: If 'True', consider the absolute value instead of the raw value of score.
        """
        self.alm_threshold = alm_threshold
        self.abs_score = abs_score

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        assert (
            time_series.dim == 1
        ), f"{type(self).__name__} post-rule can only be applied on single-variable time series"
        if self.alm_threshold is None:
            raise RuntimeError(f"alm_threshold is None. Please train the post-rule before attempting to use it.")
        k = time_series.names[0]
        times = time_series.univariates[k].index
        scores = time_series.univariates[k].np_values
        scores = np.where(np.isnan(scores), 0.0, scores)
        if self.abs_score:
            scores = np.where(np.abs(scores) >= self.alm_threshold, scores, 0.0)
        else:
            scores = np.where(scores >= self.alm_threshold, scores, 0.0)
        return TimeSeries({k: UnivariateTimeSeries(times, scores)})

    def train(
        self,
        anomaly_scores: TimeSeries,
        anomaly_labels: TimeSeries = None,
        metric: TSADMetric = None,
        unsup_quantile: float = None,
        max_early_sec=None,
        max_delay_sec=None,
        min_allowed_score=None,
    ) -> TimeSeries:
        """
        If ``metric`` is available, generates candidate percentiles: ``[80, 90, 95, 98, 99, 99.5, 99.9]``.
        Also considers the user-specified candidate percentile in ``unsup_quantile``. Chooses the best
        percentile based on ``metric``.

        If ``metric`` is not provided, uses ``unsup_quantile`` to choose the threshold. Otherwise,
        uses the default threshold specified in ``alm_threshold``.

        :param anomaly_scores: `TimeSeries` of anomaly scores returned by the model.
        :param anomaly_labels: `TimeSeries` of ground truth anomaly labels.
        :param metric: Metric used to evaluate the performance of candidate thresholds.
        :param unsup_quantile: User-specified quantile to use as a candidate.
        :param max_early_sec: Maximum allowed lead time (in seconds) from a detection
            to the start of an anomaly.
        :param max_delay_sec: Maximum allowed delay (in seconds) from the start of an
            anomaly and a valid detection.
        :param min_allowed_score: The minimum allowed value of the evaluation
            ``metric``. If the best candidate threshold achieves a lower value of the
            metric, we retain with the current (default) threshold.
        """
        metric = TSADMetric[metric] if isinstance(metric, str) else metric
        assert anomaly_scores.dim == 1 and (
            anomaly_labels is None or anomaly_labels.dim == 1
        ), f"{type(self).__name__} post-rule can only be applied on single-variable time series"

        k = anomaly_scores.names[0]
        scores = np.asarray(anomaly_scores.univariates[k].np_values)
        scores = np.abs(scores) if self.abs_score else scores
        default = self.alm_threshold
        if unsup_quantile is not None:
            default = np.percentile(scores, unsup_quantile * 100)
        if default is None:
            default = np.percentile(scores, 99)
        if metric is not None and any(y for t, (y,) in anomaly_labels or []):
            # We grid search thresholds over the default value, a linspace of
            # all scores seen, and some extreme percentiles of the scores
            percentiles = [80, 90, 95, 98, 99, 99.5, 99.9]
            if unsup_quantile is not None:
                percentiles.append(unsup_quantile * 100)
            candidates = np.concatenate(
                (np.percentile(scores, percentiles), np.linspace(min(scores), max(scores), num=11))
            )
            if self.alm_threshold is not None:
                candidates = np.concatenate(([self.alm_threshold], candidates))

            # Obtain the metric value at each candidate threshold
            thresh2score = {}
            for threshold in sorted(candidates):
                self.alm_threshold = threshold
                thresh2score[threshold] = metric.value(
                    ground_truth=anomaly_labels,
                    predict=self(anomaly_scores),
                    max_early_sec=max_early_sec,
                    max_delay_sec=max_delay_sec,
                )
                logger.debug(f"threshold={threshold:6.2f} --> {metric.name}={thresh2score[threshold]:.4f}")

            # The threshold is the one which achieves the highest metric value.
            # However, we stick with the default score if the best one achieves
            # a metric value under the minimum allowed score.
            t = sorted(thresh2score.keys(), key=lambda t: (thresh2score[t], t))[-1]
            if min_allowed_score is not None and thresh2score[t] < min_allowed_score:
                t = default
            elif thresh2score[t] == thresh2score[default]:
                t = default
            self.alm_threshold = t
            logger.info(f"Threshold {t:.4f} achieves {metric.name}={thresh2score[t]:.4f}.")

        elif unsup_quantile is not None:
            self.alm_threshold = np.percentile(scores, unsup_quantile * 100)

        elif self.alm_threshold is None:
            self.alm_threshold = np.max(np.abs(scores)) + 1e-8

        return self(anomaly_scores)

    def to_simple_threshold(self):
        return self


class AggregateAlarms(Threshold):
    """
    Applies basic post-filtering to a time series of anomaly scores

    1. Determine which points are anomalies by comparing the absolute value of
       their anomaly score to ``alm_threshold``
    2. Only fire an alarm when ``min_alm_in_window`` of points (within a window
       of ``alarm_window_minutes`` minutes) are labeled as anomalies.
    3. If there is an alarm, then all alarms for the next ``alm_suppress_minutes``
       minutes will be suppressed.

    Return a time series of filtered anomaly scores, where the only non-zero
    values are the anomaly scores which were marked as alarms (and not
    suppressed).
    """

    threshold_class = Threshold

    def __init__(
        self,
        alm_threshold: float = None,
        abs_score=True,
        min_alm_in_window: int = 2,
        alm_window_minutes: float = 60,
        alm_suppress_minutes: float = 120,
    ):
        self.threshold = Threshold(alm_threshold, abs_score)
        super().__init__(alm_threshold, abs_score)
        self.min_alm_in_window = min_alm_in_window
        self.alm_window_minutes = alm_window_minutes
        self.alm_suppress_minutes = alm_suppress_minutes

    @property
    def alm_threshold(self):
        return self.threshold.alm_threshold

    @alm_threshold.setter
    def alm_threshold(self, x):
        self.threshold.alm_threshold = x

    @property
    def abs_score(self):
        return self.threshold.abs_score

    @abs_score.setter
    def abs_score(self, x):
        self.threshold.abs_score = x

    @property
    def window_secs(self):
        return self.alm_window_minutes * 60

    @property
    def suppress_secs(self):
        return self.alm_suppress_minutes * 60

    def filter(self, time_series: TimeSeries) -> TimeSeries:
        k = time_series.names[0]
        times = time_series.univariates[k].time_stamps
        alarms = np.array(time_series.univariates[k].np_values)
        alarm_idxs = alarms.nonzero()[0].tolist()

        filtered = np.zeros(len(alarms))
        for idx in alarm_idxs:
            start = bisect.bisect_left(times, times[idx] - self.window_secs)
            n_recent_alms = (alarms[start : idx + 1] != 0).sum()

            start_sup = bisect.bisect_left(times, times[idx] - self.suppress_secs)
            suppress = (filtered[start_sup : idx + 1] != 0).sum() > 0

            min_alm_in_window = min(self.min_alm_in_window, idx - start)
            if n_recent_alms >= min_alm_in_window and not suppress:
                filtered[idx] = alarms[idx]

        return TimeSeries({k: UnivariateTimeSeries(times, filtered)})

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        return self.filter(self.threshold(time_series))

    def train(
        self,
        anomaly_scores: TimeSeries,
        anomaly_labels: TimeSeries = None,
        metric: TSADMetric = None,
        unsup_quantile: float = None,
        max_early_sec=None,
        max_delay_sec=None,
        min_allowed_score=None,
    ) -> TimeSeries:
        if max_early_sec is None:
            max_early_sec = self.suppress_secs
        return self.threshold_class.train(
            self,
            anomaly_scores=anomaly_scores,
            anomaly_labels=anomaly_labels,
            metric=metric,
            unsup_quantile=unsup_quantile,
            max_early_sec=max_early_sec,
            max_delay_sec=max_delay_sec,
            min_allowed_score=min_allowed_score,
        )

    def to_simple_threshold(self):
        return self.threshold


########## Adaptive thresholding code ##############


def get_adaptive_thres(x, hist_gap_thres=None, bin_sz=None):
    """
    Look for gaps in the histogram of anomaly scores (i.e. histogram bins with
    zero items inside them). Set the detection threshold to the avg bin size s.t.
    the 2 bins have a gap of hist_gap_thres or more
    """
    nbins = x.shape[0] // bin_sz  # FIXME
    hist, bins = np.histogram(x, bins=nbins)
    idx_list = np.where((hist > 0)[1:] != (hist > 0)[:-1])[0] + 1

    for i in list(range((idx_list.shape[0]) - 1)):
        if bins[idx_list[i + 1]] / bins[idx_list[i]] > hist_gap_thres:
            thres = (bins[idx_list[i + 1]] + bins[idx_list[i]]) / 2
            return x > thres, thres
    return np.zeros((x.shape[0],)), np.inf


class AdaptiveThreshold(Threshold):
    """
    Zeroes all anomaly scores whose absolute value is less than the threshold.
    """

    def __init__(self, alm_threshold: float = None, abs_score=True, bin_sz=10, default_hist_gap_thres=1.2):
        super().__init__(alm_threshold, abs_score)
        self.bin_sz = bin_sz
        self.default_hist_gap_thres = default_hist_gap_thres

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        assert (
            time_series.dim == 1
        ), f"{type(self).__name__} post-rule can only be applied on single-variable time series"
        k = time_series.names[0]
        times = time_series.univariates[k].time_stamps
        scores = np.asarray(time_series.univariates[k].np_values)
        scores = np.where(np.isnan(scores), 0.0, scores)
        if self.abs_score:
            scores = np.abs(scores)
        if self.alm_threshold is None:
            logger.info(
                "No trained threshold found (either metric was None, "
                "training data had no anomalies, or no training data "
                "given). Using default histogram gap factor "
                f"{self.default_hist_gap_thres}."
            )
            alm_threshold = get_adaptive_thres(
                np.asarray(scores), hist_gap_thres=self.default_hist_gap_thres, bin_sz=self.bin_sz
            )[1]
        else:
            alm_threshold = self.alm_threshold
        scores = np.where(scores >= alm_threshold, scores, 0.0)
        return TimeSeries({k: UnivariateTimeSeries(times, scores)})

    def train(
        self,
        anomaly_scores: TimeSeries,
        anomaly_labels: TimeSeries = None,
        metric: TSADMetric = None,
        unsup_quantile: float = None,
        max_early_sec=None,
        max_delay_sec=None,
        min_allowed_score=None,
    ) -> TimeSeries:

        metric = TSADMetric[metric] if isinstance(metric, str) else metric
        assert anomaly_scores.dim == 1 and (
            anomaly_labels is None or anomaly_labels.dim == 1
        ), f"{type(self).__name__} post-rule can only be applied on single-variable time series"

        k = anomaly_scores.names[0]
        scores = np.asarray(anomaly_scores.univariates[k].np_values)
        scores = np.abs(scores) if self.abs_score else scores
        if metric is not None and any(y for t, (y,) in anomaly_labels or []):
            candidates = [1.1, 1.2, 1.5, 1.7, 2.0]
            bin_sz_cand = {5, 10, 15, 20, 25, self.bin_sz}

            # Obtain the metric value at each candidate threshold and bin_sz
            thresh2score = {}
            for bin_sz in sorted(bin_sz_cand):
                for hist_gap in sorted(candidates):
                    self.alm_threshold = get_adaptive_thres(np.asarray(scores), hist_gap_thres=hist_gap, bin_sz=bin_sz)[
                        1
                    ]
                    score = metric.value(
                        ground_truth=anomaly_labels,
                        predict=self(anomaly_scores),
                        max_early_sec=max_early_sec,
                        max_delay_sec=max_delay_sec,
                    )
                    thresh2score[(self.alm_threshold, bin_sz)] = score
                    logger.debug(f"hist gap threshold={hist_gap:6.2f}, bin_sz={bin_sz:2} --> score={score:.4f}")

            # The threshold is the one which achieves the highest metric value
            best_cand = sorted(thresh2score.keys(), key=lambda t: (thresh2score[t], t))[-1]
            if min_allowed_score is None or thresh2score[best_cand] >= min_allowed_score:
                self.alm_threshold, self.bin_sz = best_cand
                self.alm_threshold, self.bin_sz = float(self.alm_threshold), int(self.bin_sz)
                logger.info(
                    f"Threshold {self.alm_threshold:.4f} and "
                    f"bin_sz {self.bin_sz:d} achieves a metric "
                    f"value of {thresh2score[best_cand]:.4f}."
                )
            else:
                self.alm_threshold = None
        else:
            self.alm_threshold = None

        return self(anomaly_scores)


class AdaptiveAggregateAlarms(AggregateAlarms):
    """
    Applies basic post-filtering to a time series of anomaly scores

    1. Determine which points are anomalies by comparing the absolute value of
       their anomaly score to ``alm_threshold``
    2. Only fire an alarm when ``min_alm_in_window`` of points (within a window
       of ``alarm_window_minutes`` minutes) are labeled as anomalies.
    3. If there is an alarm, then all alarms for the next ``alm_suppress_minutes``
       minutes will be suppressed.

    Return a time series of filtered anomaly scores, where the only non-zero
    values are the anomaly scores which were marked as alarms (and not
    suppressed).
    """

    threshold_class = AdaptiveThreshold

    def __init__(
        self,
        alm_threshold: float = None,
        abs_score=True,
        min_alm_in_window: int = 2,
        alm_window_minutes: float = 60,
        alm_suppress_minutes: float = 120,
        bin_sz: int = 10,
        default_hist_gap_thres: float = 1.2,
    ):
        super().__init__(
            alm_threshold=alm_threshold,
            abs_score=abs_score,
            min_alm_in_window=min_alm_in_window,
            alm_window_minutes=alm_window_minutes,
            alm_suppress_minutes=alm_suppress_minutes,
        )
        self.threshold = AdaptiveThreshold(
            alm_threshold=alm_threshold,
            abs_score=abs_score,
            bin_sz=bin_sz,
            default_hist_gap_thres=default_hist_gap_thres,
        )

    @property
    def bin_sz(self):
        return self.threshold.bin_sz

    @bin_sz.setter
    def bin_sz(self, x):
        self.threshold.bin_sz = x

    @property
    def default_hist_gap_thres(self):
        return self.threshold.default_hist_gap_thres

    @default_hist_gap_thres.setter
    def default_hist_gap_thres(self, x):
        self.threshold.default_hist_gap_thres = x
