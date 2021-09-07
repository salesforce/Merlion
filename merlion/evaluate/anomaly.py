#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Metrics and utilities for evaluating time series anomaly detection models.
"""
from bisect import bisect_left
from enum import Enum
from functools import partial
from typing import Tuple, Union

import numpy as np
import pandas as pd

from merlion.evaluate.base import EvaluatorBase, EvaluatorConfig
from merlion.utils import TimeSeries


def scaled_sigmoid(x, scale=2.5):
    """
    :meta private:
    """
    vals = (np.tanh(scale * (1 - x)) / np.tanh(scale)).reshape(-1)
    return np.where(x > 2.0, -1.0, np.where(x < 0, 1.0, vals))


class ScoreType(Enum):
    """
    The algorithm to use to compute true/false positives/negatives. See the technical report
    for more details on each score type. Merlion's preferred default is revised point-adjusted.
    """

    Pointwise = 0
    PointAdjusted = 1
    RevisedPointAdjusted = 2


class TSADScoreAccumulator:
    """
    Accumulator which maintains summary statistics describing an anomaly
    detection algorithm's performance. Can be used to compute many different
    time series anomaly detection metrics.
    """

    def __init__(
        self,
        num_tp_anom=0,
        num_tp_pointwise=0,
        num_tp_point_adj=0,
        num_fn_anom=0,
        num_fn_pointwise=0,
        num_fn_point_adj=0,
        num_fp=0,
        num_tn=0,
        tp_score=0.0,
        fp_score=0.0,
        tp_detection_delays=None,
        tp_anom_durations=None,
        anom_durations=None,
    ):
        self.num_tp_anom = num_tp_anom
        self.num_tp_pointwise = num_tp_pointwise
        self.num_tp_point_adj = num_tp_point_adj
        self.num_fn_anom = num_fn_anom
        self.num_fn_pointwise = num_fn_pointwise
        self.num_fn_point_adj = num_fn_point_adj
        self.num_fp = num_fp
        self.num_tn = num_tn
        self.tp_score = tp_score
        self.fp_score = fp_score
        self.tp_detection_delays = tp_detection_delays or []
        self.tp_anom_durations = tp_anom_durations or []
        self.anom_durations = anom_durations or []

    def __add__(self, acc):
        kwargs = {k: getattr(self, k) + getattr(acc, k) for k in self.__dict__}
        return TSADScoreAccumulator(**kwargs)

    def precision(self, score_type: ScoreType = ScoreType.RevisedPointAdjusted):
        if score_type is ScoreType.Pointwise:
            tp, fp = self.num_tp_pointwise, self.num_fp
        elif score_type is ScoreType.PointAdjusted:
            tp, fp = self.num_tp_point_adj, self.num_fp
        elif score_type is ScoreType.RevisedPointAdjusted:
            tp, fp = self.num_tp_anom, self.num_fp
        else:
            raise NotImplementedError(f"Cannot compute precision for score_type={score_type.name}")
        return 0.0 if tp + fp == 0 else tp / (tp + fp)

    def recall(self, score_type: ScoreType = ScoreType.RevisedPointAdjusted):
        if score_type is ScoreType.Pointwise:
            tp, fn = self.num_tp_pointwise, self.num_fn_pointwise
        elif score_type is ScoreType.PointAdjusted:
            tp, fn = self.num_tp_point_adj, self.num_fn_point_adj
        elif score_type is ScoreType.RevisedPointAdjusted:
            tp, fn = self.num_tp_anom, self.num_fn_anom
        else:
            raise NotImplementedError(f"Cannot compute recall for score_type={score_type.name}")
        return 0.0 if tp + fn == 0 else tp / (tp + fn)

    def f1(self, score_type: ScoreType = ScoreType.RevisedPointAdjusted):
        if isinstance(score_type, tuple) and len(score_type) == 2:
            prec_score_type, rec_score_type = score_type
        else:
            prec_score_type = rec_score_type = score_type
        p = self.precision(prec_score_type)
        r = self.recall(rec_score_type)
        return 0.0 if p == 0 or r == 0 else 2 * p * r / (p + r)

    def f_beta(self, score_type: ScoreType = ScoreType.RevisedPointAdjusted, beta=1.0):
        if isinstance(score_type, tuple) and len(score_type) == 2:
            prec_score_type, rec_score_type = score_type
        else:
            prec_score_type = rec_score_type = score_type
        p = self.precision(prec_score_type)
        r = self.recall(rec_score_type)
        return 0.0 if p == 0 or r == 0 else (1 + beta ** 2) * p * r / (beta ** 2 * p + r)

    def mean_time_to_detect(self):
        t = np.mean(self.tp_detection_delays) if self.tp_detection_delays else 0
        return pd.Timedelta(seconds=int(t))

    def mean_detected_anomaly_duration(self):
        t = np.mean(self.tp_anom_durations) if self.tp_anom_durations else 0
        return pd.Timedelta(seconds=int(t))

    def mean_anomaly_duration(self):
        t = np.mean(self.anom_durations) if self.anom_durations else 0
        return pd.Timedelta(seconds=int(t))

    def nab_score(self, tp_weight=1.0, fp_weight=0.11, fn_weight=1.0, tn_weight=0.0):
        """
        Computes the NAB score, given the accumulated performance metrics and
        the specified weights for different types of errors. The score is
        described in section II.C of https://arxiv.org/pdf/1510.03336.pdf.
        At a high level, this score is a cost-sensitive, recency-weighted
        accuracy measure for time series anomaly detection.

        NAB uses the following profiles for benchmarking
        (https://github.com/numenta/NAB/blob/master/config/profiles.json):

        -   standard (default)
            -   tp_weight = 1.0, fp_weight = 0.11, fn_weight = 1.0
        -   reward low false positive rate
            -   tp_weight = 1.0, fp_weight = 0.22, fn_weight = 1.0
        -   reward low false negative rate
            -   tp_weight = 1.0, fp_weight = 0.11, fn_weight = 2.0

        Note that tn_weight is ignored.

        :param tp_weight: relative weight of true positives.
        :param fp_weight: relative weight of false positives.
        :param fn_weight: relative weight of false negatives.
        :param tn_weight: relative weight of true negatives. Ignored, but
            included for completeness.
        :return: NAB score
        """
        # null: label everything as negative
        null_score = -(self.num_tp_anom + self.num_fn_anom) * fn_weight
        # perfect: detect all anomalies as early as possible, no false positives
        perfect_score = (self.num_tp_anom + self.num_fn_anom) * tp_weight
        # our score is based on our model's performance
        score = self.tp_score * tp_weight - self.fp_score * fp_weight - self.num_fn_anom * fn_weight
        return (score - null_score) / (perfect_score - null_score + 1e-8)


def accumulate_tsad_score(
    ground_truth: TimeSeries, predict: TimeSeries, max_early_sec=None, max_delay_sec=None, metric=None
) -> Union[TSADScoreAccumulator, float]:
    """
    Computes the components required to compute multiple different types of
    performance metrics for time series anomaly detection.

    :param ground_truth: A time series indicating whether each time step
        corresponds to an anomaly.
    :param predict: A time series with the anomaly score predicted for each
        time step. Detections correspond to nonzero scores.
    :param max_early_sec: The maximum amount of time (in seconds) the anomaly
        detection is allowed to occur before the actual incidence. If None, no
        early detections are allowed. Note that None is the same as 0.
    :param max_delay_sec: The maximum amount of time (in seconds) the anomaly
        detection is allowed to occur after the start of the actual incident
        (but before the end of the actual incident). If None, we allow any
        detection during the duration of the incident. Note that None differs
        from 0 because 0 means that we only permit detections that are early
        or exactly on time!
    :param metric: A function which takes a `TSADScoreAccumulator` as input and
        returns a ``float``. The `TSADScoreAccumulator` object is returned if
        ``metric`` is ``None``.
    """
    assert (
        ground_truth.dim == 1 and predict.dim == 1
    ), "Can only evaluate anomaly scores when ground truth and prediction are single-variable time series."
    ground_truth = ground_truth.univariates[ground_truth.names[0]]
    ts = ground_truth.np_time_stamps
    ys = ground_truth.np_values.astype(bool)
    i_split = np.where(ys[1:] != ys[:-1])[0] + 1

    predict = predict.univariates[predict.names[0]]
    ts_pred = predict.np_time_stamps
    ys_pred = predict.np_values.astype(bool)

    t = t_prev = ts[0]
    window_is_anomaly = ys[0]
    t0_anomaly, tf_anomaly = None, None
    num_tp_pointwise, num_tp_point_adj, num_tp_anom = 0, 0, 0
    num_fn_pointwise, num_fn_point_adj, num_fn_anom = 0, 0, 0
    num_tn, num_fp = 0, 0
    tp_score, fp_score = 0.0, 0.0
    tp_detection_delays, anom_durations, tp_anom_durations = [], [], []
    for i in [*i_split, -1]:
        t_next = ts[i] + int(i == -1)

        # Determine the boundaries of the window
        # Add buffer if it's anomalous, remove buffer if it's not
        t0, tf = t, t_next
        if window_is_anomaly:
            t0_anomaly, tf_anomaly = t0, tf
            if max_early_sec is not None and max_early_sec > 0:
                t0 = max(t_prev, t - max_early_sec)
            if max_delay_sec is not None and max_delay_sec > 0 and i != -1:
                tf = min(t_next, t + max_delay_sec)
        else:
            if max_delay_sec is not None and max_delay_sec > 0:
                t0 = min(t, t_prev + max_delay_sec)
            if max_early_sec is not None and max_early_sec > 0:
                tf = max(t, t_next - max_early_sec)

        j0 = bisect_left(ts_pred, t0)
        jf = max(bisect_left(ts_pred, tf), j0 + 1)
        window = ys_pred[j0:jf]
        if window_is_anomaly:
            anom_durations.append(tf_anomaly - t0_anomaly)
            num_tp_pointwise += sum(y != 0 for y in window)
            num_fn_pointwise += sum(y == 0 for y in window)
            if not any(window):
                num_fn_anom += 1
                num_fn_point_adj += len(window)
            # true positives are more beneficial if they occur earlier
            else:
                num_tp_anom += 1
                num_tp_point_adj += len(window)
                t_detect = ts_pred[np.where(window)[0][0] + j0]
                tp_detection_delays.append(t_detect - t0_anomaly)
                tp_anom_durations.append(tf_anomaly - t0_anomaly)
                delay = 0 if tf - t0 == 0 else (t_detect - t0) / (tf - t0)
                tp_score += sum(scaled_sigmoid(delay))
        else:
            # false positives are more severe if they occur later
            # FIXME: false positives can be fired in data spans that are
            #        not present in the original data. Should we still
            #        count these, or should we remove them from the window?
            if any(window):
                t_fp = ts_pred[np.where(window)[0] + j0]
                num_fp += len(t_fp)
                if tf != t0:
                    delays = (t_fp - t0) / (tf - t0)
                else:
                    delays = np.infty * np.ones(len(t_fp))
                fp_score += sum(scaled_sigmoid(delays))
            # do nothing for true negatives, except count them
            num_tn += sum(window == 0)

        # Advance to the next window
        t_prev = t
        t = t_next
        window_is_anomaly = not window_is_anomaly

    score_components = TSADScoreAccumulator(
        num_tp_anom=num_tp_anom,
        num_tp_pointwise=num_tp_pointwise,
        num_tp_point_adj=num_tp_point_adj,
        num_fp=num_fp,
        num_fn_anom=num_fn_anom,
        num_fn_pointwise=num_fn_pointwise,
        num_fn_point_adj=num_fn_point_adj,
        num_tn=num_tn,
        tp_score=tp_score,
        fp_score=fp_score,
        tp_detection_delays=tp_detection_delays,
        tp_anom_durations=tp_anom_durations,
        anom_durations=anom_durations,
    )
    if metric is not None:
        return metric(score_components)
    return score_components


class TSADMetric(Enum):
    """
    Enumeration of evaluation metrics for time series anomaly detection.
    For each value, the name is the metric, and the value is a partial
    function of form ``f(ground_truth, predicted, **kwargs)``
    """

    MeanTimeToDetect = partial(accumulate_tsad_score, metric=TSADScoreAccumulator.mean_time_to_detect)

    # Revised point-adjusted metrics (default)
    F1 = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.f1, score_type=ScoreType.RevisedPointAdjusted)
    )
    Precision = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.precision, score_type=ScoreType.RevisedPointAdjusted)
    )
    Recall = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.recall, score_type=ScoreType.RevisedPointAdjusted)
    )

    # Pointwise metrics
    PointwiseF1 = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.f1, score_type=ScoreType.Pointwise)
    )
    PointwisePrecision = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.precision, score_type=ScoreType.Pointwise)
    )
    PointwiseRecall = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.recall, score_type=ScoreType.Pointwise)
    )

    # Point-adjusted metrics
    PointAdjustedF1 = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.f1, score_type=ScoreType.PointAdjusted)
    )
    PointAdjustedPrecision = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.precision, score_type=ScoreType.PointAdjusted)
    )
    PointAdjustedRecall = partial(
        accumulate_tsad_score, metric=partial(TSADScoreAccumulator.recall, score_type=ScoreType.PointAdjusted)
    )

    # NAB scores
    NABScore = partial(accumulate_tsad_score, metric=TSADScoreAccumulator.nab_score)
    NABScoreLowFN = partial(accumulate_tsad_score, metric=partial(TSADScoreAccumulator.nab_score, fn_weight=2.0))
    NABScoreLowFP = partial(accumulate_tsad_score, metric=partial(TSADScoreAccumulator.nab_score, fp_weight=0.22))

    # Argus metrics
    F2 = partial(
        accumulate_tsad_score,
        metric=partial(TSADScoreAccumulator.f_beta, score_type=ScoreType.RevisedPointAdjusted, beta=2.0),
    )
    F5 = partial(
        accumulate_tsad_score,
        metric=partial(TSADScoreAccumulator.f_beta, score_type=ScoreType.RevisedPointAdjusted, beta=5.0),
    )


class TSADEvaluatorConfig(EvaluatorConfig):
    """
    Configuration class for a `TSADEvaluator`.
    """

    def __init__(self, max_early_sec: float = None, max_delay_sec: float = None, **kwargs):
        """
        :param max_early_sec: the maximum number of seconds we allow an anomaly
            to be detected early.
        :param max_delay_sec: if an anomaly is detected more than this many
            seconds after its start, it is not counted as being detected.
        """
        super().__init__(**kwargs)
        self.max_early_sec = max_early_sec
        self.max_delay_sec = max_delay_sec


class TSADEvaluator(EvaluatorBase):
    """
    Simulates the live deployment of an anomaly detection model.
    """

    config_class = TSADEvaluatorConfig

    def __init__(self, model, config):
        from merlion.models.anomaly.base import DetectorBase

        assert isinstance(model, DetectorBase)
        super().__init__(model=model, config=config)

    @property
    def max_early_sec(self):
        return self.config.max_early_sec

    @property
    def max_delay_sec(self):
        return self.config.max_delay_sec

    def _call_model(self, time_series: TimeSeries, time_series_prev: TimeSeries) -> TimeSeries:
        return self.model.get_anomaly_score(time_series, time_series_prev)

    def default_retrain_kwargs(self) -> dict:
        from merlion.models.ensemble.anomaly import DetectorEnsemble

        no_train = dict(metric=None, unsup_quantile=None, retrain_calibrator=False)
        if isinstance(self.model, DetectorEnsemble):
            return {
                "post_rule_train_config": no_train,
                "per_model_post_rule_train_configs": [no_train] * len(self.model.models),
            }
        return {"post_rule_train_config": no_train}

    def get_predict(
        self,
        train_vals: TimeSeries,
        test_vals: TimeSeries,
        train_kwargs: dict = None,
        retrain_kwargs: dict = None,
        post_process=True,
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Initialize the model by training it on an initial set of train data.
        Simulate real-time anomaly detection by the model, while re-training it
        at the desired frequency.

        :param train_vals: initial training data
        :param test_vals: all data where we want to get the model's predictions
            and compare it to the ground truth
        :param train_kwargs: dict of keyword arguments we want to use for the
            initial training process. Typically, you will want to provide the
            key "anomaly_labels" here, if you have training data with labeled
            anomalies, as well as the key "post_rule_train_config", if you want
            to use a custom training config for the model's post-rule.
        :param retrain_kwargs: dict of keyword arguments we want to use for all
            subsequent retrainings. Typically, you will not supply any this
            argument.
        :param post_process: whether to apply the model's post-rule on the
            returned results.

        :return: ``(train_result, result)``. ``train_result`` is a `TimeSeries`
            of the model's anomaly scores on ``train_vals``. ``result`` is a
            `TimeSeries` of the model's anomaly scores on ``test_vals``.
        """
        train_result, result = super().get_predict(
            train_vals=train_vals, test_vals=test_vals, train_kwargs=train_kwargs, retrain_kwargs=retrain_kwargs
        )
        if post_process:
            train_result = self.model.post_rule(train_result)
            result = self.model.post_rule(result)
        return train_result, result

    def evaluate(
        self, ground_truth: TimeSeries, predict: TimeSeries, metric: TSADMetric = None
    ) -> Union[TSADScoreAccumulator, float]:
        """
        :param ground_truth: `TimeSeries` of ground truth anomaly labels
        :param predict: `TimeSeries` of predicted anomaly scores
        :param metric: the `TSADMetric` we wish to evaluate.

        :return: the value of the evaluation ``metric``, if one is given. A
            `TSADScoreAccumulator` otherwise.
        """
        if metric is not None:
            assert isinstance(metric, TSADMetric)
            return metric.value(
                ground_truth, predict, max_early_sec=self.max_early_sec, max_delay_sec=self.max_delay_sec
            )

        return accumulate_tsad_score(
            ground_truth=ground_truth,
            predict=predict,
            metric=None,
            max_early_sec=self.max_early_sec,
            max_delay_sec=self.max_delay_sec,
        )
