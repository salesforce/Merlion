#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Post-rule to transform anomaly scores to follow a standard normal distribution.
"""
import logging
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator

from merlion.post_process.base import PostRuleBase
from merlion.utils import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class AnomScoreCalibrator(PostRuleBase):
    """
    Learns a monotone function which reshapes an input sequence of anomaly scores,
    to follow a standard normal distribution. This makes the anomaly scores from
    many diverse models interpretable as z-scores.
    """

    def __init__(self, max_score: float, abs_score: bool = True, anchors: List[Tuple[float, float]] = None):
        """
        :param max_score: the maximum possible uncalibrated score
        :param abs_score: whether to consider the absolute values of the
            anomaly scores, rather than the raw value.
        :param anchors: a sequence of (x, y) pairs mapping an uncalibrated
            anomaly score to a calibrated anomaly score. Optional, as this
            will be set by `AnomScoreCalibrator.train`.
        """
        self.max_score = max_score
        self.abs_score = abs_score
        self.anchors = anchors

    @property
    def anchors(self):
        return self._anchors

    @anchors.setter
    def anchors(self, anchors):
        """
        :return: a sequence of (x, y) pairs mapping an uncalibrated
            anomaly score to a calibrated anomaly score.
        """
        if anchors is None or len(anchors) < 2:
            self._anchors = None
            self.interpolator = None
        else:
            self._anchors = anchors
            self.interpolator = PchipInterpolator(*zip(*anchors))

    def train(self, anomaly_scores: TimeSeries, retrain_calibrator=False) -> TimeSeries:
        """
        :param anomaly_scores: `TimeSeries` of raw anomaly scores that we will use
            to train the calibrator.
        :param retrain_calibrator: Whether to re-train the calibrator on a new
            sequence of anomaly scores, if it has already been trained once.
            In practice, we find better results if this is ``False``.
        """
        if self.interpolator is not None and not retrain_calibrator:
            return self(anomaly_scores)

        x = anomaly_scores.to_pd().values[:, 0]
        if self.abs_score:
            x = np.abs(x)

        targets = [0, 0, 0.5, 1, 1.5, 2]
        inputs = np.quantile(x, 2 * norm.cdf(targets) - 1).tolist()

        # ub is an upper bound on E[max(X_1, ..., X_n)], for X_i ~ N(0, 1)
        ub = self.expected_max(len(x), ub=True)
        x_max = x.max()
        if self.max_score < x_max:
            logger.warning(
                f"Obtained max score of {x_max:.2f}, but self.max_score "
                f"is only {self.max_score:.2f}. Updating self.max_score "
                f"to {x_max * 2:.2f}."
            )
            self.max_score = x_max * 2
        if ub > 4:
            targets.append(ub)
            inputs.append(x.max())
            targets.append(ub + 1)
            inputs.append(min(self.max_score, 2 * x_max))
        else:
            targets.append(5)
            inputs.append(min(self.max_score, 2 * x_max))

        targets = np.asarray(targets)
        inputs = np.asarray(inputs)
        valid = np.concatenate(([True], np.abs(inputs[1:] - inputs[:-1]) > 1e-8))
        self.anchors = list(zip(inputs[valid], targets[valid]))
        return self(anomaly_scores)

    @staticmethod
    def expected_max(n, ub=False):
        """
        :meta private:
        """
        if ub:
            return np.sqrt(2 * np.log(n))
        g = np.euler_gamma
        return (1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / np.e / n)

    def __call__(self, anomaly_scores: TimeSeries) -> TimeSeries:
        if self.interpolator is None:
            return anomaly_scores
        x = anomaly_scores.to_pd().values[:, 0]
        b = self.anchors[-1][0]
        m = self.interpolator.derivative()(self.anchors[-1][0])
        if self.abs_score:
            vals = np.maximum(self.interpolator(np.abs(x)), 0) * np.sign(x)
            idx = np.abs(x) > b
            if idx.any():
                sub = x[idx]
                vals[idx] = np.sign(sub) * ((np.abs(sub) - b) * m + self.interpolator(b))
        else:
            vals = self.interpolator(x)
            idx = x > b
            if idx.any():
                vals[idx] = (x[idx] - b) * m + self.interpolator(b)
        return UnivariateTimeSeries(anomaly_scores.time_stamps, vals, anomaly_scores.names[0]).to_ts()
