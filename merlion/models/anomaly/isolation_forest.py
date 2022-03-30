#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The classic isolation forest model for anomaly detection.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as skl_IsolationForest

from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import Shingle
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


class IsolationForestConfig(DetectorConfig):
    """
    Configuration class for `IsolationForest`.
    """

    _default_transform = TransformSequence([DifferenceTransform(), Shingle(size=2, stride=1)])

    def __init__(self, max_n_samples: int = None, n_estimators: int = 100, **kwargs):
        """
        :param max_n_samples: Maximum number of samples to allow the isolation
            forest to train on. Specify ``None`` to use all samples in the
            training data.
        :param n_estimators: number of trees in the isolation forest.
        """
        self.max_n_samples = 1.0 if max_n_samples is None else max_n_samples
        self.n_estimators = n_estimators
        # Isolation forest's uncalibrated scores are between 0 and 1
        kwargs["max_score"] = 1.0
        super().__init__(**kwargs)


class IsolationForest(DetectorBase):
    """
    The classic isolation forest algorithm, proposed in
    `Liu et al. 2008 <https://ieeexplore.ieee.org/document/4781136>`_
    """

    config_class = IsolationForestConfig

    def __init__(self, config: IsolationForestConfig):
        super().__init__(config)
        self.model = skl_IsolationForest(
            max_samples=config.max_n_samples, n_estimators=config.n_estimators, random_state=0
        )

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        times, train_values = train_data.index, train_data.values
        self.model.fit(train_values)
        train_scores = -self.model.score_samples(train_values)
        return pd.DataFrame(train_scores, index=times, columns=["anom_score"])

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        time_series, _ = self.transform_time_series(time_series, time_series_prev)
        time_stamps, values = zip(*time_series.align())
        values = np.asarray(values)

        # Return negative of model's score, since model scores are in (0, -1],
        # where more negative = more anomalous
        scores = -self.model.score_samples(np.array(values))
        return TimeSeries({"anom_score": UnivariateTimeSeries(time_stamps, scores)})
