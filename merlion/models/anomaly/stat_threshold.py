#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Simple static thresholding model for anomaly detection.
"""
import pandas as pd

from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.transform.moving_average import DifferenceTransform


class StatThresholdConfig(DetectorConfig, NormalizingConfig):
    """
    Config class for `StatThreshold`.
    """

    def __init__(self, target_seq_index: int = None, **kwargs):
        """
        :param target_seq_index (optional): The index of the univariate whose value we are considering thresholds of.
            If not provided, the model only works for univariate data.
        """
        super().__init__(**kwargs)
        self.target_seq_index = target_seq_index


class StatThreshold(DetectorBase):
    """
    Anomaly detection based on a static threshold.
    """

    config_class = StatThresholdConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return self.config.target_seq_index is None

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        return pd.DataFrame(train_data.iloc[:, self.config.target_seq_index or 0])

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        return pd.DataFrame(time_series.iloc[:, self.config.target_seq_index or 0])
