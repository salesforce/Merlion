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
from merlion.utils import TimeSeries


class StatThresholdConfig(DetectorConfig, NormalizingConfig):
    """
    Config class for `StatThreshold`.
    """

    _default_transform = DifferenceTransform()


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
        return True

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        train_anom_scores = pd.DataFrame(train_data.to_numpy(), columns=["anom_score"])
        return train_anom_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        time_series, _ = self.transform_time_series(time_series, time_series_prev)

        assert time_series.dim == 1, (
            f"{type(self).__name__} model only accepts univariate time "
            f"series, but time series (after transform {self.transform}) "
            f"has dimension {time_series.dim}"
        )

        anom_scores = time_series.univariates[time_series.names[0]]
        return TimeSeries({"anom_score": anom_scores})
