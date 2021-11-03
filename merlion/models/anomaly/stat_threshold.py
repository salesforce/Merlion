#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Simple static thresholding model for anomaly detection.
"""
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

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=True)

        train_anom_scores = train_data.univariates[train_data.names[0]]
        train_anom_scores = TimeSeries({"anom_score": train_anom_scores})
        self.train_post_rule(train_anom_scores, anomaly_labels, post_rule_train_config)
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
