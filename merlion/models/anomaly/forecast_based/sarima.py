#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Seasonal ARIMA (SARIMA) forecasting model, adapted for anomaly detection.
"""
from merlion.models.anomaly.base import DetectorConfig
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.forecast.sarima import SarimaConfig, Sarima
from merlion.post_process.threshold import AggregateAlarms


class SarimaDetectorConfig(SarimaConfig, DetectorConfig):
    _default_threshold = AggregateAlarms(alm_threshold=2.5)


class SarimaDetector(ForecastingDetectorBase, Sarima):
    config_class = SarimaDetectorConfig
