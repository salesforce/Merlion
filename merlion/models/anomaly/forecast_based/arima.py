#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Classic ARIMA (AutoRegressive Integrated Moving Average) forecasting model,
adapted for anomaly detection.
"""
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.anomaly.base import DetectorConfig
from merlion.models.forecast.arima import ArimaConfig, Arima
from merlion.post_process.threshold import AggregateAlarms


class ArimaDetectorConfig(ArimaConfig, DetectorConfig):
    _default_threshold = AggregateAlarms(alm_threshold=2.5)


class ArimaDetector(ForecastingDetectorBase, Arima):
    config_class = ArimaDetectorConfig
