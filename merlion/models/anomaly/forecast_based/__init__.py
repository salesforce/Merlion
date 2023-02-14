#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains all forecaster-based anomaly detectors. These models support all functionality
of both anomaly detectors (:py:mod:`merlion.models.anomaly`) and forecasters
(:py:mod:`merlion.models.forecast`).

Forecasting-based anomaly detectors are instances of an abstract `ForecastingDetectorBase`
class. Many forecasting models support anomaly detection variants, where the anomaly score
is based on the difference between the predicted and true time series value, and optionally
the model's uncertainty in its own prediction.

Note that the model will detect anomalies in only one target univariate, though the underlying
forecaster may model the full multivariate time series to predict said univariate.
"""
