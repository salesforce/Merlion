"""
Contains all forecaster-based anomaly detectors. These models support all functionality
of both anomaly detectors (:py:mod:`merlion.models.anomaly`) and forecasters
(:py:mod:`merlion.models.forecast`).

Forecasting-based anomaly detectors are instances of an abstract `ForecastingDetectorBase`
class. Many forecasting models support anomaly detection variants, where the anomaly score
is based on the difference between the predicted and true time series value, and optionally
the model's uncertainty in its own prediction.
"""
