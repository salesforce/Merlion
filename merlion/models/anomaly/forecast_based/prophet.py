"""
Adaptation of Facebook's Prophet forecasting model to anomaly detection.
"""

from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.anomaly.base import DetectorConfig
from merlion.models.forecast.prophet import ProphetConfig, Prophet
from merlion.post_process.threshold import AggregateAlarms
from merlion.transform.moving_average import DifferenceTransform


class ProphetDetectorConfig(ProphetConfig, DetectorConfig):
    _default_transform = DifferenceTransform()
    _default_threshold = AggregateAlarms(alm_threshold=3)


class ProphetDetector(ForecastingDetectorBase, Prophet):
    config_class = ProphetDetectorConfig
