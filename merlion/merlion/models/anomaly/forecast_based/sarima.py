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
