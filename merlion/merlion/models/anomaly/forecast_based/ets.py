"""
ETS (error, trend, seasonal) forecasting model, adapted for anomaly detection.
"""
from merlion.models.anomaly.base import NoCalibrationDetectorConfig
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.forecast.ets import ETSConfig, ETS
from merlion.post_process.threshold import AggregateAlarms


class ETSDetectorConfig(ETSConfig, NoCalibrationDetectorConfig):
    # Because the errors & residuals returned by ETS.train() are not
    # representative of the test-time errors & residuals, ETSDetector inherits
    # from NoCalibrationDetectorConfig and uses the model-predicted z-scores
    # directly as anomaly scores.
    _default_threshold = AggregateAlarms(alm_threshold=3.0)


class ETSDetector(ForecastingDetectorBase, ETS):
    config_class = ETSDetectorConfig
