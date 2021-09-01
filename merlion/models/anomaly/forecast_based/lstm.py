"""
Adaptation of a LSTM neural net forecaster, to the task of anomaly detection.
"""
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.anomaly.base import DetectorConfig
from merlion.models.forecast.lstm import LSTMConfig, LSTMTrainConfig, LSTM
from merlion.post_process.threshold import AggregateAlarms

# Note: we import LSTMTrainConfig just to get it into the namespace


class LSTMDetectorConfig(LSTMConfig, DetectorConfig):
    _default_threshold = AggregateAlarms(alm_threshold=2.5)


class LSTMDetector(ForecastingDetectorBase, LSTM):
    config_class = LSTMDetectorConfig
