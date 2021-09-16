#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""Default models for anomaly detection & forecasting that balance speed and performance."""
import logging
from typing import List, Optional, Tuple, Union

from merlion.models.factory import ModelFactory
from merlion.models.base import Config, ModelWrapper
from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class DefaultModelConfig(Config):
    def __init__(self, granularity=None, **kwargs):
        super().__init__()
        self.granularity = granularity

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = set() if _skipped_keys is None else _skipped_keys
        return super().to_dict(_skipped_keys.union("transform"))


class DefaultDetectorConfig(DetectorConfig, DefaultModelConfig):
    """
    Config object for default anomaly detection model.
    """

    def __init__(self, granularity=None, threshold=None, n_threads: int = 1, **kwargs):
        """
        :param granularity: the granularity at which the input time series should
            be sampled, e.g. "5min", "1h", "1d", etc.
        :param threshold: `Threshold` object setting a default anomaly detection
            threshold in units of z-score.
        :param n_threads: the number of parallel threads to use for relevant models
        """
        super().__init__(granularity=granularity, threshold=threshold, enable_threshold=True, enable_calibrator=False)
        self.n_threads = n_threads


class DefaultDetector(ModelWrapper, DetectorBase):
    """
    Default anomaly detection model that balances efficiency with performance.
    """

    config_class = DefaultDetectorConfig

    @property
    def _default_post_rule_train_config(self):
        from merlion.evaluate.anomaly import TSADMetric

        return dict(metric=TSADMetric.F1, unsup_quantile=None)

    @property
    def granularity(self):
        return self.config.granularity

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        transform_dict = dict(name="TemporalResample", granularity=self.granularity)

        # Multivariate model is ensemble of VAE and RRCF
        n_threads = self.config.n_threads
        if train_data.dim > 1:
            self.model = ModelFactory.create(
                "DetectorEnsemble",
                enable_threshold=False,
                models=[
                    ModelFactory.create("VAE", transform=transform_dict),
                    ModelFactory.create(
                        "RandomCutForest",
                        online_updates=True,
                        parallel=n_threads > 1,
                        thread_pool_size=n_threads,
                        n_estimators=100,
                        max_n_samples=512,
                    ),
                ],
            )

        # Univariate model is ETS/RRCF/ZMS ensemble
        else:
            dt = "1h" if self.granularity is None else self.granularity
            ets_transform = dict(name="TemporalResample", granularity=dt)
            self.model = ModelFactory.create(
                "DetectorEnsemble",
                enable_threshold=False,
                models=[
                    ModelFactory.create(
                        "ETSDetector", damped_trend=True, max_forecast_steps=None, transform=ets_transform
                    ),
                    ModelFactory.create(
                        "RandomCutForest",
                        online_updates=True,
                        parallel=n_threads > 1,
                        thread_pool_size=n_threads,
                        n_estimators=100,
                        max_n_samples=512,
                    ),
                    ModelFactory.create("ZMS", n_lags=3, transform=transform_dict),
                ],
            )

        train_data = self.train_pre_process(train_data, False, False)
        train_scores = self.model.train(
            train_data=train_data,
            anomaly_labels=anomaly_labels,
            train_config=train_config,
            post_rule_train_config=post_rule_train_config,
        )
        self.train_post_rule(
            anomaly_scores=train_scores, anomaly_labels=anomaly_labels, post_rule_train_config=post_rule_train_config
        )
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        # we use get_anomaly_label() because the underlying model's calibration is
        # enabled, but its threshold is enabled
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
        return self.model.get_anomaly_label(time_series, time_series_prev)

    def get_anomaly_label(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        return super().get_anomaly_label(time_series, time_series_prev)


class DefaultForecasterConfig(ForecasterConfig, DefaultModelConfig):
    """
    Config object for default forecasting model.
    """

    def __init__(self, granularity=None, max_forecast_steps=100, target_seq_index=None, **kwargs):
        """
        :param granularity: the granularity at which the input time series should
            be sampled, e.g. "5min", "1h", "1d", etc.
        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param target_seq_index: If doing multivariate forecasting, the index of
            univariate whose value you wish to forecast.
        """
        super().__init__(
            granularity=granularity, max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index
        )


class DefaultForecaster(ModelWrapper, ForecasterBase):
    """
    Default forecasting model that balances efficiency with performance.
    """

    config_class = DefaultForecasterConfig

    @property
    def granularity(self):
        return self.config.granularity

    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        transform_dict = dict(name="TemporalResample", granularity=self.granularity)
        kwargs = dict(
            transform=transform_dict, max_forecast_steps=self.max_forecast_steps, target_seq_index=self.target_seq_index
        )

        # LGBM forecaster for multivariate data
        if train_data.dim > 1:
            self.model = ModelFactory.create(
                "LGBMForecaster",
                prediction_stride=1,
                maxlags=21,
                n_estimators=100,
                max_depth=7,
                sampling_mode="normal",
                learning_rate=0.1,
                **kwargs
            )

        # ETS for univariate data
        else:
            self.model = ModelFactory.create("ETS", damped_trend=True, **kwargs)
        train_data = self.train_pre_process(train_data, False, False)
        return self.model.train(train_data=train_data, train_config=train_config)

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Union[Tuple[TimeSeries, Optional[TimeSeries]], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        """
        Returns the model's forecast on the timestamps given.

        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for,
            or the number of steps (``int``) we wish to forecast for.
        :param time_series_prev: a list of (timestamp, value) pairs immediately
            preceding ``time_series``. If given, we use it to initialize the time
            series model. Otherwise, we assume that ``time_series`` immediately
            follows the training data.
        :param return_iqr: whether to return the inter-quartile range for the
            forecast. Note that not all models support this option.
        :param return_prev: whether to return the forecast for
            ``time_series_prev`` (and its stderr or IQR if relevant), in addition
            to the forecast for ``time_stamps``. Only used if ``time_series_prev``
            is provided.
        :return: ``(forecast, forecast_stderr)`` if ``return_iqr`` is false,
            ``(forecast, forecast_lb, forecast_ub)`` otherwise.

            - ``forecast``: the forecast for the timestamps given
            - ``forecast_stderr``: the standard error of each forecast value.
                May be ``None``.
            - ``forecast_lb``: 25th percentile of forecast values for each timestamp
            - ``forecast_ub``: 75th percentile of forecast values for each timestamp
        """
        if time_series_prev is not None:
            time_series_prev = self.transform(time_series_prev)
        return self.model.forecast(
            time_stamps=time_stamps, time_series_prev=time_series_prev, return_iqr=return_iqr, return_prev=return_prev
        )
