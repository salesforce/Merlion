#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""Default models for anomaly detection & forecasting that balance speed and performance."""
import logging
from typing import Optional, Tuple

from merlion.models.factory import ModelFactory
from merlion.models.layers import LayeredDetector, LayeredForecaster, LayeredModelConfig
from merlion.models.anomaly.base import DetectorBase
from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class DefaultDetectorConfig(LayeredModelConfig):
    """
    Config object for default anomaly detection model.
    """

    def __init__(self, model=None, granularity=None, n_threads: int = 1, **kwargs):
        """
        :param granularity: the granularity at which the input time series should
            be sampled, e.g. "5min", "1h", "1d", etc.
        :param n_threads: the number of parallel threads to use for relevant models
        """
        self.granularity = granularity
        self.n_threads = n_threads
        super().__init__(model=model, **kwargs)
        assert self.base_model is None or isinstance(self.base_model, DetectorBase)


class DefaultDetector(LayeredDetector):
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
                **self.config.model_kwargs,
            )

        # Univariate model is ETS/RRCF/ZMS ensemble
        else:
            dt = "1h" if self.granularity is None else self.granularity
            ets_transform = dict(name="TemporalResample", granularity=dt)
            self.model = ModelFactory.create(
                "DetectorEnsemble",
                models=[
                    ModelFactory.create(
                        "AutoETS", model=dict(name="ETSDetector"), damped_trend=True, transform=ets_transform
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
                **self.config.model_kwargs,
            )

        return super().train(
            train_data=train_data,
            anomaly_labels=anomaly_labels,
            train_config=train_config,
            post_rule_train_config=post_rule_train_config,
        )


class DefaultForecasterConfig(LayeredModelConfig):
    """
    Config object for default forecasting model.
    """

    def __init__(self, model=None, max_forecast_steps=100, target_seq_index=None, granularity=None, **kwargs):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.
            Required for some models like `MSES` and `LGBMForecaster`.
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param granularity: the granularity at which the input time series should
            be sampled, e.g. "5min", "1h", "1d", etc.
        """
        self.granularity = granularity
        super().__init__(
            model=model, max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs
        )
        assert self.base_model is None or isinstance(self.base_model, ForecasterBase)


class DefaultForecaster(LayeredForecaster):
    """
    Default forecasting model that balances efficiency with performance.
    """

    config_class = DefaultForecasterConfig

    @property
    def granularity(self):
        return self.config.granularity

    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        transform_dict = dict(name="TemporalResample", granularity=self.granularity)
        kwargs = dict(transform=transform_dict, **self.config.model_kwargs)

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
                **kwargs,
            )

        # ETS for univariate data
        else:
            self.model = ModelFactory.create("AutoETS", damped_trend=True, **kwargs)
        return super().train(train_data=train_data, train_config=train_config)
