#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic (multi)-seasonality detection for Facebook's Prophet.
"""
import copy
import logging
from typing import Any, Iterator, Optional, Tuple, Union

import numpy as np

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.models.forecast.prophet import Prophet
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class AutoProphetConfig(SeasonalityConfig):
    """
    Config class for `Prophet` with automatic seasonality detection.
    """

    def __init__(
        self,
        model: Union[Prophet, dict] = None,
        periodicity_strategy: Union[PeriodicityStrategy, str] = PeriodicityStrategy.All,
        **kwargs,
    ):
        model = dict(name="Prophet") if model is None else model
        super().__init__(model=model, periodicity_strategy=periodicity_strategy, **kwargs)

    @property
    def multi_seasonality(self):
        """
        :return: ``True`` because Prophet supports multiple seasonality.
        """
        return True


class AutoProphet(SeasonalityLayer):
    """
    `Prophet` with automatic seasonality detection. Automatically detects and adds
    additional seasonalities that the existing Prophet may not detect (e.g. hourly).
    """

    config_class = AutoProphetConfig

    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        seasonalities = list(super().generate_theta(train_data))
        seasonality_modes = ["additive", "multiplicative"]
        return ((seasonalities, mode) for mode in seasonality_modes)

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        seasonalities, seasonality_mode = theta
        super().set_theta(model=model, theta=seasonalities, train_data=train_data)
        model.base_model.config.seasonality_mode = seasonality_mode
        model.base_model.model.seasonality_mode = seasonality_mode

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None, **kwargs
    ) -> Tuple[Any, Prophet, Tuple[TimeSeries, Optional[TimeSeries]]]:
        candidates = []
        for seas, seasonality_mode in thetas:
            # Get the right seasonality & set the theta for this candidate model
            model = copy.deepcopy(self.model)
            seas, _, _ = super().evaluate_theta(thetas=seas, train_data=train_data, train_config=train_config, **kwargs)
            theta = seas, seasonality_mode
            self.set_theta(model=model, theta=theta, train_data=train_data)

            # Train the model and obtain its forecast on the training data
            train_result = model.train(train_data=train_data, train_config=train_config, **kwargs)
            if isinstance(model, ForecastingDetectorBase):
                train_data = model.transform(train_data)
                pred = model.train_forecast
            else:
                train_data = train_data if model.invert_transform else model.transform(train_data)
                pred, err = train_result

            # Evaluate the model based on RMSE.
            rmse = ForecastMetric.RMSE.value(train_data, pred, target_seq_index=self.model.target_seq_index)
            candidates.append({"theta": theta, "model": model, "rmse": rmse, "train_result": train_result})

        # Choose model with the best RMSE
        best = candidates[np.argmin([c["rmse"] for c in candidates])]
        logger.info(f"Best model: seasonality={best['theta'][0]}, seasonality_mode={best['theta'][1]}")
        return best["theta"], best["model"], best["train_result"]
