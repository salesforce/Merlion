#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic hyperparameter selection for Facebook's Prophet.
"""
import copy
import logging
from typing import Any, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from merlion.models.automl.base import InformationCriterion, ICConfig, ICAutoMLForecaster
from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.models.forecast.prophet import Prophet
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class AutoProphetConfig(SeasonalityConfig, ICConfig):
    """
    Config class for `Prophet` with automatic seasonality detection.
    """

    def __init__(
        self,
        model: Union[Prophet, dict] = None,
        periodicity_strategy: Union[PeriodicityStrategy, str] = PeriodicityStrategy.All,
        information_criterion: InformationCriterion = InformationCriterion.AIC,
        **kwargs,
    ):
        model = dict(name="Prophet") if model is None else model
        super().__init__(
            model=model,
            periodicity_strategy=periodicity_strategy,
            information_criterion=information_criterion,
            **kwargs,
        )

    @property
    def multi_seasonality(self):
        """
        :return: ``True`` because Prophet supports multiple seasonality.
        """
        return True


class AutoProphet(ICAutoMLForecaster, SeasonalityLayer):
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
        seasonalities, _, _ = SeasonalityLayer.evaluate_theta(self, thetas=iter(seasonalities), train_data=train_data)
        SeasonalityLayer.set_theta(self, model=model, theta=seasonalities, train_data=train_data)
        model.base_model.config.seasonality_mode = seasonality_mode
        model.base_model.model.seasonality_mode = seasonality_mode

    def _model_name(self, theta) -> str:
        seas, mode = theta
        return f"Prophet(seasonalities={seas}, seasonality_mode={mode})"

    def get_ic(self, model, train_data_df: pd.DataFrame, train_result: Tuple[pd.DataFrame, pd.DataFrame]) -> float:
        pred, stderr = train_result
        log_like = norm.logpdf((pred.values - train_data_df.values) / stderr.values).sum()
        n_params = sum(len(v.flatten()) for k, v in model.base_model.model.params.items() if k != "trend")
        ic_id = self.config.information_criterion
        if ic_id is InformationCriterion.AIC:
            ic = 2 * n_params - 2 * log_like.sum()
        elif ic_id is InformationCriterion.BIC:
            ic = n_params * np.log(len(y)) - 2 * log_like
        elif ic_id is InformationCriterion.AICc:
            ic = 2 * n_params - 2 * log_like + (2 * n_params * (n_params + 1)) / (len(y) - n_params - 1)
        else:
            raise ValueError(f"{type(self.model).__name__} doesn't support information criterion {ic_id.name}")
        return ic
