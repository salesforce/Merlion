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
import pandas as pd
from scipy.stats import norm

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.automl.base import InformationCriterion, InformationCriterionConfigMixIn
from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.models.forecast.prophet import Prophet
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class AutoProphetConfig(SeasonalityConfig, InformationCriterionConfigMixIn):
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
        best = None
        y = pd.DataFrame(train_data.univariates[self.model.target_name].to_pd())

        for i, (seas, mode) in enumerate(thetas):
            # Get the right seasonality & set the theta for this candidate model
            model = copy.deepcopy(self.model)
            seas, _, _ = super().evaluate_theta(thetas=seas, train_data=train_data, train_config=train_config, **kwargs)
            theta = seas, mode
            self.set_theta(model=model, theta=theta, train_data=train_data)

            # Train the model & evaluate it based on AIC/BIC. We use the method suggested by the author to compute the
            # log likelihood: https://github.com/facebook/prophet/issues/549#issuecomment-435482584
            pred, stderr = model._train(train_data=train_data.to_pd(), train_config=train_config)
            log_like = norm.logpdf((pred.values - y.values) / stderr.values).sum()
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

            logger.debug(f"Prophet(seas={seas}, mode={mode}) : {ic_id.name} = {ic:.3f}")
            curr = {"theta": theta, "model": model, "train_result": (pred, stderr), "ic": ic}
            if best is None:
                best = curr
                logger.debug("First best model found (%.3f)" % ic)
            current_ic = best["ic"]
            if ic < current_ic:
                logger.debug("New best model found (%.3f < %.3f)" % (ic, current_ic))
                best = curr

        # Return best Prophet model after post-processing its train result
        theta, model, train_result = best["theta"], best["model"], best["train_result"]
        return theta, model, model.train_post_process(train_result, **kwargs)
