#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic hyperparamter selection for ETS.
"""
from collections import OrderedDict
import logging
from typing import Union, Iterator, Tuple

import numpy as np
import pandas as pd

from merlion.models.forecast.ets import ETS
from merlion.models.automl.base import InformationCriterion, ICConfig, ICAutoMLForecaster
from merlion.models.automl.search import GridSearch
from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class AutoETSConfig(SeasonalityConfig, ICConfig):
    """
    Configuration class for `AutoETS`. Act as a wrapper around a `ETS` model, which automatically detects
    the hyperparameters ``seasonal_periods``, ``error``, ``trend``, ``damped_trend`` and ``seasonal``.
    """

    # This is adapted from ets.R from forecast package
    def __init__(
        self,
        model: Union[ETS, dict] = None,
        auto_seasonality: bool = True,
        auto_error: bool = True,
        auto_trend: bool = True,
        auto_seasonal: bool = True,
        auto_damped: bool = True,
        periodicity_strategy: PeriodicityStrategy = PeriodicityStrategy.ACF,
        information_criterion: InformationCriterion = InformationCriterion.AIC,
        additive_only: bool = False,
        allow_multiplicative_trend: bool = False,
        restrict: bool = True,
        **kwargs,
    ):
        """
        :param auto_seasonality: Whether to automatically detect the seasonality.
        :param auto_error: Whether to automatically detect the error components.
        :param auto_trend: Whether to automatically detect the trend components.
        :param auto_seasonal: Whether to automatically detect the seasonal components.
        :param auto_damped: Whether to automatically detect the damped trend components.
        :param additive_only: If True, the search space will only consider additive models.
        :param allow_multiplicative_trend: If True, models with multiplicative trend are allowed in the search space.
        :param restrict: If True, the models with infinite variance will not be allowed in the search space.
        """
        model = dict(name="ETS") if model is None else model
        super().__init__(
            model=model,
            periodicity_strategy=periodicity_strategy,
            information_criterion=information_criterion,
            **kwargs,
        )

        self.auto_seasonality = auto_seasonality
        self.auto_trend = auto_trend
        self.auto_seasonal = auto_seasonal
        self.auto_error = auto_error
        self.auto_damped = auto_damped
        self.additive_only = additive_only
        self.allow_multiplicative_trend = allow_multiplicative_trend
        self.restrict = restrict


class AutoETS(ICAutoMLForecaster, SeasonalityLayer):
    """
    Wrapper around a `ETS` model, which automatically detects
    the hyperparameters ``seasonal_periods``, ``error``, ``trend``, ``damped_trend`` and ``seasonal``.
    """

    config_class = AutoETSConfig

    def __init__(self, config: AutoETSConfig):
        super().__init__(config)

    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        """
        generate [theta]. theta is a list of parameter combination [error, trend, damped_trend, seasonal]
        """
        y = train_data.univariates[self.target_name].np_values

        # check the size of y
        n_samples = y.shape[0]
        if n_samples <= 3:
            self.information_criterion = InformationCriterion.AIC

        # auto-detect seasonality if desired, otherwise just get it from seasonal order
        if self.config.auto_seasonality:
            candidate_m = SeasonalityLayer.generate_theta(self, train_data=train_data)
            m, _, _ = SeasonalityLayer.evaluate_theta(self, thetas=candidate_m, train_data=train_data)
        else:
            if self.model.config.seasonal_periods is None:
                m = 1
            else:
                m = max(1, self.model.config.seasonal_periods)

        # set the parameters ranges for error, trend, damped_trend and seasonal
        if np.any(y <= 0):
            E_range = ["add"]
            T_range = ["add", None]
        else:
            E_range = ["add", "mul"]
            if self.config.allow_multiplicative_trend:
                T_range = ["add", "mul", None]
            else:
                T_range = ["add", None]

        if m <= 1 or y.shape[0] <= m:
            m = 1
            S_range = [None]
        elif np.any(y <= 0):
            S_range = ["add"]
        else:
            S_range = ["add", "mul"]
        D_range = [True, False]

        if not self.config.auto_error:
            E_range = [self.model.config.error]
        if not self.config.auto_trend:
            T_range = [self.model.config.trend]
        if not self.config.auto_seasonal:
            S_range = [self.model.config.seasonal]
        if not self.config.auto_damped:
            D_range = [self.model.config.damped_trend]

        # Construct a grid search object
        param_values = OrderedDict(error=E_range, trend=T_range, seasonal=S_range, damped=D_range, m=[m])
        restrictions = [dict(trend=None, damped=True)]
        if self.config.additive_only:
            restrictions.extend([dict(error="mul"), dict(trend="mul"), dict(seasonal="mul")])
        if self.config.restrict:
            restrictions.append(dict(error="add", trend="mul"))
            restrictions.append(dict(error="add", seasonal="mul"))
            restrictions.append(dict(error="mul", trend="mul", seasonal="add"))
        return iter(GridSearch(param_values=param_values, restrictions=restrictions))

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        m = theta["m"]
        model.config.error = theta["error"]
        model.config.trend = theta["trend"]
        model.config.damped_trend = theta["damped"]
        model.config.seasonal = None if m <= 1 else theta["seasonal"]
        model.config.seasonal_periods = m

    def _model_name(self, theta):
        return f"ETS(err={theta['error']},trend={theta['trend']},seas={theta['seasonal']},damped={theta['damped']})"

    def get_ic(self, model, train_data: pd.DataFrame, train_result: Tuple[pd.DataFrame, pd.DataFrame]) -> float:
        ic = self.config.information_criterion.name
        if ic in ["AIC", "BIC", "AICc"]:
            return getattr(model.base_model.model, ic.lower())
        else:
            raise ValueError(f"{type(self.model).__name__} doesn't support information criterion {ic}")
