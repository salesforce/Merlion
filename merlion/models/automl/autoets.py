#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic seasonality detection for ETS.
"""
import warnings
import logging
import time
from copy import deepcopy
from typing import Union, Iterator, Any, Optional, Tuple
from itertools import product
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from merlion.models.forecast.ets import ETS, ETSConfig
from merlion.models.automl.base import InformationCriterion, InformationCriterionConfigMixIn
from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.utils import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class AutoETSConfig(SeasonalityConfig, InformationCriterionConfigMixIn):
    """
    Configuration class for `AutoETS`. Act as a wrapper around a `ETS` model, which automatically detects
    the seasonal_periods, error, trend, damped_trend and seasonal.
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


class AutoETS(SeasonalityLayer):
    """
    ETS with automatic seasonality detection.
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
            candidate_m = super().generate_theta(train_data=train_data)
            m, _, _ = super().evaluate_theta(thetas=candidate_m, train_data=train_data)
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
            S_range = ["add", None]
        else:
            S_range = ["add", "mul", None]
        D_range = [True, False]

        if not self.config.auto_error:
            E_range = [self.model.config.error]
        if not self.config.auto_trend:
            T_range = [self.model.config.trend]
        if not self.config.auto_seasonal:
            S_range = [self.model.config.seasonal]
        if not self.config.auto_damped:
            D_range = [self.model.config.damped_trend]

        thetas = []
        for error, trend, seasonal, damped in product(E_range, T_range, S_range, D_range):
            if trend is None and damped:
                continue
            if self.config.additive_only:
                if error == "mul" or trend == "mul" or seasonal == "mul":
                    continue
            if self.config.restrict:
                if error == "add" and (trend == "mul" or seasonal == "mul"):
                    continue
                if error == "mul" and trend == "mul" and seasonal == "add":
                    continue

            thetas.append((error, trend, seasonal, damped, m))
        return iter(thetas)

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None, **kwargs
    ) -> Tuple[Any, Optional[ETS], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:
        def _model_name(cfg: ETSConfig):
            return " ETS(err={error},trend={trend},seas={seasonal},damped={damped})".format(
                error=str(cfg.error), trend=str(cfg.trend), seasonal=str(cfg.seasonal), damped=str(cfg.damped_trend)
            )

        best = None
        y = train_data.to_pd()
        for theta in thetas:
            start = time.time()
            model = deepcopy(self.model)
            self.set_theta(model, theta, train_data)
            train_result = model._train(y, train_config=train_config)
            fit_time = time.time() - start
            ic = getattr(model.model, self.config.information_criterion.name.lower())
            logger.debug(
                "{model:47}: {ic_name}={ic:.3f}, Time={time:.2f} sec".format(
                    model=_model_name(model.config),
                    ic_name=self.config.information_criterion.name,
                    ic=ic,
                    time=fit_time,
                )
            )
            curr = {"theta": theta, "model": model, "train_result": train_result, "ic": ic}
            if best is None:
                best = curr
                logger.debug("First best model found (%.3f)" % ic)
            current_ic = best["ic"]
            if ic < current_ic:
                logger.debug("New best model found (%.3f < %.3f)" % (ic, current_ic))
                best = curr

        # Return best ETS model after post-processing its train result
        theta, model, train_result = best["theta"], best["model"], best["train_result"]
        return theta, model, model.train_post_process(train_result, **kwargs)

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        error, trend, seasonal, damped_trend, seasonal_periods = theta
        model.config.error = error
        model.config.trend = trend
        model.config.damped_trend = damped_trend
        model.config.seasonal = seasonal
        model.config.seasonal_periods = seasonal_periods
