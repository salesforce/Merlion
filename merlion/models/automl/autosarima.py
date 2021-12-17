#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic hyperparameter selection for SARIMA.
"""
from collections import Iterator
from copy import copy, deepcopy
import logging
from typing import Any, Optional, Tuple, Union

import numpy as np

from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.models.forecast.sarima import Sarima
from merlion.transform.resample import TemporalResample
from merlion.utils import autosarima_utils, TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class AutoSarimaConfig(SeasonalityConfig):
    """
    Configuration class for `AutoSarima`. Acts as a wrapper around a `Sarima` model, which automatically detects
    the seasonality, (seasonal) differencing order, and (seasonal) AR/MA orders. If a non-numeric value is specified
    for any of the relevant parameters in the order or seasonal order, we assume that the user wishes to detect that
    parameter automatically.

    .. note::

        The automatic selection of AR, MA, seasonal AR, and seasonal MA parameters is implemented in a coupled way.
        The user must specify all of these parameters explicitly to avoid automatic selection.
    """

    _default_transform = TemporalResample()

    def __init__(
        self,
        model: Union[Sarima, dict] = None,
        auto_seasonality: bool = True,
        periodicity_strategy: PeriodicityStrategy = PeriodicityStrategy.ACF,
        auto_pqPQ: bool = True,
        auto_d: bool = True,
        auto_D: bool = True,
        maxiter: int = None,
        max_k: int = 100,
        max_dur: float = 3600,
        approximation: bool = None,
        approx_iter: int = None,
        **kwargs,
    ):
        """
        :param auto_seasonality: Whether to automatically detect the seasonality.
        :param auto_pqPQ: Whether to automatically choose AR/MA orders ``p, q`` and seasonal AR/MA orders ``P, Q``.
        :param auto_d: Whether to automatically choose the difference order ``d``.
        :param auto_D: Whether to automatically choose the seasonal difference order ``D``.
        :param maxiter: The maximum number of iterations to perform
        :param max_k: Maximum number of models considered in the stepwise search
        :param max_dur: Maximum training time considered in the stepwise search
        :param approximation: Whether to use ``approx_iter`` iterations (instead
            of ``maxiter``) to speed up computation. If ``None``, we use
            approximation mode when the training data is too long (>150), or when
            the length off the period is too high (``periodicity > 12``).
        :param approx_iter: The number of iterations to perform in approximation mode
        """
        if model is None:
            model = dict(name="Sarima", transform=dict(name="Identity"))
        super().__init__(model=model, periodicity_strategy=periodicity_strategy, **kwargs)

        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        self.auto_seasonality = auto_seasonality or not isinstance(m, (int, float))
        self.auto_pqPQ = auto_pqPQ or any(not isinstance(x, (int, float)) for x in (p, q, P, Q))
        self.auto_d = auto_d or not isinstance(d, (int, float))
        self.auto_D = auto_D or not isinstance(D, (int, float))
        self.maxiter = maxiter
        self.max_k = max_k
        self.max_dur = max_dur
        self.approximation = approximation
        self.approx_iter = approx_iter

    @property
    def order(self):
        return self.model.order

    @property
    def seasonal_order(self):
        return self.model.seasonal_order


class AutoSarima(SeasonalityLayer):

    config_class = AutoSarimaConfig
    require_even_sampling = True
    require_univariate = True

    def _generate_sarima_parameters(self, train_data: TimeSeries) -> dict:
        y = train_data.univariates[self.target_name].np_values
        X = None

        order = list(self.config.order)
        seasonal_order = list(self.config.seasonal_order)
        approximation = self.config.approximation
        maxiter = self.config.maxiter
        approx_iter = self.config.approx_iter
        max_k = self.config.max_k
        max_dur = self.config.max_dur

        # These should be set in config
        stationary = False
        seasonal_test = "seas"
        method = "lbfgs"
        test = "kpss"
        stepwise = True
        max_d = 2
        start_p = 2
        max_p = 5
        start_q = 2
        max_q = 5
        max_D = 1
        start_P = 1
        max_P = 2
        start_Q = 1
        max_Q = 2
        relative_improve = 0
        trend = None
        information_criterion = "aic"

        # auto-detect seasonality if desired, otherwise just get it from seasonal order
        if self.config.auto_seasonality:
            candidate_m = super().generate_theta(train_data=train_data)
            m, _, _ = super().evaluate_theta(thetas=candidate_m, train_data=train_data)
        else:
            m = max(1, seasonal_order[-1])

        #  adjust max p,q,P,Q start p,q,P,Q
        n_samples = len(y)
        max_p = int(min(max_p, np.floor(n_samples / 3)))
        max_q = int(min(max_q, np.floor(n_samples / 3)))
        max_P = int(min(max_P, np.floor(n_samples / 3 / m)))
        max_Q = int(min(max_Q, np.floor(n_samples / 3 / m)))
        start_p = min(start_p, max_p)
        start_q = min(start_q, max_q)
        start_P = min(start_P, max_Q)
        start_Q = min(start_Q, max_Q)

        #  set the seasonal differencing order with statistical test
        D = None if self.config.auto_D else seasonal_order[1]
        D = 0 if m == 1 else D
        xx = y.copy()
        if stationary:
            D = 0
        elif D is None:
            D = autosarima_utils.nsdiffs(xx, m=m, max_D=max_D, test=seasonal_test)
            if D > 0:
                dx = autosarima_utils.diff(xx, differences=D, lag=m)
                if dx.shape[0] == 0:
                    D = D - 1
        dx = autosarima_utils.diff(xx, differences=D, lag=m) if D > 0 else xx
        logger.info(f"Seasonal difference order is {str(D)}")

        #  set the differencing order by estimating the number of orders
        #  it would take in order to make the time series stationary
        d = autosarima_utils.ndiffs(dx, alpha=0.05, max_d=max_d, test=test) if self.config.auto_d else order[1]
        if stationary:
            d = 0
        if d > 0:
            dx = autosarima_utils.diff(dx, differences=d, lag=1)
        logger.info(f"Difference order is {str(d)}")

        # pqPQ is an indicator about whether need to automatically select
        # AR, MA, seasonal AR and seasonal MA parameters
        pqPQ = not self.config.auto_pqPQ

        # automatically detect whether to use approximation method and the periodicity
        if approximation is None:
            approximation = (y.shape[0] > 150) or (m > 12)

        # check the size of y
        n_samples = y.shape[0]
        if n_samples <= 3:
            information_criterion = "aic"

        if m > 1:
            if max_P > 0:
                max_p = min(max_p, m - 1)
            if max_Q > 0:
                max_q = min(max_q, m - 1)
        if (d + D) in (0, 1):
            trend = "c"

        if n_samples < 10:
            start_p = min(start_p, 1)
            start_q = min(start_q, 1)
            start_P = start_Q = 0

        # seed p, q, P, Q vals
        p = min(start_p, max_p)
        q = min(start_q, max_q)
        P = min(start_P, max_P)
        Q = min(start_Q, max_Q)

        refititer = maxiter

        return_dict = dict(
            y=y,
            X=X,
            p=p,
            d=d,
            q=q,
            P=P,
            D=D,
            Q=Q,
            m=m,
            dx=dx,
            pqPQ=pqPQ,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            max_P=max_P,
            max_D=max_D,
            max_Q=max_Q,
            trend=trend,
            method=method,
            maxiter=maxiter,
            information_criterion=information_criterion,
            relative_improve=relative_improve,
            approximation=approximation,
            max_k=max_k,
            max_dur=max_dur,
            approx_iter=approx_iter,
            refititer=refititer,
            stepwise=stepwise,
            order=order,
            seasonal_order=seasonal_order,
        )
        return return_dict

    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        """
        generate [action, theta]. action is an indicator for stepwise seach (stepwsie) of
        p, q, P, Q, trend parameters or use a predefined parameter combination (pqPQ)
        theta is a list of parameter combination [order, seasonal_order, trend]
        """

        val_dict = self._generate_sarima_parameters(train_data)
        y = val_dict["y"]
        pqPQ = val_dict["pqPQ"]
        order = val_dict["order"]
        seasonal_order = val_dict["seasonal_order"]
        d = val_dict["d"]
        D = val_dict["D"]
        m = val_dict["m"]
        dx = val_dict["dx"]
        stepwise = val_dict["stepwise"]

        action = None
        trend = None

        # input time-series is completely constant
        if np.max(y) == np.min(y):
            order = [0, 0, 0]
            seasonal_order = [0, 0, 0, 0]
        elif pqPQ:
            action = "pqPQ"
            order[1] = d
            seasonal_order[1] = D
            seasonal_order[3] = m
            if m == 1:
                seasonal_order = [0, 0, 0, m]
        elif np.max(dx) == np.min(dx):
            order = [0, 0, 0]
            seasonal_order = (0, 0, 0, m) if m == 1 else (0, D, 0, m)
        elif stepwise:
            action = "stepwise"

        return iter([{"action": action, "theta": [order, seasonal_order, trend], "val_dict": val_dict}])

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None
    ) -> Tuple[Any, Optional[Sarima], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:

        theta_value = next(thetas)

        # preprocess
        train_config = copy(train_config) if train_config is not None else {}
        if "enforce_stationarity" not in train_config:
            train_config["enforce_stationarity"] = False
        if "enforce_invertibility" not in train_config:
            train_config["enforce_invertibility"] = False
        val_dict = theta_value["val_dict"]
        y = val_dict["y"]
        X = val_dict["X"]
        method = val_dict["method"]
        maxiter = val_dict["maxiter"]
        information_criterion = val_dict["information_criterion"]
        approximation = val_dict["approximation"]
        approx_iter = val_dict["approx_iter"]

        # use zero model to automatically detect the optimal maxiter
        if maxiter is None:
            maxiter = autosarima_utils.detect_maxiter_sarima_model(**val_dict)

        if theta_value["action"] == "stepwise":
            refititer = maxiter
            if approximation:
                if approx_iter is None:
                    maxiter = max(int(maxiter / 5), 1)
                else:
                    maxiter = approx_iter
                logger.info(f"Fitting models using approximations(approx_iter is {str(maxiter)}) to speed things up")
            train_config["maxiter"] = maxiter

            # stepwise search
            stepwise_search = autosarima_utils._StepwiseFitWrapper(**{**val_dict, **train_config})
            filtered_models_ics = stepwise_search.stepwisesearch()

            if approximation:
                logger.debug(f"Now re-fitting the best model(s) without approximations...")
                if len(filtered_models_ics) > 0:
                    best_model_theta = filtered_models_ics[0][1]
                    best_model_fit = autosarima_utils._refit_sarima_model(
                        filtered_models_ics[0][0],
                        filtered_models_ics[0][2],
                        method,
                        maxiter,
                        refititer,
                        information_criterion,
                    )
                    logger.info(f"Best model: {autosarima_utils._model_name(best_model_fit.model)}")
                else:
                    raise ValueError("Could not successfully fit a viable SARIMA model")
            else:
                if len(filtered_models_ics) > 0:
                    best_model_fit = filtered_models_ics[0][0]
                    best_model_theta = filtered_models_ics[0][1]
                    logger.info(f"Best model: {autosarima_utils._model_name(best_model_fit.model)}")
                else:
                    raise ValueError("Could not successfully fit a viable SARIMA model")

        elif theta_value["action"] == "pqPQ":
            best_model_theta = theta_value["theta"]
            order, seasonal_order, trend = theta_value["theta"]
            if seasonal_order[3] == 1:
                seasonal_order = [0, 0, 0, 0]
            best_model_fit, fit_time, ic = autosarima_utils._fit_sarima_model(
                y=y,
                X=X,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                method=method,
                maxiter=maxiter,
                information_criterion=information_criterion,
                **train_config,
            )

        else:
            return theta_value, None, None

        model = deepcopy(self.model)
        model.reset()
        self.set_theta(model, best_model_theta, train_data)

        model.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)
        model.model = best_model_fit
        name = model.target_name
        train_data = train_data.univariates[name].to_pd()
        times = train_data.index
        yhat = model.model.fittedvalues
        err = [np.sqrt(model.model.params[-1])] * len(train_data)
        train_result = (
            UnivariateTimeSeries(times, yhat, name).to_ts(),
            UnivariateTimeSeries(times, err, f"{name}_err").to_ts(),
        )

        return best_model_theta, model, train_result

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        order, seasonal_order, trend = theta
        model.config.order = order
        model.config.seasonal_order = seasonal_order
