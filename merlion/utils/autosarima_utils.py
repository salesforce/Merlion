#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import functools
import logging
import time
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from scipy.signal import argrelmax
from scipy.stats import norm
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _model_name(model_spec):
    """
    Return model name
    """
    p, d, q = model_spec.order
    P, D, Q, m = model_spec.seasonal_order
    return " SARIMA({p},{d},{q})({P},{D},{Q})[{m}] {constant_trend}".format(
        p=p,
        d=d,
        q=q,
        P=P,
        D=D,
        Q=Q,
        m=m,
        constant_trend="   with constant" if model_spec.trend is not None else "without constant",
    )


def diff(x, lag=1, differences=1):
    """
    Return suitably lagged and iterated differences from the given 1D or 2D array x
    """

    n = x.shape[0]

    if any(v < 1 for v in (lag, differences)):
        raise ValueError("lag and differences must be positive (> 0) integers")

    if lag >= n:
        raise ValueError("lag should be smaller than the length of array")

    if differences >= n:
        raise ValueError("differences should be smaller than the length of array")

    res = x
    for i in range(differences):
        if res.ndim == 1:  # compute the lag for vector
            res = res[lag : res.shape[0]] - res[: res.shape[0] - lag]
        else:
            res = res[lag : res.shape[0], :] - res[: res.shape[0] - lag, :]
    return res


def _root_test(model_fit, ic):
    """
    Check the roots of the sarima model, and set IC to inf if the roots are
    near non-invertible.
    """
    # This is identical to the implementation of pmdarima and forecast

    max_invroot = 0
    p, d, q = model_fit.model.order
    P, D, Q, m = model_fit.model.seasonal_order
    if p + P > 0:
        max_invroot = max(0, *np.abs(1 / model_fit.arroots))
    if q + Q > 0 and np.isfinite(ic):
        max_invroot = max(0, *np.abs(1 / model_fit.maroots))

    if max_invroot > 1 - 1e-2:
        ic = np.inf
        logger.debug(
            "Near non-invertible roots for order "
            "(%i, %i, %i)(%i, %i, %i, %i); setting score to inf (at "
            "least one inverse root too close to the border of the "
            "unit circle: %.3f)" % (p, d, q, P, D, Q, m, max_invroot)
        )
    return ic


def _fit_sarima_model(y, X, order, seasonal_order, trend, method, maxiter, information_criterion, **kwargs):
    """
    Train a sarima model with the given time-series and hyperparamteres tuple.
    Return the trained model, training time and information criterion
    """
    start = time.time()
    ic = np.inf
    model_fit = None
    model_spec = sm.tsa.SARIMAX(
        endog=y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend, validate_specification=False, **kwargs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model_fit = model_spec.fit(method=method, maxiter=maxiter, disp=0)
        except (LinAlgError, ValueError) as v:
            logger.warning(f"Caught exception {type(v).__name__}: {str(v)}")
        else:
            ic = model_fit.info_criteria(information_criterion)
            ic = _root_test(model_fit, ic)
        fit_time = time.time() - start
        logger.debug(
            "{model}   : {ic_name}={ic:.3f}, Time={time:.2f} sec".format(
                model=_model_name(model_spec), ic_name=information_criterion.upper(), ic=ic, time=fit_time
            )
        )
    return model_fit, fit_time, ic


def _refit_sarima_model(model_fitted, approx_ic, method, inititer, maxiter, information_criterion):
    """
    Re-train the the approximated sarima model which is used in approximation mode.
    Take the approximated model as initialization, fine tune with (maxiter - initier)
    rounds or multiple rounds until no improvement about information criterion
    Return the trained model
    """
    start = time.time()
    fit_time = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_fit = model_fitted
        ic = approx_ic
        logger.debug(
            "Initial Model: {model} Iter={iter:d}, {ic_name}={ic:.3f}".format(
                model=_model_name(model_fitted.model), iter=inititer, ic_name=information_criterion.upper(), ic=ic
            )
        )
        for cur_iter in range(inititer + 1, maxiter + 1):
            try:
                model_fitted = model_fitted.model.fit(
                    method=method, maxiter=1, disp=0, start_params=model_fitted.params
                )
            except (LinAlgError, ValueError) as v:
                logger.warning(f"Caught exception {type(v).__name__}: {str(v)}")
            else:
                cur_ic = model_fitted.info_criteria(information_criterion)
                cur_ic = _root_test(model_fitted, cur_ic)
                if cur_ic > ic or np.isinf(cur_ic):
                    break
                else:
                    ic = cur_ic
                    best_fit = model_fitted
            fit_time = time.time() - start
            logger.debug(
                "{model}   : Iter={iter:d}, {ic_name}={ic:.3f}, Time={time:.2f} sec".format(
                    model=_model_name(model_fitted.model),
                    iter=cur_iter,
                    ic_name=information_criterion.upper(),
                    ic=ic,
                    time=fit_time,
                )
            )
    return best_fit


def detect_maxiter_sarima_model(y, X, d, D, m, method, information_criterion, **kwargs):
    """
    run a zero model with SARIMA(2; d; 2)(1; D; 1) / ARIMA(2; d; 2) determine the optimal maxiter
    """
    logger.debug("Automatically detect the maxiter")
    order = (2, d, 2)
    if m == 1:
        seasonal_order = (0, 0, 0, 0)
    else:
        seasonal_order = (1, D, 1, m)

    # default setting of maxiter is 10
    start = time.time()
    fit_time = np.nan
    maxiter = 10
    ic = np.inf
    model_spec = sm.tsa.SARIMAX(
        endog=y, exog=X, order=order, seasonal_order=seasonal_order, trend="c", validate_specification=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model_fit = model_spec.fit(method=method, maxiter=maxiter, disp=0)
        except (LinAlgError, ValueError) as v:
            logger.warning(f"Caught exception {type(v).__name__}: {str(v)}")
            return maxiter
        else:
            ic = model_fit.info_criteria(information_criterion)
            ic = _root_test(model_fit, ic)

        for cur_iter in range(maxiter + 1, 51):
            try:
                model_fit = model_fit.model.fit(method=method, maxiter=1, disp=0, start_params=model_fit.params)
            except (LinAlgError, ValueError) as v:
                logger.warning(f"Caught exception {type(v).__name__}: {str(v)}")
            else:
                cur_ic = model_fit.info_criteria(information_criterion)
                cur_ic = _root_test(model_fit, cur_ic)
                if cur_ic > ic or np.isinf(cur_ic):
                    break
                else:
                    ic = cur_ic
                    maxiter = cur_iter
                    logger.debug(
                        "Zero model: {model} Iter={iter:d}, {ic_name}={ic:.3f}".format(
                            model=_model_name(model_fit.model),
                            iter=maxiter,
                            ic_name=information_criterion.upper(),
                            ic=ic,
                        )
                    )
        fit_time = time.time() - start
        logger.debug(
            "Zero model: {model} Iter={iter:d}, {ic_name}={ic:.3f}, Time={time:.2f} sec".format(
                model=_model_name(model_fit.model),
                iter=maxiter,
                ic_name=information_criterion.upper(),
                ic=ic,
                time=fit_time,
            )
        )
        logger.info(f"Automatically detect the maxiter is {maxiter}")
    return maxiter


def multiperiodicity_detection(x, pval=0.05, max_lag=None):
    """
    Detect multiple periodicity of a time series
    The idea can be found in theta method
    (https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R).
    Returns a list of periods, which indicates the seasonal periods of the
    time series
    """
    tcrit = norm.ppf(1 - pval / 2)
    if max_lag is None:
        max_lag = max(min(int(10 * np.log10(x.shape[0])), x.shape[0] - 1), 40)
    xacf = sm.tsa.acf(x, nlags=max_lag, fft=False)
    xacf[np.isnan(xacf)] = 0

    # select the local maximum points with acf > 0
    candidates = np.intersect1d(np.where(xacf > 0), argrelmax(xacf)[0])

    # the periods should be smaller than one third of the lenght of time series
    candidates = candidates[candidates < int(x.shape[0] / 3)]
    if candidates.shape[0] == 0:
        return []
    else:
        candidates = candidates[np.insert(argrelmax(xacf[candidates])[0], 0, 0)]

    xacf = xacf[1:]
    clim = tcrit / np.sqrt(x.shape[0]) * np.sqrt(np.cumsum(np.insert(np.square(xacf) * 2, 0, 1)))

    # statistical test if acf is significant w.r.t a normal distribution
    candidate_filter = candidates[xacf[candidates - 1] > clim[candidates - 1]]
    # return candidate seasonalities, sorted by ACF value
    candidate_filter = sorted(candidate_filter.tolist(), key=lambda c: xacf[c - 1], reverse=True)
    return candidate_filter


def seas_seasonalstationaritytest(x, m):
    """
    Estimate the strength of seasonal component. The idea can be found in
    https://otexts.com/fpp2/seasonal-strength.html
    R implementation uses mstl instead of stl to deal with multiple seasonality
    """
    stlfit = sm.tsa.STL(x, m).fit()
    vare = np.nanvar(stlfit.resid)
    season = max(0, min(1, 1 - vare / np.nanvar(stlfit.resid + stlfit.seasonal)))
    return season > 0.64


def nsdiffs(x, m, max_D=1, test="seas"):
    """
    Estimate the seasonal differencing order D with statistical test

    Parameters:
    x : the time series to difference
    m : the number of seasonal periods
    max_D : the maximal number of seasonal differencing order allowed
    test: the type of test of seasonality to use to detect seasonal periodicity
    """
    D = 0
    if max_D <= 0:
        raise ValueError("max_D must be a positive integer")
    if np.max(x) == np.min(x) or m < 2:
        return D
    if test == "seas":
        dodiff = seas_seasonalstationaritytest(x, m)
        while dodiff and D < max_D:
            D += 1
            x = diff(x, lag=m)
            if np.max(x) == np.min(x):
                return D
            if len(x) >= 2 * m and D < max_D:
                dodiff = seas_seasonalstationaritytest(x, m)
            else:
                dodiff = False
    return D


def KPSS_stationaritytest(xx, alpha=0.05):
    """
    The KPSS test is used with the null hypothesis that
    x has a stationary root against a unit-root alternative

    The KPSS test is used with the null hypothesis that
    x has a stationary root against a unit-root alternative.
    Then the test returns the least number of differences required to
    pass the test at the level alpha
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = sm.tsa.stattools.kpss(xx, regression="c", nlags=round(3 * np.sqrt(len(xx)) / 13))
    yout = results[1]
    return yout, yout < alpha


def ndiffs(x, alpha=0.05, max_d=2, test="kpss"):
    """
    Estimate the differencing order d with statistical test

    Parameters:
    x : the time series to difference
    alpha : level of the test, possible values range from 0.01 to 0.1
    max_d : the maximal number of differencing order allowed
    test: the type of test of seasonality to use to detect seasonal periodicity
    """
    d = 0
    if max_d <= 0:
        raise ValueError("max_d must be a positive integer")
    if alpha < 0.01:
        warnings.warn("Specified alpha value is less than the minimum, setting alpha=0.01")
        alpha = 0.01
    if alpha > 0.1:
        warnings.warn("Specified alpha value is larger than the maximum, setting alpha=0.1")
        alpha = 0.1
    if np.max(x) == np.min(x):
        return d

    if test == "kpss":
        pval, dodiff = KPSS_stationaritytest(x, alpha)
        if np.isnan(pval):
            return 0
        while dodiff and d < max_d:
            d += 1
            x = diff(x)
            if np.max(x) == np.min(x):
                return d
            pval, dodiff = KPSS_stationaritytest(x, alpha)
            if np.isnan(pval):
                return d - 1
    return d


class _StepwiseFitWrapper:
    """
    Stepwise search the hyper-parameters
    """

    # This is identical to the implementation of auto.arma in forecast package in R.
    def __init__(
        self,
        y,
        X,
        p,
        d,
        q,
        P,
        D,
        Q,
        m,
        max_p,
        max_q,
        max_P,
        max_Q,
        trend,
        method,
        maxiter,
        information_criterion,
        relative_improve,
        max_k,
        max_dur,
        **kwargs,
    ):
        self._fit_arima = functools.partial(
            _fit_sarima_model,
            y=y,
            X=X,
            method=method,
            maxiter=maxiter,
            information_criterion=information_criterion,
            **kwargs,
        )
        self.information_criterion = information_criterion
        self.relative_improve = relative_improve
        self.trend = trend
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.k = self.start_k = 0
        self.max_k = max_k
        self.max_dur = max_dur

        # results stored in dict
        # dict[tuple -> ARIMA]
        self.results_dict = dict()

        # dict[tuple -> float]
        self.ic_dict = dict()

        # dict[tuple -> float]
        self.fit_time_dict = dict()
        self.bestfit = None
        self.bestfit_key = None

    def _do_fit(self, order, seasonal_order, trend=None):
        """Do a fit and determine whether the model is better"""
        if seasonal_order[-1] == 1:
            seasonal_order = (0, 0, 0, 0)

        # constant 0, trend None/ constant 1, trend 'c'
        constant = 0 if trend is None else 1
        if (order, seasonal_order, constant) not in self.results_dict:
            self.k += 1
            fit, fit_time, new_ic = self._fit_arima(order=order, seasonal_order=seasonal_order, trend=trend)
            self.results_dict[(order, seasonal_order, constant)] = fit
            self.ic_dict[(order, seasonal_order, constant)] = new_ic
            self.fit_time_dict[(order, seasonal_order, constant)] = fit_time
            if fit is None or np.isinf(new_ic):
                return False
            if self.bestfit is None:
                self.bestfit = fit
                self.bestfit_key = (order, seasonal_order, constant)
                logger.debug("First best model found (%.3f)" % new_ic)
                return True
            current_ic = self.ic_dict[self.bestfit_key]
            if new_ic < current_ic:
                logger.debug("New best model found (%.3f < %.3f)" % (new_ic, current_ic))
                self.bestfit = fit
                self.bestfit_key = (order, seasonal_order, constant)
                if new_ic < current_ic * (1 - self.relative_improve):
                    return True
                else:
                    return False
        return False

    def stepwisesearch(self):
        """
        return a list of sarima model ordered by information criterion
        """
        start_time = time.time()
        p, d, q = self.p, self.d, self.q
        P, D, Q, m = self.P, self.D, self.Q, self.m
        max_p, max_q = self.max_p, self.max_q
        max_P, max_Q = self.max_P, self.max_Q
        logger.debug("Performing stepwise search to minimize %s" % self.information_criterion)

        # We try four possible models to start with:
        # 1. SARIMA(2; d; 2)(1; D; 1) / ARIMA(2; d; 2)
        self._do_fit((p, d, q), (P, D, Q, m), self.trend)

        # 2. SARIMA(0; d; 0)(0; D; 0) / ARIMA(0; d; 0)
        if self._do_fit((0, d, 0), (0, D, 0, m), self.trend):
            p = q = P = Q = 0

        # 3. SARIMA(1; d; 0)(1; D; 0) / ARIMA(1; d; 0)
        if max_p > 0 or max_P > 0:
            _p = 1 if max_p > 0 else 0
            _P = 1 if (m > 1 and max_P > 0) else 0
            if self._do_fit((_p, d, 0), (_P, D, 0, m), self.trend):
                p = _p
                P = _P
                q = Q = 0

        # 4. SARIMA(0; d; 1)(0; D; 1) / ARIMA(0; d; 1)
        if max_q > 0 or max_Q > 0:
            _q = 1 if max_q > 0 else 0
            _Q = 1 if (m > 1 and max_Q > 0) else 0
            if self._do_fit((0, d, _q), (0, D, _Q, m), self.trend):
                p = P = 0
                Q = _Q
                q = _q

        # 5. NO trend model (if we haven't tried it yet)
        if self.trend is not None:
            if self._do_fit((0, d, 0), (0, D, 0, m), None):
                p = q = P = Q = 0
        while self.start_k < self.k < self.max_k:
            # break loop if no new model included
            self.start_k = self.k

            # break loop if execution time exceeds the timeout threshold.
            dur = time.time() - start_time
            if self.max_dur and dur > self.max_dur:
                warnings.warn(
                    "early termination of stepwise search due to "
                    "max_dur threshold (%.3f > %.3f)" % (dur, self.max_dur)
                )
                break

            # where one of P and Q is allowed to vary by +-1 from the current mode
            if P > 0 and self._do_fit((p, d, q), (P - 1, D, Q, m), self.trend):
                P -= 1
                continue

            if Q > 0 and self._do_fit((p, d, q), (P, D, Q - 1, m), self.trend):
                Q -= 1
                continue

            if P < max_P and self._do_fit((p, d, q), (P + 1, D, Q, m), self.trend):
                P += 1
                continue

            if Q < max_Q and self._do_fit((p, d, q), (P, D, Q + 1, m), self.trend):
                Q += 1
                continue

            # where P and Q both vary by +-1 from the current model
            if Q > 0 and P > 0 and self._do_fit((p, d, q), (P - 1, D, Q - 1, m), self.trend):
                Q -= 1
                P -= 1
                continue

            if Q < max_Q and P > 0 and self._do_fit((p, d, q), (P - 1, D, Q + 1, m), self.trend):
                Q += 1
                P -= 1
                continue

            if Q > 0 and P < max_P and self._do_fit((p, d, q), (P + 1, D, Q - 1, m), self.trend):
                Q -= 1
                P += 1
                continue

            if Q < max_Q and P < max_P and self._do_fit((p, d, q), (P + 1, D, Q + 1, m), self.trend):
                Q += 1
                P += 1
                continue

            # where one of p and q is allowed to vary by +-1 from the current mode
            if p > 0 and self._do_fit((p - 1, d, q), (P, D, Q, m), self.trend):
                p -= 1
                continue
            if q > 0 and self._do_fit((p, d, q - 1), (P, D, Q, m), self.trend):
                q -= 1
                continue
            if p < max_p and self._do_fit((p + 1, d, q), (P, D, Q, m), self.trend):
                p += 1
                continue
            if q < max_q and self._do_fit((p, d, q + 1), (P, D, Q, m), self.trend):
                q += 1
                continue

            # where P and Q both vary by +-1 from the current model
            if q > 0 and p > 0 and self._do_fit((p - 1, d, q - 1), (P, D, Q, m), self.trend):
                q -= 1
                p -= 1
                continue
            if q < max_q and p > 0 and self._do_fit((p - 1, d, q + 1), (P, D, Q, m), self.trend):
                q += 1
                p -= 1
                continue
            if q > 0 and p < max_p and self._do_fit((p + 1, d, q - 1), (P, D, Q, m), self.trend):
                q -= 1
                p += 1
                continue
            if q < max_q and p < max_p and self._do_fit((p + 1, d, q + 1), (P, D, Q, m), self.trend):
                q += 1
                p += 1
                continue

            # where the constant trend is included if the current model has trend = None
            # or excluded if the current model has trend = 'c'
            if self._do_fit((p, d, q), (P, D, Q, m), trend=None if self.trend is not None else "c"):
                self.trend = None if self.trend is not None else "c"
                continue

        # check if the search has been ended after max_steps
        if self.k >= self.max_k:
            warnings.warn("stepwise search has reached the maximum number of tries to find the best fit model")
        filtered_models_ics = sorted(
            [
                (v, k, self.ic_dict[k])
                for k, v in self.results_dict.items()
                if v is not None and np.isfinite(self.ic_dict[k])
            ],
            key=(lambda fit_ic: fit_ic[2]),
        )
        return filtered_models_ics
