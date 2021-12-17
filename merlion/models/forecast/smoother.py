#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Multi-Scale Exponential Smoother for univariate time series forecasting.
"""
from copy import deepcopy
import logging
from math import floor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries, assert_equal_timedeltas
from merlion.utils.istat import ExponentialMovingAverage, RecencyWeightedVariance
from merlion.utils.resample import to_pd_datetime, to_timestamp
from merlion.transform.moving_average import LagTransform
from merlion.transform.resample import TemporalResample
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig

logger = logging.getLogger(__name__)


class MSESConfig(ForecasterConfig):
    """
    Configuration class for an MSES forecasting model.
    """

    _default_transform = TemporalResample(trainable_granularity=True)

    def __init__(
        self,
        max_forecast_steps: int,
        max_backstep: int = None,
        recency_weight: float = 0.5,
        accel_weight: float = 1.0,
        optimize_acc: bool = True,
        eta: float = 0.0,
        rho: float = 0.0,
        phi: float = 2.0,
        inflation: float = 1.0,
        **kwargs,
    ):
        r"""
        Letting ``w`` be the recency weight, ``B`` the maximum backstep, ``x_t`` the last seen data point,
        and ``l_s,t`` the series of losses for scale ``s``. 

        .. math:: 
            \begin{align*}
            \hat{x}_{t+h} & = \sum_{b=0}^B p_{b} \cdot (x_{t-b} + v_{b+h,t} + a_{b+h,t}) \\
            \space \\
            \text{where} \space\space & v_{b+h,t} = \text{EMA}_w(\Delta_{b+h} x_t)   \\
            & a_{b+h,t} = \text{EMA}_w(\Delta_{b+h}^2 x_t) \\
            \text{and} \space\space & p_b = \sigma(z)_b \space\space \\
            \text{if} & \space\space  z_b = (b+h)^\phi \cdot \text{EMA}_w(l_{b+h,t}) \cdot \text{RWSE}_w(l_{b+h,t})\\
            \end{align*}

        :param max_backstep: Max backstep to use in forecasting. If we train with x(0),...,x(t),
            Then, the b-th model MSES uses will forecast x(t+h) by anchoring at x(t-b) and 
            predicting xhat(t+h) = x(t-b) + delta_hat(b+h).
        :param recency_weight: The recency weight parameter to use when estimating delta_hat.
        :param accel_weight: The weight to scale the acceleration by when computing delta_hat.
            Specifically, delta_hat(b+h) = velocity(b+h) + accel_weight * acceleration(b+h).
        :param optimize_acc: If True, the acceleration correction will only be used at scales
            ranging from 1,...(max_backstep+max_forecast_steps)/2.
        :param eta: The parameter used to control the rate at which recency_weight gets
            tuned when online updates are made to the model and losses can be computed.
        :param rho: The parameter that determines what fraction of the overall error is due to
            velcity error, while the rest is due to the complement. The error at any scale 
            will be determined as ``rho * velocity_error + (1-rho) * loss_error``.
        :param phi: The parameter used to exponentially inflate the magnitude of loss error at
            different scales. Loss error for scale ``s`` will be increased by a factor of ``phi ** s``.  
        :param inflation: The inflation exponent to use when computing the distribution
            p(b|h) over the models when forecasting at horizon h according to standard
            errors of the estimated velocities over the models; inflation=1 is equivalent
            to using the softmax function.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, **kwargs)
        assert 0.0 <= rho <= 1.0
        assert 1.0 <= phi
        self.max_backstep = max_forecast_steps if max_backstep is None else max_backstep
        self.recency_weight = recency_weight
        self.accel_weight = accel_weight
        self.optimize_acc = optimize_acc
        self.eta = eta
        self.rho = rho
        self.phi = phi
        self.inflation = inflation

    @property
    def max_scale(self):
        return self.max_backstep + self.max_forecast_steps

    @property
    def backsteps(self):
        return list(range(self.max_backstep + 1))


class MSESTrainConfig(object):
    """
    MSES training configuration.
    """

    def __init__(
        self,
        incremental: bool = True,
        process_losses: bool = True,
        tune_recency_weights: bool = False,
        init_batch_sz: int = 2,
        train_cadence: int = None,
    ):
        """
        :param incremental: If True, train the MSES model incrementally with the initial
            training data at the given ``train_cadence``. This allows MSES to return a
            forecast for the training data.
        :param: If True, track the losses encountered during incremental initial training.
        :tune_recency_weights: If True, tune recency weights during incremental initial
            training.
        :param init_batch_sz: The size of the inital training batch for MSES. This is
            necessary because MSES cannot predict the past, but needs to start with some
            data. This should be very small. 2 is the minimum, and is recommended because
            2 will result in the most representative train forecast.
        :param train_cadence: The frequency at which the training forecasts will be generated
            during incremental training.
        """
        assert init_batch_sz >= 2
        self.incremental = incremental
        self.process_losses = process_losses
        self.tune_recency_weights = tune_recency_weights
        self.init_batch_sz = init_batch_sz
        self.train_cadence = train_cadence


class MSES(ForecasterBase):
    r"""
    Multi-scale Exponential Smoother (MSES) is a forecasting algorithm modeled heavily 
    after classical mechanical concepts, namely, velocity and acceleration.

    Having seen data points of a time series up to time t, MSES forecasts x(t+h) by
    anchoring at a value b steps back from the last known value, x(t-b), and estimating the
    delta between x(t-b) and x(t+h). The delta over these b+h timesteps, delta(b+h), also known 
    as the delta at scale b+h, is predicted by estimating the velocity over these timesteps 
    as well as the change in the velocity, acceleration. Specifically, 
    
        xhat(t+h) = x(t-b) + velocity_hat(b+h) + acceleration_hat(b+h)

    This estimation is done for each b, known as a backstep, from 0, which anchors at x(t), 
    1,... up to a maximum backstep configurable by the user. The algorithm then takes the 
    seperate forecasts of x(t+h), indexed by which backstep was used, xhat_b(t+h), and determines
    a final forecast: p(b|h) dot xhat_b, where p(b|h) is a distribution over the xhat_b's that is 
    determined according to the lowest standard errors of the recency-weighted velocity estimates.

    Letting ``w`` be the recency weight, ``B`` the maximum backstep, ``x_t`` the last seen data point,
    and ``l_s,t`` the series of losses for scale ``s``. 

        .. math:: 
            \begin{align*}
            \hat{x}_{t+h} & = \sum_{b=0}^B p_{b} \cdot (x_{t-b} + v_{b+h,t} + a_{b+h,t}) \\
            \space \\
            \text{where} \space\space & v_{b+h,t} = \text{EMA}_w(\Delta_{b+h} x_t)   \\
            & a_{b+h,t} = \text{EMA}_w(\Delta_{b+h}^2 x_t) \\
            \text{and} \space\space & p_b = \sigma(z)_b \space\space \\
            \text{if} & \space\space  z_b = (b+h)^\phi \cdot \text{EMA}_w(l_{b+h,t}) \cdot \text{RWSE}_w(l_{b+h,t})\\
            \end{align*}
    """
    config_class = MSESConfig
    _default_train_config = MSESTrainConfig()

    def __init__(self, config: MSESConfig):
        super().__init__(config)
        self.delta_estimator = DeltaEstimator(
            max_scale=self.config.max_scale,
            recency_weight=self.config.recency_weight,
            accel_weight=self.config.accel_weight,
            optimize_acc=self.config.optimize_acc,
            eta=self.config.eta,
            phi=self.config.phi,
        )

    @property
    def rho(self):
        return self.config.rho

    @property
    def backsteps(self):
        return self.config.backsteps

    @property
    def max_horizon(self):
        return self.max_forecast_steps * self.timedelta

    def train(self, train_data: TimeSeries, train_config: MSESTrainConfig = None) -> Tuple[Optional[TimeSeries], None]:
        if train_config is None:
            train_config = deepcopy(self._default_train_config)
            if isinstance(train_config, dict):
                train_config = MSESTrainConfig(**train_config)

        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)
        name = self.target_name
        train_data = train_data.univariates[name]

        if not train_config.incremental:
            self.delta_estimator.train(train_data)
            return None, None

        # train on initial batch
        b = train_config.init_batch_sz
        init_train_data, train_data = train_data[:b], train_data[b:]
        self.delta_estimator.train(init_train_data)
        self.last_train_time = init_train_data.tf

        # use inital batch as train forecast
        init_train_forecast = init_train_data.to_ts()
        init_train_err = UnivariateTimeSeries(
            time_stamps=init_train_data.time_stamps,
            name=f"{init_train_data.name}_err",
            values=[0] * len(init_train_data),
        ).to_ts()

        # train incrementally
        h = train_config.train_cadence
        h = h * self.timedelta if h is not None else None
        h = min(h, self.max_horizon) if h is not None else self.max_horizon
        train_forecast, train_err = self._incremental_train(
            train_data=train_data,
            train_cadence=h,
            process_losses=train_config.process_losses,
            tune_recency_weights=train_config.tune_recency_weights,
        )

        train_forecast = init_train_forecast + train_forecast
        train_err = init_train_err + train_err

        return train_forecast, train_err

    def _incremental_train(self, train_data, train_cadence, process_losses, tune_recency_weights):
        # train incrementally
        t, tf = train_data.t0, train_data.tf
        train_forecast, train_err = [], []
        if train_cadence is None:
            train_cadence = self.max_horizon
        all_t = train_data.time_stamps
        while t <= tf:
            i = np.searchsorted(all_t, t)
            if i + 1 < len(all_t):
                t_next = max(to_timestamp(to_pd_datetime(t) + train_cadence), all_t[i + 1])
            else:
                t_next = all_t[-1] + 0.001
            train_batch = train_data.window(t, t_next, include_tf=False)
            if len(train_batch) > 0:
                # forecast & process losses
                if process_losses:
                    scale_losses, (forecast, err) = self._compute_losses(train_batch, return_forecast=True)
                    self.delta_estimator.process_losses(scale_losses, tune_recency_weights)
                else:
                    forecast, err = self.forecast(train_batch.time_stamps)
                # store forecast results
                train_forecast.append(forecast)
                train_err.append(err)
                # train on batch
                self.delta_estimator.train(train_batch)
                self.last_train_time = train_batch.tf
            # increment time
            t = t_next
        train_forecast, train_err = [sum(v[1:], v[0]) for v in (train_forecast, train_err)]
        return train_forecast, train_err

    def update(
        self, new_data: TimeSeries, tune_recency_weights: bool = True, train_cadence=None
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Updates the MSES model with new data that has been acquired since the model's
        initial training.

        :param new_data: New data that has occured since the last training time.
        :param tune_recency_weights: If True, the model will first forecast the values at the
            new_data's timestamps, calculate the associated losses, and use these losses
            to make updates to the recency weight.
        :param train_cadence: The frequency at which the training forecasts will be generated
            during incremental training.
        """
        name = self.target_name
        if new_data.is_empty():
            return (
                UnivariateTimeSeries.empty(name=name).to_ts(),
                UnivariateTimeSeries.empty(name=f"{name}_err").to_ts(),
            )
        new_data = self.transform(new_data).univariates[name]

        assert_equal_timedeltas(new_data, self.timedelta)
        next_train_time = self.last_train_time + self.timedelta
        if to_pd_datetime(new_data.t0) > next_train_time:
            logger.warning(
                f"Updating the model with new data requires the "
                f"new data to start at or before time "
                f"{to_pd_datetime(next_train_time)}, which is the time "
                f"directly after the last train time. Got data starting "
                f"at {to_pd_datetime(new_data.t0)} instead."
            )

        _, new_data = new_data.bisect(next_train_time, t_in_left=False)
        if new_data.is_empty():
            return (
                UnivariateTimeSeries.empty(name=name).to_ts(),
                UnivariateTimeSeries.empty(name=f"{name}_err").to_ts(),
            )

        return self._incremental_train(
            train_data=new_data,
            train_cadence=train_cadence,
            process_losses=True,
            tune_recency_weights=tune_recency_weights,
        )

    def _compute_losses(
        self, data: UnivariateTimeSeries, return_forecast: bool = False, return_iqr: bool = False
    ) -> Union[Dict[int, List[float]], Tuple[Dict[int, List[float]], TimeSeries]]:
        """
        Computes forecast losses at every point possible in data for every backstep, and
        then associates the losses with the relevant scale.

        :param data: Data to forecast and compute losses for. The first timestamp in
            data must be the timestamp directly after the last train time.
        :return: A hash map mapping each scale to a list of losses.
        """
        # forecast at every scale possible for every point possible in data
        forecastable_data = data[: self.max_forecast_steps]
        xhat_hb = [self.xhat_h(h) for h in range(1, len(forecastable_data) + 1)]
        xtrue_h = forecastable_data.values
        losses_hb = np.array(xhat_hb, dtype=float) - np.expand_dims(xtrue_h, 1)
        losses = dict()
        for i, j in np.ndindex(losses_hb.shape):
            # scale=i+j+1 because b=i and h=j+1
            scale = i + j + 1
            if not np.isnan(losses_hb[i, j]):
                if scale in losses:
                    losses[scale] += [losses_hb[i, j]]
                else:
                    losses[scale] = [losses_hb[i, j]]

        if not return_forecast:
            return losses

        # generate forecast
        name = self.target_name
        forecast = [self.marginalize_xhat_h(i + 1, xhat_h) for i, xhat_h in enumerate(xhat_hb)]
        xhat, neg_err, pos_err = [[f[i] for f in forecast] for i in (0, 1, 2)]

        if not return_iqr:
            err = UnivariateTimeSeries(
                time_stamps=forecastable_data.time_stamps,
                name=f"{name}_err",
                values=(np.abs(pos_err) + np.abs(neg_err)) / 2,
            ).to_ts()
            xhat = UnivariateTimeSeries(time_stamps=forecastable_data.time_stamps, values=xhat, name=name).to_ts()
            return losses, (xhat, err)

        # return forecast with iqr
        t = forecastable_data.time_stamps
        lb = UnivariateTimeSeries(
            name=f"{name}_lower", time_stamps=t, values=xhat + norm.ppf(0.75) * np.asarray(neg_err)
        ).to_ts()
        ub = UnivariateTimeSeries(
            name=f"{name}_upper", time_stamps=t, values=xhat + norm.ppf(0.75) * np.asarray(pos_err)
        ).to_ts()
        xhat = UnivariateTimeSeries(t, xhat, name).to_ts()
        return losses, (xhat, lb, ub)

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Tuple[TimeSeries, None]:

        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)

        if time_series_prev is not None and not time_series_prev.is_empty():
            self.update(time_series_prev)
            prev = self.transform(time_series_prev)
            prev = time_series_prev.univariates[prev.names[self.target_seq_index]]
            prev_t = prev.time_stamps
            prev_x = prev.values

        # forecast
        forecast = [self.marginalize_xhat_h(h, self.xhat_h(h)) for h in range(1, len(time_stamps) + 1)]
        xhat, neg_err, pos_err = [[f[i] for f in forecast] for i in (0, 1, 2)]

        if return_prev and time_series_prev is not None:
            assert not return_iqr, "MSES does not yet support uncertainty for previous time series"
            xhat = prev_x + xhat
            time_stamps = prev_t + time_stamps
            orig_t = None if orig_t is None else prev_t + orig_t

        name = self.target_name
        if return_iqr:
            lb = (
                UnivariateTimeSeries(
                    name=f"{name}_lower", time_stamps=time_stamps, values=xhat + norm.ppf(0.75) * np.asarray(neg_err)
                )
                .to_ts()
                .align(reference=orig_t)
            )
            ub = (
                UnivariateTimeSeries(
                    name=f"{name}_upper", time_stamps=time_stamps, values=xhat + norm.ppf(0.75) * np.asarray(pos_err)
                )
                .to_ts()
                .align(reference=orig_t)
            )
            xhat = UnivariateTimeSeries(time_stamps, xhat, name).to_ts().align(reference=orig_t)
            return xhat, lb, ub

        xhat = UnivariateTimeSeries(time_stamps, xhat, name).to_ts().align(reference=orig_t)
        err = (
            UnivariateTimeSeries(
                time_stamps=time_stamps, name=f"{name}_err", values=(np.abs(pos_err) + np.abs(neg_err)) / 2
            )
            .to_ts()
            .align(reference=orig_t)
        )
        return xhat, err

    def _forecast(self, horizon: int, backstep: int) -> Optional[float]:
        """
        Returns the forecast at input horizon using input backstep.
        """
        x = self.delta_estimator.x
        scale = backstep + horizon
        delta_hat = self.delta_estimator.delta_hat(scale)
        if len(x) < backstep + 1 or delta_hat is None:
            return None
        return x[-(backstep + 1)] + delta_hat

    def xhat_h(self, horizon: int) -> List[Optional[float]]:
        """
        Returns the forecasts for the input horizon at every backstep.
        """
        return [self._forecast(horizon, backstep) for backstep in self.backsteps]

    def marginalize_xhat_h(self, horizon: int, xhat_h: List[Optional[float]]):
        """
        Given a list of forecasted values produced by delta estimators at
        different backsteps, compute a weighted average of these values. The
        weights are assigned based on the standard errors of the velocities,
        where the b'th estimate will be given more weight if its velocity has a
        lower standard error relative to the other estimates.

        :param horizon: the horizon at which we want to predict
        :param xhat_h: the forecasted values at this horizon, using each of
            the possible backsteps
        """

        assert len(xhat_h) == len(self.backsteps)
        if all(x is None for x in xhat_h):
            t = self.last_train_time = self.last_train_time + horizon * self.timedelta
            raise RuntimeError(
                f"Not enough training data to forecast at horizon {horizon} "
                f"(estimated time {pd.to_datetime(t, unit='s')}, last train "
                f"time is {pd.to_datetime(self.last_train_time, unit='s')})"
            )

        # Get the non None xhat's & their corresponding std errs
        xhat_h, neg_err_h, pos_err_h, vel_errs, loss_errs = np.asarray(
            [
                (
                    x,
                    self.delta_estimator.neg_err(b + horizon),
                    self.delta_estimator.pos_err(b + horizon),
                    self.delta_estimator.vel_err(b + horizon),
                    self.delta_estimator.loss_err(b + horizon),
                )
                for x, b in zip(xhat_h, self.backsteps)
                if x is not None
            ],
            dtype=float,
        ).T

        if self.rho == 1.0:
            q = vel_errs
        elif self.rho == 0.0:
            q = loss_errs
        else:
            if (vel_errs == np.inf).all():
                vel_errs.fill(0)
            if (loss_errs == np.inf).all():
                loss_errs.fill(0)
            q = self.rho * vel_errs + (1 - self.rho) * loss_errs

        if (q == np.inf).all():
            q = np.ones(len(q))

        # Do a softmax to get probabilities
        q = np.exp(-(q - q.min()) * self.config.inflation)
        q = q / q.sum()

        # compute estimate with lower and upper bounds
        xhat, neg_err, pos_err = [np.sum(q * v).item() for v in (xhat_h, neg_err_h, pos_err_h)]
        return xhat, neg_err, pos_err


class DeltaStats:
    """
    A wrapper around the statistics used to estimate deltas at a given scale.
    """

    def __init__(self, scale: int, recency_weight: float):
        """
        :param scale: The scale associated with the statistics
        :param recency_weight: The recency weight parameter that that the incremental
            velocity, acceleration and standard error statistics should use.
        """
        self.velocity = ExponentialMovingAverage(recency_weight, value=0, n=1)
        self.acceleration = ExponentialMovingAverage(recency_weight, value=0, n=1)
        self.loss = ExponentialMovingAverage(recency_weight, value=1, n=1)
        self.pos_err = ExponentialMovingAverage(recency_weight, value=1, n=1)
        self.neg_err = ExponentialMovingAverage(recency_weight, value=-1, n=1)
        self.vel_var = RecencyWeightedVariance(recency_weight, ex_value=0, ex2_value=0, n=1)
        self.loss_var = RecencyWeightedVariance(recency_weight, ex_value=1, ex2_value=1, n=1)
        self.scale = scale
        self.recency_weight = recency_weight

    @property
    def lag(self):
        return LagTransform(self.scale)

    def update_velocity(self, vels: UnivariateTimeSeries):
        self.velocity.add_batch(vels.values)
        self.vel_var.add_batch(vels.values)

    def update_acceleration(self, accs: UnivariateTimeSeries):
        self.acceleration.add_batch(accs.values)

    def update_loss(self, losses: Union[List[float], UnivariateTimeSeries]):
        if isinstance(losses, UnivariateTimeSeries):
            losses = losses.values

        # update errs
        for loss in losses:
            if loss > 0:
                self.pos_err.add(loss)
            elif loss < 0:
                self.neg_err.add(loss)

        # update loss
        losses = np.abs(losses).tolist()
        self.loss.add_batch(losses)
        self.loss_var.add_batch(losses)

    def tune(self, losses: List[float], eta: float):
        """
        Tunes the recency weight according to recent forecast losses.

        :param losses: List of recent losses.
        :param eta: Constant by which to scale the update to the recency weight.
            A bigger eta means more aggressive updates to the recency_weight.
        """
        if self.velocity.n == 1:
            return
        tune_stats = [self.velocity, self.vel_var] + [self.acceleration] * (self.acceleration.n > 1)
        for loss in losses:
            nerr = np.tanh(eta * loss)
            for stat in tune_stats:
                gap = 1.0 - stat.recency_weight if nerr > 0 else stat.recency_weight
                stat.recency_weight += eta * gap * nerr


class DeltaEstimator:
    """
    Class for estimating the delta for MSES.
    """

    def __init__(
        self,
        max_scale: int,
        recency_weight: float,
        accel_weight: float,
        optimize_acc: bool,
        eta: float,
        phi: float,
        data: UnivariateTimeSeries = None,
        stats: Dict[int, DeltaStats] = None,
    ):
        """
        :param max_scale: Delta Estimator can estimate delta over multiple scales, or
            time steps, ranging from 1,2,...,max_scale.
        :param recency_weight: The recency weight parameter to use when estimating delta_hat.
        :param accel_weight: The weight to scale the acceleration by when computing delta_hat.
            Specifically, delta_hat(b+h) = velocity(b+h) + accel_weight * acceleration(b+h).
        :param optimize_acc: If True, the acceleration correction will only be used at scales
            ranging from 1,...,max_scale/2.
        :param eta: The parameter used to control the rate at which recency_weight gets
            tuned when online updates are made to the model and losses can be computed.
        :param data: The data to initialize the delta estimator with.
        :param stats: Dictionary mapping scales to DeltaStats objects to be used for delta
            estimation.
        """
        self.stats = dict() if stats is None else stats
        self.recency_weight = recency_weight
        self.max_scale = max_scale
        self.accel_weight = accel_weight
        self.optimize_acc = optimize_acc
        self.eta = eta
        self.phi = phi
        self.data = UnivariateTimeSeries.empty() if data is None else data

    @property
    def max_scale(self):
        return self._max_scale

    @property
    def acc_max_scale(self):
        return floor(self.max_scale / 2) if self.optimize_acc else self.max_scale

    @max_scale.setter
    def max_scale(self, scale: int):
        self._max_scale = scale
        # update delta stats
        self.stats = {s: ds for s, ds in self.stats.items() if s <= scale}
        for scale in range(1, self.max_scale + 1):
            if scale not in self.stats:
                self.stats[scale] = DeltaStats(scale, self.recency_weight)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: UnivariateTimeSeries):
        """
        Only keeps data necessary for future updates
        let previous seen and used data be ...,x(t-1), x(t).
        If an incoming batch of data x(t+1),...,x(t+B) arrives, we need
        to compute both v(t+1),...,v(t+B) and a(t+1),...,a(t+B).

        v(t+1),...,v(t+B) requires x(t+1-scale),...,x(t+B)
        a(t+1),...,a(t+B) requires v(t+1-scale),...,v(t+B) requires x(t+1-2*scale),...,x(t+B)

        From the data already seen we need to retain x(t+1-2*scale),...,x(t)
        which are the last 2*scale points.

        :param data: time series to retain for future updates.
        """
        self._data = data[-(2 * self.max_scale + 1) :]

    @property
    def x(self):
        return self.data.values

    def __setstate__(self, state):
        for name, value in state.items():
            if name == "_data" and isinstance(value, pd.Series):
                setattr(self, name, UnivariateTimeSeries.from_pd(value))
            else:
                setattr(self, name, value)

    def train(self, new_data: UnivariateTimeSeries):
        """
        Updates the delta statistics: velocity, acceleration and velocity
        standard error at each scale using new data.

        :param new_data: new datapoints in the time series.
        """
        needed_data = self.data.concat(new_data)
        for scale, stat in self.stats.items():
            if len(needed_data) < scale + 1:
                continue
            # compute and update velocity
            vels = stat.lag(needed_data.to_ts())
            bs = min(len(new_data), len(vels))
            stat.update_velocity(vels[-bs:].univariates[vels.names[0]])
            # compute and update acceleration
            if len(needed_data) >= 2 * scale + 1 and scale <= self.acc_max_scale:
                accs = stat.lag(vels)
                bs = min(len(new_data), len(accs))
                stat.update_acceleration(accs[-bs:].univariates[accs.names[0]])
        # update data to retain for future updates
        self.data = needed_data

    def process_losses(self, scale_losses: Dict[int, List[float]], tune_recency_weights: bool = False):
        """
        Uses recent forecast errors to improve the delta estimator. This is done by updating
        the recency_weight that is used by delta stats at particular scales.

        :param scale_losses: A dictionary mapping a scale to a list of forecasting errors
            that associated with that scale.
        """
        for scale, losses in scale_losses.items():
            stat = self.stats.get(scale)
            stat.update_loss(losses)
            if tune_recency_weights:
                stat.tune(losses, self.eta)

    def velocity(self, scale: int) -> float:
        stat = self.stats.get(scale)
        return 0 if stat is None else stat.velocity.value

    def acceleration(self, scale: int) -> float:
        stat = self.stats.get(scale)
        return 0 if stat is None else stat.acceleration.value

    def vel_err(self, scale: int) -> float:
        stat = self.stats.get(scale)
        return np.inf if stat is None else stat.vel_var.sd

    def pos_err(self, scale: int) -> float:
        stat = self.stats.get(scale)
        return 1 if stat is None else stat.pos_err.value

    def neg_err(self, scale: int) -> float:
        stat = self.stats.get(scale)
        return 1 if stat is None else stat.neg_err.value

    def loss_err(self, scale: int) -> float:
        stat = self.stats.get(scale)
        if stat is None or stat.loss.value is None:
            return np.inf
        return (scale ** self.phi) * stat.loss.value * stat.loss_var.se

    def delta_hat(self, scale: int) -> float:
        return self.velocity(scale) + self.accel_weight * self.acceleration(scale)
