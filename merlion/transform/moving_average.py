#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that compute moving averages and k-step differences.
"""

from collections import OrderedDict
import logging
from typing import List, Sequence

import numpy as np
import scipy.signal
from scipy.stats import norm

from merlion.transform.base import TransformBase, InvertibleTransformBase
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


class MovingAverage(TransformBase):
    """
    Computes the n_steps-step moving average of the time series, with
    the given relative weights assigned to each time in the moving average
    (default is to take the non-weighted average). Zero-pads the input time
    series to the left before taking the moving average.
    """

    def __init__(self, n_steps: int = None, weights: Sequence[float] = None):
        super().__init__()
        assert (
            n_steps is not None or weights is not None
        ), "Must specify at least one of n_steps or weights for MovingAverage"
        if weights is None:
            weights = np.ones(n_steps) / n_steps
        elif n_steps is None:
            n_steps = len(weights)
        else:
            assert len(weights) == n_steps
        self.n_steps = n_steps
        self.weights = weights

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        conv_remainders = []
        for name, var in time_series.items():
            t, x = var.index, var.np_values
            ma = scipy.signal.correlate(x, self.weights, mode="full")
            y0, y1 = ma[: len(x)], ma[len(x) :]
            new_vars[name] = UnivariateTimeSeries(t, y0)
            conv_remainders.append(y1)

        self.inversion_state = conv_remainders
        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for (name, var), y1 in zip(time_series.items(), self.inversion_state):
            t, y0 = var.index, var.np_values
            x = scipy.signal.deconvolve(np.concatenate((y0, y1)), self.weights[-1::-1])[0]
            new_vars[name] = UnivariateTimeSeries(t, x)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret


class MovingPercentile(TransformBase):
    """
    Computes the n-step moving percentile of the time series.
    For datapoints at the start of the time series which are preceded by
    fewer than ``n_steps`` datapoints, the percentile is computed using only the
    available datapoints.
    """

    def __init__(self, n_steps: int, q: float):
        """
        :param q: The percentile to use. Between 0 and 100 inclusive.
        :param n_steps: The number of steps to use.
        """
        super().__init__()
        assert 0 <= q <= 100
        assert 1 <= n_steps
        self.n_steps = int(n_steps)
        self.q = q

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for name, var in time_series.items():
            x = var.np_values
            new_x = []
            for i, _ in enumerate(x):
                window = x[max(0, i - self.n_steps + 1) : i + 1]
                new_x.append(np.percentile(window, self.q))
            new_vars[name] = UnivariateTimeSeries(var.index, new_x)
        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret


class ExponentialMovingAverage(InvertibleTransformBase):
    r"""
    Computes the exponential moving average (normalized or un-normalized) of the
    time series, with smoothing factor alpha (lower alpha = more smoothing).
    alpha must be between 0 and 1.
    
    The unnormalized moving average ``y`` of ``x`` is computed as

    .. math::
        \begin{align*}
        y_0 & = x_0 \\
        y_i & = (1 - \alpha) \cdot y_{i-1} + \alpha \cdot x_i
        \end{align*}
    
    The normalized moving average ``y`` of ``x`` is computed as

    .. math::

        y_i = \frac{x_i + (1 - \alpha) x_{i-1} + \ldots + (1 - \alpha)^i x_0}
        {1 + (1 - \alpha) + \ldots + (1 - \alpha)^i}

    Upper and lower confidence bounds, ``l`` and ``u``, of the exponential moving
    average are computed using the exponential moving standard deviation, ``s``, and ``y`` as

    .. math::
        l_i = y_i + z_{\frac{1}{2} (1-p)} \times s_i \\
        u_i = u_o + z_{\frac{1}{2} (1+p)} \times s_i

    If condfidence bounds are included, the returned time series will contain
    the upper and lower bounds as additional univariates. For example if the 
    transform is applied to a time series with two univariates "x" and "y", 
    the resulting time series will contain univariates with the following names:
    "x", "x_lb", "x_ub", "y", "y_lb", "y_ub".
    """

    def __init__(self, alpha: float, normalize: bool = True, p: float = 0.95, ci: bool = False):
        """
        :param alpha: smoothing factor to use for exponential weighting.
        :param normalize: If True, divide by the decaying adjustment in
            beginning periods.
        :param p: confidence level to use if returning the upper and lower
            bounds of the confidence interval.
        :param ci: If True, return the the upper and lower confidence bounds
            of the the exponential moving average as well.
        """
        super().__init__()
        self.alpha = alpha
        self.normalize = normalize
        self.p = p
        self.ci = ci

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the exponential moving average is stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for name, var in time_series.items():
            emw = var.to_pd().ewm(alpha=self.alpha, adjust=self.normalize)
            ema = emw.mean()
            new_vars[name] = UnivariateTimeSeries.from_pd(ema)
            if self.ci:
                ems = emw.std()
                ems[0] = ems[1]
                new_vars[f"{name}_lb"] = UnivariateTimeSeries.from_pd(ema + norm.ppf(0.5 * (1 - self.p)) * ems)
                new_vars[f"{name}_ub"] = UnivariateTimeSeries.from_pd(ema + norm.ppf(0.5 * (1 + self.p)) * ems)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for name, var in time_series.items():
            # check whether varaiable is an upper or lower confidence bound
            if isinstance(name, str) and (name.endswith("_lb") or name.endswith("_ub")):
                continue
            t, y = var.index, var.np_values

            # Geometric series formula for (1 - alpha)^0 + ... + (1 - alpha)^i
            # to unnormalize the EWM before inverting it
            if self.normalize:
                weights = 1 - (1 - self.alpha) ** np.arange(1, len(y) + 1)
                y = y * weights / self.alpha
                x = y[1:] - (1 - self.alpha) * y[:-1]

            # Direct inversion of one-step update for unnormalized EWMA
            else:
                x = (y[1:] - (1 - self.alpha) * y[:-1]) / self.alpha

            x = np.concatenate((y[:1], x))
            new_vars[name] = UnivariateTimeSeries(t, x)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret


class DifferenceTransform(InvertibleTransformBase):
    """
    Applies a difference transform to the input time series. We include it
    as a moving average because we can consider the difference transform
    to be a 2-step moving "average" with weights w = [-1, 1].
    """

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        x0 = []
        new_vars = OrderedDict()
        for name, var in time_series.items():
            x0.append(var[0])
            if len(var) <= 1:
                logger.warning(f"Cannot apply a difference transform to a time series of length {len(var)} < 2")
                new_vars[name] = UnivariateTimeSeries([], [])
            else:
                new_vars[name] = UnivariateTimeSeries.from_pd(var.diff())[1:]

        self.inversion_state = x0
        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for (t0, x0), (name, var) in zip(self.inversion_state, time_series.items()):
            var = UnivariateTimeSeries([t0], [x0]).concat(var).cumsum()
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series.is_aligned
        return ret


class LagTransform(InvertibleTransformBase):
    """
    Applies a lag transform to the input time series. Each x(i) gets mapped
    to x(i) - x(i-k). We include it as a moving average because we can consider
    the lag transform to be a k+1-step moving "average" with weights
    w = [-1, 0,..., 0, 1]. One may optionally left-pad the sequence with the
    first value in the time series.
    """

    def __init__(self, k: int, pad: bool = False):
        super().__init__()
        assert k >= 1
        self.k = k
        self.pad = pad

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        all_tk, all_xk = [], []
        new_vars = OrderedDict()
        for name, var in time_series.items():
            # Apply any x-padding or t-truncating necessary
            t, x = var.index, var.np_values
            all_xk.append(x[: self.k])
            if self.pad:
                all_tk.append(t[:0])
                x = np.concatenate((np.full(self.k, x[0]), x))
            else:
                all_tk.append(t[: self.k])
                t = t[self.k :]

            if len(var) <= self.k and not self.pad:
                logger.warning(
                    f"Cannot apply a {self.k}-lag transform to a time series of length {len(var)} <= {self.k}"
                )
                new_vars[name] = UnivariateTimeSeries([], [])
            else:
                new_vars[name] = UnivariateTimeSeries(t, x[self.k :] - x[: -self.k])

        self.inversion_state = all_tk, all_xk
        return TimeSeries(new_vars)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        all_tk, all_xk = self.inversion_state
        new_vars = OrderedDict()
        for (name, var), tk, xk in zip(time_series.items(), all_tk, all_xk):
            t = tk.union(var.index)
            if len(t) == len(xk) + len(var):  # no padding
                y = np.concatenate((xk, var.np_values))
            elif len(t) == len(var):  # padding
                y = np.asarray(var.values)
                y[: len(xk)] = xk
            else:
                raise RuntimeError("Something went wrong: inversion state has unexpected size.")

            x = np.zeros(len(t))
            for i in range(self.k):
                x[i :: self.k] = np.cumsum(y[i :: self.k])
            new_vars[name] = UnivariateTimeSeries(t, x)
        return TimeSeries(new_vars)

    def compute_lag(self, var: UnivariateTimeSeries) -> UnivariateTimeSeries:
        t, x = var.index, var.np_values
        if self.pad:
            x = np.concatenate((np.full(self.k, x[0]), x))
        vals = x[self.k :] - x[: -self.k]
        times = t if self.pad else t[self.k :]
        return UnivariateTimeSeries(times, vals)
