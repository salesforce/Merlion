#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that inject synthetic anomalies into time series.
"""

from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

from merlion.transform.base import Identity, TransformBase
from merlion.transform.bound import LowerUpperClip
from merlion.transform.moving_average import DifferenceTransform
from merlion.utils.time_series import UnivariateTimeSeries, TimeSeries
from merlion.utils.resample import get_gcd_timedelta


class Anomalize(TransformBase):
    """
    Injects anomalies into a time series with controlled randomness and returns
    both the anomalized time series along with associated anomaly labels.
    """

    def __init__(self, anom_prob: float = 0.01, natural_bounds: Tuple[float, float] = (None, None), **kwargs):
        """
        :param anom_prob: The probability of anomalizing a particular data point.
        :param natural_bounds: Upper and lower natrual boundaries which injected anomalies should
            a particular time series must stay within.
        """
        super().__init__(**kwargs)
        assert 0 <= anom_prob <= 1
        self.anom_prob = anom_prob
        self.natural_bounds = natural_bounds

    @property
    def natural_bounds(self):
        return self.nat_lower, self.nat_upper

    @natural_bounds.setter
    def natural_bounds(self, bounds: Tuple[float, float]):
        lower, upper = bounds
        if lower is not None or upper is not None:
            self.bound = LowerUpperClip(lower, upper)
        else:
            self.bound = Identity()
        self.nat_lower = lower
        self.nat_upper = upper

    @property
    def is_trained(self) -> bool:
        return True

    def random_is_anom(self):
        return np.random.uniform() < self.anom_prob

    def __call__(self, time_series: TimeSeries, label_anoms: bool = True) -> TimeSeries:
        """
        :param label_anoms: If True, label injected anomalies with 1, otherwise, do not
            label injected anomalies.
        """
        if not self.is_trained:
            raise RuntimeError(f"Cannot use {type(self).__name__} without training it first!")

        assert time_series.dim <= 2, (
            "anomalize transforms may only be applied to univariate time series "
            "or bivariate time series, in which the second variable is a series "
            "of anomaly labels"
        )

        if time_series.dim == 2:
            var, prev_label_var = [time_series.univariates[name] for name in time_series.names]
            assert "anom" in prev_label_var.name
        else:
            var, prev_label_var = time_series.univariates[time_series.names[0]], None

        new_var, label_var = self._anomalize_univariate(var)
        if not label_anoms:
            label_var = UnivariateTimeSeries(label_var.time_stamps, [0] * len(prev_label_var), prev_label_var.name)

        # combine label univariates
        if prev_label_var is not None:
            labels = []
            for (t1, lab), (t2, prev_lab) in zip(prev_label_var, label_var):
                labels.append(max(lab, prev_lab))
            label_var = UnivariateTimeSeries(label_var.time_stamps, labels, label_var.name)

        # bound result
        return TimeSeries.from_ts_list([self.bound(new_var.to_ts()), label_var.to_ts()])

    @abstractmethod
    def _anomalize_univariate(self, var: UnivariateTimeSeries):
        pass


class Shock(Anomalize):
    """
    Injects random spikes or dips into a time series.

    Letting ``y_t`` be a time series, if an anomaly is injected into 
    the time series at time ``t``, the anomalous value that gets injected is as follows:

    .. math::
        \\tilde{y}_t &= y_t + \\text{shock} \\\\
        \\begin{split}
        \\text{where } \\space & \\text{shock} = Sign \\times Z\\times \\text{RWSD}_{\\alpha}(y_t), \\\\
        & Z \\sim \\mathrm{Unif}(a,b), \\\\
        & Sign \\text{ is a random sign} \\\\
        \\end{split}


    Additionally, the ``shock`` that is added to ``y_t`` is also applied to 
    ``y_t+1``, ... ``y_w-1``, where ``w``, known as the "anomaly width" is
    randomly determined by a random draw from a uniform distribution.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        pos_prob: float = 1.0,
        sd_range: Tuple[float, float] = (3, 6),
        anom_width_range: Tuple[int, int] = (1, 5),
        persist_shock: bool = False,
        **kwargs,
    ):
        """
        :param alpha: The recency weight to use when calculating recency-weighted
            standard deviation.
        :param pos_prob: The probably with which a shock's sign is positive.
        :param sd_range: The range of standard units that is used to create a shock
        :param anom_width_range: The range of anomaly widths.
        :param persist_shock: whether to apply the shock to all successive datapoints.
        """
        super().__init__(**kwargs)
        assert 0.0 <= pos_prob <= 1.0
        self.alpha = alpha
        self.pos_prob = pos_prob
        self.sd_range = sd_range
        self.anom_width_range = anom_width_range
        self.persist_shock = persist_shock

    @property
    def anom_width_range(self):
        return self.width_lower, self.width_upper

    @anom_width_range.setter
    def anom_width_range(self, range: Tuple[int, int]):
        lower, upper = range
        assert 0 < lower <= upper
        self.width_lower = lower
        self.width_upper = upper

    @property
    def sd_range(self):
        return self.sd_lower, self.sd_upper

    @sd_range.setter
    def sd_range(self, range: Tuple[float, float]):
        lower, upper = range
        assert lower <= upper
        self.sd_lower = lower
        self.sd_upper = upper

    def random_sd_units(self):
        sign = 1 if np.random.uniform() < self.pos_prob else -1
        return sign * np.random.uniform(self.sd_lower, self.sd_upper)

    def random_anom_width(self):
        return np.random.choice(range(self.width_lower, self.width_upper + 1))

    def random_is_anom(self):
        return np.random.uniform() < self.anom_prob

    def train(self, time_series: TimeSeries):
        """
        The `Shock` transform doesn't require training.
        """
        pass

    def _anomalize_univariate(self, var: UnivariateTimeSeries) -> Tuple[UnivariateTimeSeries, UnivariateTimeSeries]:
        ems = var.to_pd().ewm(alpha=self.alpha, adjust=False).std(bias=True)

        new_vals, labels = [], []
        anom_width, shock = 0, 0
        for ((t, x), sd) in zip(var, ems):
            if anom_width == 0:
                is_anom = self.random_is_anom()
                if is_anom:
                    shock = self.random_sd_units() * sd
                    anom_width = self.random_anom_width() - 1
                    val = x + shock
                else:
                    val = x + shock * self.persist_shock
            elif anom_width > 0:
                is_anom = True
                val = x + shock
                anom_width -= 1

            new_vals.append(val)
            labels.append(is_anom)

        anomalized_var = UnivariateTimeSeries(var.time_stamps, new_vals, var.name)
        labels_var = UnivariateTimeSeries(var.time_stamps, labels, "anomaly")

        return anomalized_var, labels_var


class LevelShift(Shock):
    """
    Injects random level shift anomalies into a time series.

    A level shift is a sudden change of level in a time series. It is equivalent to
    a shock that, when applied to ``y_t``, is also applied to every datapoint after ``t``.
    """

    def __init__(self, **kwargs):
        kwargs["persist_shock"] = True
        # We count a level shift anomaly as lasting for 20 points after the fact
        kwargs["anom_width_range"] = (20, 20)
        super().__init__(**kwargs)


class TrendChange(Anomalize):
    r"""
    Injects random trend changes into a time series.

    At a high level, the transform tracks the velocity (trend) of a time series
    and then, when injecting a trend change at a particular time, it scales
    the current velocity by a random factor. The disturbance to the velocity is 
    persisted to values in the near future, thus emulating a sudden change of trend.

    Let, ``(a,b)`` be the scale range. If the first trend change happens at time ``t*``, 
    it is injected as follows:

    .. math::
        \tilde{y}_{t^*} = y_{t^*-1} + v_{t^*} + \Delta v_{t^*} \\
        \begin{align*}
        \text{where } & \Delta v_{t^*} = Sign \times Z \times v_{t^*}, \\
        & v_{t^*} = y_{t^*} - y_{t^*-1}
        & Z \sim Unif(a,b), \\
        & Sign \text{ is a random sign} \\
        \end{align*}

    Afterward, the trend change is persisted and ``y_t`` (for ``t > t*``) is changed as follows:

    .. math::
        \tilde{y}_{t} = \tilde{y}_{t-1} + v_t + \beta \times \Delta v_{t^*}
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.95,
        pos_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.5, 3.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        """
        :param alpha: The recency weight to use when calculating recency-weighted
            standard deviation.
        :param beta: A parameter controlling the degree of trend change persistence.
        :param pos_prob: The probably with which a shock's sign is positive.
        :param scale_range: The range of possible values by which a time series's
            velocity will be scaled.
        """
        assert all(0 <= param <= 1 for param in (alpha, beta, pos_prob))
        self.alpha = alpha
        self.beta = beta
        self.scale_range = scale_range
        self.pos_prob = pos_prob

    @property
    def scale_range(self):
        return self.scale_lower, self.scale_upper

    @scale_range.setter
    def scale_range(self, scale_range: Tuple[float, float]):
        lower, upper = scale_range
        assert 0 < lower <= upper
        self.scale_lower = lower
        self.scale_upper = upper

    def random_scale(self):
        sign = 1 if np.random.uniform() < self.pos_prob else -1
        return sign * np.random.uniform(self.scale_lower, self.scale_upper)

    def _anomalize_univariate(self, var: UnivariateTimeSeries):
        vels = [0] + var.diff()[1:].tolist()
        emv = pd.Series(vels).ewm(alpha=self.alpha, adjust=False).mean()
        new_vals, labels = [], []

        x_prev, v_delta = var.values[0], vels[0]
        for v, mv in zip(vels, emv):
            is_anom = self.random_is_anom()
            v_delta = self.random_scale() * mv if is_anom else self.beta * v_delta
            x = x_prev + v + v_delta
            new_vals.append(x)
            labels.append(is_anom)
            x_prev = x

        anomalized_var = UnivariateTimeSeries(var.time_stamps, new_vals, var.name)
        labels_var = UnivariateTimeSeries(var.time_stamps, labels, "anomaly")

        return anomalized_var, labels_var

    def train(self, time_series: TimeSeries):
        """
        The `TrendChange` transform doesn't require training.
        """
        pass
