#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that rescale the input or otherwise normalize it.
"""
from collections import OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import StandardScaler

from merlion.transform.base import InvertibleTransformBase, TransformBase
from merlion.utils import UnivariateTimeSeries, TimeSeries


class AbsVal(TransformBase):
    """
    Takes the absolute value of the input time series.
    """

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the "pseudo-inverse" is just the identity (i.e. we
        lose sign information).
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        return TimeSeries(
            OrderedDict(
                (name, UnivariateTimeSeries(var.index, np.abs(var.np_values))) for name, var in time_series.items()
            )
        )


class Rescale(InvertibleTransformBase):
    """
    Rescales the bias & scale of input vectors or scalars by pre-specified amounts.
    """

    def __init__(self, bias=0.0, scale=1.0, normalize_bias=True, normalize_scale=True):
        super().__init__()
        self.bias = bias
        self.scale = scale
        self.normalize_bias = normalize_bias
        self.normalize_scale = normalize_scale

    @property
    def requires_inversion_state(self):
        """
        ``False`` because rescaling operations are stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    @property
    def is_trained(self):
        return self.bias is not None and self.scale is not None

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        if not self.is_trained:
            raise RuntimeError(f"Cannot use {type(self).__name__} without training it first!")

        d = time_series.dim
        bias = self.bias if isinstance(self.bias, Iterable) else [self.bias] * d
        scale = self.scale if isinstance(self.scale, Iterable) else [self.scale] * d
        assert len(bias) == d and len(scale) == d, (
            f"Expected {len(bias)}-dimensional time series to match scale and "
            f"bias, but got {d}-dimensional time series instead."
        )

        new_vars = OrderedDict()
        for i, (name, var) in enumerate(time_series.items()):
            if self.normalize_bias:
                var = var - bias[i]
            if self.normalize_scale:
                var = var / scale[i]
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series._is_aligned
        return ret

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        if not self.is_trained:
            raise RuntimeError(f"Cannot use {type(self).__name__} without training it first!")
        d = time_series.dim
        bias = self.bias if isinstance(self.bias, Iterable) else [self.bias] * d
        scale = self.scale if isinstance(self.scale, Iterable) else [self.scale] * d
        assert len(bias) == d and len(scale) == d, (
            f"Expected {len(bias)}-dimensional time series to match scale and "
            f"bias, but got {d}-dimensional time series instead."
        )

        new_vars = OrderedDict()
        for i, (name, var) in enumerate(time_series.items()):
            if self.normalize_scale:
                var = var * scale[i]
            if self.normalize_bias:
                var = var + bias[i]
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series._is_aligned
        return ret


class MeanVarNormalize(Rescale):
    """
    A learnable transform that rescales the values of a time series to have
    zero mean and unit variance.
    """

    def __init__(self, bias=None, scale=None, normalize_bias=True, normalize_scale=True):
        super().__init__(bias, scale, normalize_bias, normalize_scale)

    def train(self, time_series: TimeSeries):
        bias, scale = [], []
        for var in time_series.univariates:
            scaler = StandardScaler().fit(var.np_values.reshape(-1, 1))
            bias.append(float(scaler.mean_))
            scale.append(float(scaler.scale_))
        self.bias = bias
        self.scale = scale


class MinMaxNormalize(Rescale):
    """
    A learnable transform that rescales the values of a time series to be
    between zero and one.
    """

    def __init__(self, bias=None, scale=None, normalize_bias=True, normalize_scale=True):
        super().__init__(bias, scale, normalize_bias, normalize_scale)

    def train(self, time_series: TimeSeries):
        bias, scale = [], []
        for var in time_series.univariates:
            minval, maxval = var.min(), var.max()
            bias.append(minval)
            scale.append(np.maximum(1e-8, maxval - minval))
        self.bias = bias
        self.scale = scale


class PowerTransform(InvertibleTransformBase):
    """
    Applies the Box-Cox power transform to the time series, with power lmbda.
    When lmbda > 0, it is ((x + offset) ** lmbda - 1) / lmbda.
    When lmbda == 0, it is ln(lmbda + offset).
    """

    def __init__(self, lmbda=0.0, offset=0.0):
        super().__init__()
        assert lmbda >= 0
        self.lmbda = lmbda
        self.offset = offset

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the Box-Cox transform does is stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = []
        for var in time_series.univariates:
            y = scipy.stats.boxcox(var + self.offset, self.lmbda)
            var = pd.Series(y, index=var.index, name=var.name)
            new_vars.append(UnivariateTimeSeries.from_pd(var))

        return TimeSeries(new_vars)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = []
        for var in time_series.univariates:
            if self.lmbda > 0:
                var = (self.lmbda * var + 1).log() / self.lmbda
            new_vars.append(UnivariateTimeSeries.from_pd(var.apply(np.exp)))

        return TimeSeries(new_vars)
