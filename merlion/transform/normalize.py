#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that rescale the input or otherwise normalize it.
"""
from collections import OrderedDict
import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from sklearn.preprocessing import StandardScaler

from merlion.transform.base import InvertibleTransformBase, TransformBase
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


class AbsVal(TransformBase):
    """
    Takes the absolute value of the input time series.
    """

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the "pseudo-inverse" is just the identity (i.e. we lose sign information).
        """
        return False

    @property
    def identity_inversion(self):
        return True

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

        bias = self.bias if isinstance(self.bias, Mapping) else {name: self.bias for name in time_series.names}
        scale = self.scale if isinstance(self.scale, Mapping) else {name: self.scale for name in time_series.names}
        assert set(time_series.names).issubset(bias.keys()) and set(time_series.names).issubset(scale.keys())

        new_vars = OrderedDict()
        for name, var in time_series.items():
            if self.normalize_bias:
                var = var - bias[name]
            if self.normalize_scale:
                var = var / scale[name]
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series._is_aligned
        return ret

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        if not self.is_trained:
            raise RuntimeError(f"Cannot use {type(self).__name__} without training it first!")
        bias = self.bias if isinstance(self.bias, Mapping) else {name: self.bias for name in time_series.names}
        scale = self.scale if isinstance(self.scale, Mapping) else {name: self.scale for name in time_series.names}
        assert set(time_series.names).issubset(bias.keys()) and set(time_series.names).issubset(scale.keys())

        new_vars = OrderedDict()
        for name, var in time_series.items():
            if self.normalize_scale:
                var = var * scale[name]
            if self.normalize_bias:
                var = var + bias[name]
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
        bias, scale = {}, {}
        for name, var in time_series.items():
            scaler = StandardScaler().fit(var.np_values.reshape(-1, 1))
            bias[name] = float(scaler.mean_)
            scale[name] = float(scaler.scale_)
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
        bias, scale = {}, {}
        for name, var in time_series.items():
            minval, maxval = var.min(), var.max()
            bias[name] = minval
            scale[name] = np.maximum(1e-8, maxval - minval)
        self.bias = bias
        self.scale = scale


class BoxCoxTransform(InvertibleTransformBase):
    """
    Applies the Box-Cox power transform to the time series, with power lmbda.
    When lmbda is None, we
    When lmbda > 0, it is ((x + offset) ** lmbda - 1) / lmbda.
    When lmbda == 0, it is ln(lmbda + offset).
    """

    def __init__(self, lmbda=None, offset=0.0):
        super().__init__()
        if lmbda is not None:
            if isinstance(lmbda, dict):
                assert all(isinstance(x, (int, float)) for x in lmbda.values())
            else:
                assert isinstance(lmbda, (int, float))
        self.lmbda = lmbda
        self.offset = offset

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the Box-Cox transform does is stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        if self.lmbda is None:
            self.lmbda = {name: scipy.stats.boxcox(var.np_values + self.offset)[1] for name, var in time_series.items()}
            logger.info(f"Chose Box-Cox lambda = {self.lmbda}")
        elif not isinstance(self.lmbda, Mapping):
            self.lmbda = {name: self.lmbda for name in time_series.names}
        assert len(self.lmbda) == time_series.dim

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for name, var in time_series.items():
            y = scipy.special.boxcox(var + self.offset, self.lmbda[name])
            var = pd.Series(y, index=var.index, name=var.name)
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        return TimeSeries(new_vars)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = []
        for name, var in time_series.items():
            lmbda = self.lmbda[name]
            if lmbda > 0:
                var = (lmbda * var + 1) ** (1 / lmbda)
                nanvals = var.isna()
                if nanvals.any():
                    var[nanvals] = 0
            else:
                var = var.apply(np.exp)
            new_vars.append(UnivariateTimeSeries.from_pd(var - self.offset))

        return TimeSeries(new_vars)
