#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from abc import abstractmethod
from typing import List
from math import sqrt

import numpy as np


class IStat:
    """
    An abstract base class for computing various statistics incrementally,
    with emphasis on recency-weighted variants.
    """

    def __init__(self, value: float = None, n: int = 0):
        """
        :param value: Initial value of the statistic. Defaults to None.
        :param n: Initial sample size. Defaults to 0.
        """
        if n > 0:
            assert value is not None
        self.value = value
        self.n = n

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n: int):
        assert n >= 0
        self._n = n

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    @abstractmethod
    def add(self, x):
        """
        Add a new value to update the statistic.
        :param x: new value to add to the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def drop(self, x):
        """
        Drop a value to update the statistic.
        :param x: value to drop from the sample.
        """
        raise NotImplementedError

    def add_batch(self, batch: List[float]):
        """
        Add a batch of new values to update the statistic.
        :param batch: new values to add to the sample.
        """
        for x in batch:
            self.add(x)

    def drop_batch(self, batch: List[float]):
        """
        Drop a batch of new values to update the statistic.
        :param batch: new values to add to the sample.
        """
        for x in batch:
            self.drop(x)


class Mean(IStat):
    """
    Class for incrementally computing the mean of a series of numbers.
    """

    def __init__(self, value: float = None, n: int = 0):
        super().__init__(value=value, n=n)
        self.sum = n * value if n > 0 else None

    @IStat.value.getter
    def value(self):
        if self.sum is None:
            return None
        return self.sum / self.n

    def add(self, x):
        assert isinstance(x, (int, float))
        self.n += 1
        if self.n == 1:
            self._add_first(x)
        else:
            self._add(x)

    def _add_first(self, x):
        self.sum = x

    def _add(self, x):
        self.sum += float(x)

    def drop(self, x):
        assert isinstance(x, (int, float))
        if self.n == 0:
            return
        self.n -= 1
        if self.n == 0:
            self.sum = None
        else:
            self.sum -= float(x)


class Variance(IStat):
    """
    Class for incrementally computing the variance of a series of numbers.
    """

    mean_class = Mean

    def __init__(self, ex_value: float = None, ex2_value: float = None, n: int = 0, ddof: int = 1):
        """
        :param ex_value: Initial value of the first moment (mean).
        :param ex2_value: Initial value of the second moment.
        :param n: Initial sample size.
        :param ddof: The delta degrees of freedom to use when correcting
            the estimate of the variance.

        .. math::
            \\text{Var}(x_i) = \\text{E}(x_i^2) - \\text{E}(x_i)^2
        """
        if ex_value is not None and ex2_value is not None:
            super().__init__(value=ex2_value - ex_value ** 2, n=n)
        else:
            super().__init__()
        self.ex = self.mean_class(value=ex_value, n=n)
        self.ex2 = self.mean_class(value=ex2_value, n=n)
        self.ddof = ddof

    def add(self, x):
        self.n += 1
        self.ex.add(x)
        self.ex2.add(x ** 2)

    def drop(self, x):
        if self.n == 0:
            return
        self.n -= 1
        self.ex.drop(x)
        self.ex2.drop(x ** 2)

    @property
    def true_value(self):
        if self.ex2.value is None or self.ex.value is None:
            return None
        return max(0, self.ex2.value - self.ex.value ** 2)

    @property
    def corrected_value(self):
        if self.true_value is None:
            return None
        elif self.n - self.ddof <= 0:
            return np.inf
        return (self.n / (self.n - self.ddof)) * self.true_value

    @IStat.value.getter
    def value(self):
        if self.corrected_value is None:
            return None
        return self.corrected_value + 1e-16

    @property
    def sd(self):
        if self.true_value is None:
            return None
        return sqrt(self.corrected_value) + 1e-16

    @property
    def se(self):
        if self.sd is None:
            return None
        return self.sd / sqrt(self.n)


class ExponentialMovingAverage(Mean):
    """
    Class for incrementally computing the exponential moving average of a series of numbers.
    """

    def __init__(self, recency_weight: float = 0.1, **kwargs):
        """
        :param recency_weight: Recency weight to use when updating the
            exponential moving average.

        Letting ``w`` be the recency weight,

        .. math:: 
            \\begin{align*}
            \\text{EMA}_w(x_0) & = x_0 \\\\
            \\text{EMA}_w(x_t) & = w \\cdot x_t + (1-w) \\cdot \\text{EMA}_w(x_{t-1})
            \\end{align*}
        """
        super().__init__(**kwargs)
        self.recency_weight = recency_weight

    @property
    def recency_weight(self):
        return self._recency_weight

    @recency_weight.setter
    def recency_weight(self, weight: float):
        assert 0.0 < weight <= 1.0
        self._recency_weight = weight

    @IStat.value.getter
    def value(self):
        return self._value

    def _add_first(self, x):
        if self.value is None:
            self.value = x
        else:
            self._add(x)

    def _add(self, x):
        self.value = (1 - self.recency_weight) * self.value + self.recency_weight * x

    def drop(self, x):
        """
        Exponential Moving Average does not support dropping values
        """
        pass


class RecencyWeightedVariance(Variance):
    """
    Class for incrementally computing the recency-weighted variance of a series of numbers.
    """

    mean_class = ExponentialMovingAverage

    def __init__(self, recency_weight: float, **kwargs):
        """
        :param recency_weight: Recency weight to use when updating the
            recency weighted variance.

        Letting ``w`` be the recency weight,

        .. math::
            \\text{RWV}_w(x_t) = \\text{EMA}_w({x^2_t}) - \\text{EMA}_w(x_t)^2
        """
        super().__init__(**kwargs)
        self.recency_weight = recency_weight

    @property
    def recency_weight(self):
        return self._recency_weight

    @recency_weight.setter
    def recency_weight(self, weight: float):
        assert 0.0 < weight <= 1.0
        self._recency_weight = weight
        self.ex.recency_weight = weight
        self.ex2.recency_weight = weight

    def drop(self, x):
        """
        Recency Weighted Variance does not support dropping values
        """
        pass
