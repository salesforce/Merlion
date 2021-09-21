#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transform base classes and the `Identity` transform.
"""

from abc import abstractmethod
from copy import deepcopy
from enum import Enum
import inspect
import logging

from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


class TransformBase(metaclass=AutodocABCMeta):
    """
    Abstract class for a callable data pre-processing transform.

    Subclasses must override the ``train`` method (:code:`pass` if
    no training is required) and ``__call__`` method (to implement
    the actual transform).

    Subclasses may also support a pseudo inverse transform (possibly using the
    implementation-specific ``self.inversion_state``, which should be set
    in ``__call__``). If an inversion state is not required, override the
    property `requires_inversion_state` to return ``False``.

    Due to possible information loss in the forward pass, the inverse transform
    may be not be perfect/proper, and calling `TransformBase.invert` will result
    in a warning. By default, the inverse transform (implemented in
    `TransformBase._invert`) is just the identity.

    :ivar inversion_state: Implementation-specific intermediate state that is
        used to compute the inverse transform for a particular time series. Only
        used if `TransformBase.requires_inversion_state` is ``True``. The
        inversion state is destroyed upon calling `TransformBase.invert`,
        unless the option the option ``retain_inversion_state=True`` is
        specified. This is to prevent potential user error.

    .. document private members
    .. automethod:: _invert
    """

    def __init__(self):
        self.inversion_state = None

    @property
    def proper_inversion(self):
        """
        `TransformBase` objects do not support a proper inversion.
        """
        return False

    @property
    def requires_inversion_state(self):
        """
        Indicates whether any state ``self.inversion_state`` is required to
        invert the transform. Specific to each transform. ``True`` by default.
        """
        return True

    def to_dict(self):
        state = {"name": type(self).__name__}
        for k in inspect.signature(self.__init__).parameters:
            v = getattr(self, k)
            state[k] = v.name if isinstance(v, Enum) else deepcopy(v)
        return state

    @classmethod
    def from_dict(cls, state: dict):
        return cls(**state)

    def __getstate__(self):
        return {k: v for k, v in self.to_dict().items() if k != "name"}

    def __setstate__(self, state):
        self.__init__(**state)

    @abstractmethod
    def train(self, time_series: TimeSeries):
        """
        Sets all trainable parameters of the transform (if any), using the input
        time series as training data.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        raise NotImplementedError

    def invert(self, time_series: TimeSeries, retain_inversion_state=False) -> TimeSeries:
        """
        Applies the inverse of this transform on the time series.

        :param time_series: The time series on which to apply the inverse
            transform.
        :param retain_inversion_state: If an inversion state is required, supply
            ``retain_inversion_state=True`` to retain the inversion state
            even after calling this method. Otherwise, the inversion state will
            be set to ``None`` after the inversion is applied, to prevent a user
            error of accidentally using a stale state.

        :return: The (inverse) transformed time series.
        """
        if not self.proper_inversion:
            logger.info(
                f"Transform {self} is not strictly invertible. "
                f"Calling invert() is not guaranteed to recover the "
                f"original time series exactly!"
            )

        if self.requires_inversion_state and self.inversion_state is None:
            raise RuntimeError(
                "Inversion state not set. Please call this transform on an "
                "input time series before calling invert(). If you are trying "
                "to call invert() a second time, please supply the option "
                "`retain_inversion_state=True` to the first call."
            )

        inverted = self._invert(time_series)
        if not retain_inversion_state:
            self.inversion_state = None
        return inverted

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        """
        Helper method which actually performs the inverse transform
        (when possible).

        :param time_series: Time series to apply the inverse transform to
        :return: The (inverse) transformed time series.
        """
        return time_series

    def __repr__(self):
        kwargs = self.to_dict()
        name = kwargs.pop("name")
        kwargs_str = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{name}({kwargs_str})"


class InvertibleTransformBase(TransformBase):
    """
    Abstract class for a callable data pre-processing transform with a proper
    inverse.

    In addition to overriding the ``train`` and ``__call__`` methods, subclasses
    *must* also override the `InvertibleTransformBase._invert` method to
    implement the actual inverse transform.

    :ivar inversion_state: Implementation-specific intermediate state that is
        used to compute the inverse transform for a particular time series. Only
        used if `TransformBase.requires_inversion_state` is ``True``. The
        inversion state is destroyed upon calling `TransformBase.invert`,
        unless the option the option ``retain_inversion_state=True`` is
        specified. This is to prevent potential user error.

    .. document private members
    .. automethod:: _invert
    """

    @property
    def proper_inversion(self):
        """
        `InvertibleTransformBase` always supports a proper inversion.
        """
        return True

    @abstractmethod
    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        raise NotImplementedError


class Identity(InvertibleTransformBase):
    """
    The identity transformation. Does nothing.
    """

    def __init__(self):
        super().__init__()

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the identity operation is stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        return time_series

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        return time_series
