#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for post-processing rules in Merlion.
"""
from abc import abstractmethod
from copy import copy, deepcopy
import inspect

from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta


class PostRuleBase(metaclass=AutodocABCMeta):
    """
    Base class for post-processing rules in Merlion. These objects are primarily
    for post-processing the sequence of anomaly scores returned by anomaly detection
    models. All post-rules are callable objects, and they have a ``train()`` method
    which may accept additional implementation-specific keyword arguments.
    """

    def to_dict(self):
        params = inspect.signature(self.__init__).parameters
        d = {k: deepcopy(getattr(self, k)) for k in params}
        d["name"] = type(self).__name__
        return d

    @classmethod
    def from_dict(cls, state_dict):
        state_dict = copy(state_dict)
        state_dict.pop("name", None)
        return cls(**state_dict)

    def __copy__(self):
        return self.from_dict(self.to_dict())

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __repr__(self):
        kwargs = self.to_dict()
        name = kwargs.pop("name")
        kwargs_str = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{name}({kwargs_str})"

    @abstractmethod
    def train(self, anomaly_scores: TimeSeries):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, anomaly_scores: TimeSeries):
        raise NotImplementedError
