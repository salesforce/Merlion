#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that clip the input.
"""

from collections import OrderedDict
import logging
import numpy as np

from merlion.transform.base import TransformBase
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


class LowerUpperClip(TransformBase):
    """
    Clips the values of a time series to lie between lower and upper.
    """

    def __init__(self, lower=None, upper=None):
        super().__init__()
        assert not (lower is None and upper is None), "Must provide at least one of lower or upper"
        if lower is not None and upper is not None:
            assert lower < upper
        self.lower = lower
        self.upper = upper

    @property
    def requires_inversion_state(self):
        """
        ``False`` because "inverting" value clipping is stateless.
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for name, var in time_series.items():
            x = np.clip(var.np_values, self.lower, self.upper)
            new_vars[name] = UnivariateTimeSeries(var.index, x)

        return TimeSeries(new_vars)
