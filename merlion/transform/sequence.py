#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Classes to compose (`TransformSequence`) or stack (`TransformStack`) multiple transforms.
"""

from collections import OrderedDict
import logging
from typing import List

from merlion.transform.base import TransformBase, InvertibleTransformBase
from merlion.transform.factory import TransformFactory
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class TransformSequence(InvertibleTransformBase):
    """
    Applies a series of data transformations sequentially.
    """

    def __init__(self, transforms: List[TransformBase]):
        super().__init__()
        self.transforms = []
        for t in transforms:
            assert isinstance(
                t, (TransformBase, dict)
            ), f"Expected all transforms to be instances of TransformBase, or dict, but got {transforms}"
            if isinstance(t, dict):
                t = TransformFactory.create(**t)
            self.transforms.append(t)

    @property
    def proper_inversion(self):
        """
        A transform sequence is invertible if and only if all the transforms
        comprising it are invertible.
        """
        return all(f.proper_inversion for f in self.transforms)

    @property
    def requires_inversion_state(self):
        """
        ``False`` because inversion state is held by individual transforms.
        """
        return False

    def to_dict(self):
        return {"name": type(self).__name__, "transforms": [f.to_dict() for f in self.transforms]}

    def append(self, transform):
        assert isinstance(transform, TransformBase)
        self.transforms.append(transform)

    @classmethod
    def from_dict(cls, state):
        return cls([TransformFactory.create(**d) for d in state["transforms"]])

    def train(self, time_series: TimeSeries):
        for f in self.transforms:
            f.train(time_series)
            time_series = f(time_series)

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        for f in self.transforms:
            time_series = f(time_series)
        return time_series

    def invert(self, time_series: TimeSeries, retain_inversion_state=False) -> TimeSeries:
        for f in self.transforms[-1::-1]:
            time_series = f.invert(time_series, retain_inversion_state)
        return time_series

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        logger.warning(
            f"_invert() should not be called by a transform of type {type(self).__name__}. Applying the identity.",
            stack_info=True,
        )
        return time_series

    def __repr__(self):
        return "TransformSequence(\n " + ",\n ".join([repr(f) for f in self.transforms]) + "\n)"


class TransformStack(TransformSequence):
    """
    Applies a set of data transformations individually to an input time series.
    Stacks all of the results into a multivariate time series.
    """

    def __init__(self, transforms, *, check_aligned=True):
        super().__init__(transforms)
        self.check_aligned = check_aligned

    @property
    def proper_inversion(self):
        """
        A stacked transform is invertible if and only if at least one of the
        transforms comprising it are invertible.
        """
        return any(f.proper_inversion for f in self.transforms)

    @property
    def requires_inversion_state(self):
        """
        ``True`` because the inversion state tells us which stacked transform to
        invert, and which part of the output time series to apply that inverse
        to.
        """
        return True

    def train(self, time_series: TimeSeries):
        for f in self.transforms:
            f.train(time_series)

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        ts_list = [f(time_series) for f in self.transforms]

        # To invert the overall stacked transform, we pick one transform (idx)
        # to invert. The outputs of this transform are univariates d0 to df of
        # the output time series. We also need to keep track of the names of the
        # univariates in the input time series.
        if self.proper_inversion:
            idx = min(i for i, f in enumerate(self.transforms) if f.proper_inversion)
            d0 = sum(ts.dim for ts in ts_list[:idx])
            df = d0 + ts_list[idx].dim
            self.inversion_state = (idx, d0, df, time_series.names)
        else:
            self.inversion_state = (0, 0, ts_list[0].dim, time_series.names)

        return TimeSeries.from_ts_list(ts_list, check_aligned=self.check_aligned)

    def invert(self, time_series: TimeSeries, retain_inversion_state=False) -> TimeSeries:

        if self.inversion_state is None:
            raise RuntimeError(
                "Inversion state not set. Please call this transform on an "
                "input time series before calling invert(). If you are trying "
                "to call invert() a second time, please supply the option "
                "`retain_inversion_state=True` to the first call."
            )

        idx, d0, df, names = self.inversion_state
        ts = TimeSeries(OrderedDict((n, time_series.univariates[n]) for n in time_series.names[d0:df]))
        inverted = self.transforms[idx].invert(ts, retain_inversion_state)
        assert inverted.dim == len(names)
        inverted = TimeSeries(OrderedDict((name, var) for name, var in zip(names, inverted.univariates)))

        if not retain_inversion_state:
            self.inversion_state = None
        return inverted

    def __repr__(self):
        return "TransformStack(\n " + ",\n ".join([repr(f) for f in self.transforms]) + "\n)"
