#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that resample the input in time, or stack adjacent observations
into vectors.
"""

from collections import OrderedDict
import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from merlion.transform.base import TransformBase, InvertibleTransformBase
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.utils.resample import (
    granularity_str_to_seconds,
    get_gcd_timedelta,
    to_pd_datetime,
    AlignPolicy,
    AggregationPolicy,
    MissingValuePolicy,
)

logger = logging.getLogger(__name__)


class TemporalResample(TransformBase):
    """
    Defines a policy to temporally resample a time series at a specified
    granularity. Note that while this transform does support inversion, the
    recovered time series may differ from the input due to information loss
    when downsampling.
    """

    def __init__(
        self,
        granularity: Union[str, int, float] = None,
        origin: int = None,
        trainable_granularity: bool = None,
        remove_non_overlapping=True,
        aggregation_policy: Union[str, AggregationPolicy] = "Mean",
        missing_value_policy: Union[str, MissingValuePolicy] = "Interpolate",
    ):
        """
        Defines a policy to temporally resample a time series.

        :param granularity: The granularity at which we want to resample.
        :param origin: The time stamp defining the offset to start at.
        :param trainable_granularity: Whether the granularity is trainable,
            i.e. train() will set it to the GCD timedelta of a time series.
            If ``None`` (default), it will be trainable only if no granularity is
            explicitly given.
        :param remove_non_overlapping: If ``True``, we will only keep the portions
            of the univariates that overlap with each other. For example, if we
            have 3 univariates which span timestamps [0, 3600], [60, 3660], and
            [30, 3540], we will only keep timestamps in the range [60, 3540]. If
            ``False``, we will keep all timestamps produced by the resampling.
        :param aggregation_policy: The policy we will use to aggregate multiple
            values in a window (downsampling).
        :param missing_value_policy: The policy we will use to impute missing
            values (upsampling).
        """
        super().__init__()
        if not isinstance(granularity, (int, float)):
            try:
                granularity = granularity_str_to_seconds(granularity)
            except:
                pass
        self.granularity = granularity
        self.origin = origin
        if trainable_granularity is None:
            trainable_granularity = granularity is None
        self.trainable_granularity = trainable_granularity
        self.remove_non_overlapping = remove_non_overlapping
        self.aggregation_policy = aggregation_policy
        self.missing_value_policy = missing_value_policy

    @property
    def requires_inversion_state(self):
        return False

    @property
    def granularity(self):
        return self._granularity

    @granularity.setter
    def granularity(self, granularity):
        if not isinstance(granularity, (int, float)):
            try:
                granularity = granularity_str_to_seconds(granularity)
            except:
                pass
        self._granularity = granularity

    @property
    def aggregation_policy(self) -> AggregationPolicy:
        return self._aggregation_policy

    @aggregation_policy.setter
    def aggregation_policy(self, agg: Union[str, AggregationPolicy]):
        if isinstance(agg, str):
            valid = set(AggregationPolicy.__members__.keys())
            if agg not in valid:
                raise KeyError(f"{agg} is not a valid aggregation policy. Valid aggregation policies are: {valid}")
            agg = AggregationPolicy[agg]
        self._aggregation_policy = agg

    @property
    def missing_value_policy(self) -> MissingValuePolicy:
        return self._missing_value_policy

    @missing_value_policy.setter
    def missing_value_policy(self, mv: Union[str, MissingValuePolicy]):
        if isinstance(mv, str):
            valid = set(MissingValuePolicy.__members__.keys())
            if mv not in valid:
                raise KeyError(f"{mv} is not a valid missing value policy. Valid aggregation policies are: {valid}")
            mv = MissingValuePolicy[mv]
        self._missing_value_policy = mv

    def train(self, time_series: TimeSeries):
        if self.trainable_granularity:
            time_stamps = time_series.time_stamps
            freq = pd.infer_freq(to_pd_datetime(time_stamps))
            if freq is not None:
                try:
                    self.granularity = pd.to_timedelta(to_offset(freq)).total_seconds()
                except:
                    self.granularity = freq
            else:
                self.granularity = get_gcd_timedelta(time_stamps)
                logger.warning(f"Inferred granularity {pd.to_timedelta(self.granularity, unit='s')}")

        if self.trainable_granularity or self.origin is None:
            t0, tf = time_series.t0, time_series.tf
            if isinstance(self.granularity, (int, float)):
                offset = (tf - t0) % self.granularity
            else:
                offset = 0
            self.origin = t0 + offset

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        if self.granularity is None:
            logger.warning(
                f"Skipping resampling step because granularity is "
                f"None. Please either specify a granularity or train "
                f"this transformation on a time series."
            )
            return time_series

        return time_series.align(
            alignment_policy=AlignPolicy.FixedGranularity,
            granularity=self.granularity,
            origin=self.origin,
            remove_non_overlapping=self.remove_non_overlapping,
            aggregation_policy=self.aggregation_policy,
            missing_value_policy=self.missing_value_policy,
        )


class Shingle(InvertibleTransformBase):
    """
    Stacks adjacent observations into a single vector. Downsamples by the
    specified stride (less than or equal to the shingle size) if desired.

    More concretely, consider an input time series,

    .. code-block:: python

        TimeSeries(
            UnivariateTimeSeries((t1[0], x1[0]), ..., (t1[m], t1[m])),
            UnivariateTimeSeries((t2[0], x2[0]), ..., (t2[m], t2[m])),
        )

    Applying a shingle of size 3 and stride 2 will yield

    .. code-block:: python

        TimeSeries(
            UnivariateTimeSeries((t1[0], x1[0]), (t1[2], x1[2]), ..., (t1[m-2], x1[m-2])),
            UnivariateTimeSeries((t1[1], x1[1]), (t1[3], x1[3]), ..., (t1[m-1], x1[m-1])),
            UnivariateTimeSeries((t1[2], x1[2]), (t1[4], x1[4]), ..., (t1[m],   x1[m])),

            UnivariateTimeSeries((t2[0], x2[0]), (t2[2], x2[2]), ..., (t2[m-2], x2[m-2])),
            UnivariateTimeSeries((t2[1], x2[1]), (t2[3], x2[3]), ..., (t2[m-1], x2[m-1])),
            UnivariateTimeSeries((t2[2], x2[2]), (t2[4], x2[4]), ..., (t2[m],   x2[m])),
        )

    If the length of any univariate is not perfectly divisible by the stride, we
    will pad it on the left side with the first value in the univariate.
    """

    def __init__(self, size: int = 1, stride: int = 1, multivar_skip=True):
        """
        Converts the time series into shingle vectors of the appropriate size.
        This converts each univariate into a multivariate time series with
        ``size`` variables.

        :param size: let x(t) = value_t be the value of the time series at
            time index t. Then, the output vector for time index t will be
            :code:`[x(t - size + 1), ..., x(t - 1), x(t)]`.
        :param stride: The stride at which the output vectors are downsampled.
        :param multivar_skip: Whether to skip this transform if the transform
            is already multivariate.
        """
        super().__init__()
        assert size >= 0
        assert 1 <= stride <= size
        self.stride = stride
        self.size = size
        self.multivar_skip = multivar_skip

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        if self.multivar_skip and time_series.dim > 1:
            self.inversion_state = "skip"
            return time_series

        new_vars = OrderedDict()

        for name, var in time_series.items():
            # Left-pad the time series with the first value
            x0 = var.np_values[0]
            vals = np.concatenate((np.full(self.size - 1, x0), var.np_values))

            # Stack adjacent observations into vectors of length self.size,
            # and apply any striding desired
            i0 = (len(var) - 1) % self.stride
            times = var.index[i0 :: self.stride]
            all_vals = np.stack([vals[i : len(vals) - self.size + i + 1] for i in range(self.size)])
            all_vals = all_vals[:, i0 :: self.stride]

            # Convert the stacked values into UnivariateTimeSeries objects
            new_vars.update(
                OrderedDict([(f"{name}_{i}", UnivariateTimeSeries(times, x)) for i, x in enumerate(all_vals)])
            )

        # The inversion state is just the timestamps of the univariates before
        # shingling occurs, and the name of the original univariate
        self.inversion_state = [(name, v.index) for name, v in time_series.items()]
        return TimeSeries(new_vars)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        if self.inversion_state == "skip":
            return time_series

        new_vars = OrderedDict()

        for i, (name, time_stamps) in enumerate(self.inversion_state):
            vals = []
            expected_src_names = [f"{name}_{i}" for i in range(self.size)]
            src_names = time_series.names[i * self.size : (i + 1) * self.size]
            src = TimeSeries(OrderedDict([(k, time_series.univariates[k]) for k in src_names]))
            assert src.is_aligned and src.dim == self.size, (
                f"{self} should convert a univariate time series into an "
                f"aligned multivariate time series of dim {self.size}, but "
                f"something went wrong."
            )
            assert (
                src.names == expected_src_names
            ), f"Expected univariates named {expected_src_names}, but got {src.names}"

            for j, (t, val_vec) in enumerate(src[::-1]):
                j0 = j * self.stride
                val_vec = val_vec[::-1]
                vals.extend(val_vec[len(vals) - j0 :])

            vals = vals[len(time_stamps) :: -1][-len(time_stamps) :]
            new_vars[name] = UnivariateTimeSeries(time_stamps, vals)

        return TimeSeries(new_vars)
