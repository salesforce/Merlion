#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Code for resampling time series.
"""
from enum import Enum
from functools import partial
import logging
import math
import re
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset as pd_to_offset
import scipy.stats

logger = logging.getLogger(__name__)


class AlignPolicy(Enum):
    """Policies for aligning multiple univariate time series."""

    OuterJoin = 0
    InnerJoin = 1
    FixedReference = 2
    FixedGranularity = 3


class AggregationPolicy(Enum):
    """
    Aggregation policies. Values are partial functions for
    pandas.core.resample.Resampler methods.
    """

    Mean = partial(lambda df, *args, **kwargs: getattr(df, "mean")(*args, **kwargs))
    Sum = partial(lambda df, *args, **kwargs: getattr(df, "sum")(*args, **kwargs))
    Median = partial(lambda df, *args, **kwargs: getattr(df, "median")(*args, **kwargs))
    First = partial(lambda df, *args, **kwargs: getattr(df, "first")(*args, **kwargs))
    Last = partial(lambda df, *args, **kwargs: getattr(df, "last")(*args, **kwargs))
    Min = partial(lambda df, *args, **kwargs: getattr(df, "min")(*args, **kwargs))
    Max = partial(lambda df, *args, **kwargs: getattr(df, "max")(*args, **kwargs))


class MissingValuePolicy(Enum):
    """
    Missing value imputation policies. Values are partial functions for ``pd.Series`` methods.
    """

    FFill = partial(lambda df, *args, **kwargs: getattr(df, "ffill")(*args, **kwargs))
    """Fill gap with the first value before the gap."""
    BFill = partial(lambda df, *args, **kwargs: getattr(df, "bfill")(*args, **kwargs))
    """Fill gap with the first value after the gap."""
    Nearest = partial(lambda df, *args, **kwargs: getattr(df, "interpolate")(*args, **kwargs), method="nearest")
    """Replace missing value with the value closest to it."""
    Interpolate = partial(lambda df, *args, **kwargs: getattr(df, "interpolate")(*args, **kwargs), method="time")
    """Fill in missing values by linear interpolation."""
    ZFill = partial(lambda df, *args, **kwargs: getattr(df, "replace")(*args, **kwargs), to_replace=np.nan, value=0)
    """Replace missing values with zeros."""


def to_pd_datetime(timestamp):
    """
    Converts a timestamp (or list/iterable of timestamps) to pandas Datetime, truncated at the millisecond.
    """
    if isinstance(timestamp, pd.DatetimeIndex):
        return timestamp
    elif isinstance(timestamp, (int, float)):
        return pd.to_datetime(int(timestamp * 1000), unit="ms")
    elif isinstance(timestamp, Iterable) and all(isinstance(t, (int, float)) for t in timestamp):
        timestamp = pd.to_datetime(np.asarray(timestamp).astype(float) * 1000, unit="ms")
    elif isinstance(timestamp, np.ndarray) and timestamp.dtype in [int, np.float32, np.float64]:
        timestamp = pd.to_datetime(np.asarray(timestamp).astype(float) * 1000, unit="ms")
    return pd.to_datetime(timestamp)


def to_offset(dt):
    """
    Converts a time gap to a ``pd.Timedelta`` if possible, otherwise a ``pd.DateOffset``.
    """
    if dt is None:
        return None
    if isinstance(dt, (int, float)):
        dt = pd.to_timedelta(dt, unit="s")
    elif isinstance(dt, str) and not any(re.match(r"\d+" + suf, dt) for suf in ("M", "m", "MS", "Q", "Y", "y")):
        try:
            dt = pd.to_timedelta(dt)
        except ValueError:
            pass
    return dt if isinstance(dt, pd.Timedelta) else pd_to_offset(dt)


def to_timestamp(t):
    """
    Converts a datetime to a Unix timestamp.
    """
    if isinstance(t, (int, float)) or isinstance(t, Iterable) and all(isinstance(ti, (int, float)) for ti in t):
        return np.asarray(t)
    elif isinstance(t, np.ndarray) and t.dtype in [int, np.float32, np.float64]:
        return t
    return np.asarray(t).astype("datetime64[ms]").astype(float) / 1000


def granularity_str_to_seconds(granularity: Union[str, float, int, None]) -> Union[float, None]:
    """
    Converts a string/float/int granularity (representing a timedelta) to the
    number of seconds it represents, truncated at the millisecond.
    """
    if granularity is None:
        return None
    if isinstance(granularity, (float, int)):
        ms = np.floor(granularity * 1000)
    else:
        ms = np.floor(pd_to_offset(granularity).nanos / 1e6)
    return (ms / 1000).item()


def get_date_offset(time_stamps: pd.DatetimeIndex, reference: pd.DatetimeIndex) -> pd.DateOffset:
    """
    Returns the date offset one must add to ``time_stamps`` so its last timestamp aligns with that of ``reference``.
    """
    df, tf = time_stamps[-1], reference[-1]
    return pd.DateOffset(
        months=tf.month - df.month + 12 * (tf.year - df.year),
        days=tf.day - df.day,
        hours=tf.hour - df.hour,
        minutes=tf.minute - df.minute,
        seconds=tf.second - df.second,
        microseconds=tf.microsecond - df.microsecond,
    )


def infer_granularity(time_stamps, return_offset=False):
    """
    Infers the granularity of a list of time stamps.
    """
    # See if pandas can infer the granularity on its own
    orig_t = to_pd_datetime(time_stamps)
    if len(orig_t) > 2:
        freq = pd.infer_freq(orig_t)
    elif len(orig_t) == 2:
        freq = orig_t[1] - orig_t[0]
    else:
        raise ValueError("Need at least 2 timestamps to infer a granularity.")
    offset = pd.to_timedelta(0)
    if freq is not None:
        freq = pd_to_offset(freq)
        return (freq, offset) if return_offset else freq

    # Otherwise, start with the most commonly occurring timedelta
    dt = pd.to_timedelta(min(scipy.stats.mode(orig_t[1:] - orig_t[:-1], axis=None)[0]))

    # Check if the data could be sampled at a k-monthly granularity.
    candidate_freqs = [dt]
    for k in range(math.ceil(dt / pd.Timedelta(days=31)), math.ceil(dt / pd.Timedelta(days=28))):
        candidate_freqs.extend([pd_to_offset(f"{k}MS"), pd_to_offset(f"{k}M")])

    # Pick the sampling frequency which has the most overlap with the actual timestamps
    freq2idx = {f: pd.date_range(start=orig_t[0], end=orig_t[-1], freq=f) for f in candidate_freqs}
    freq2offset = {f: get_date_offset(time_stamps=freq2idx[f], reference=orig_t) for f in candidate_freqs}
    freq = sorted(freq2idx.keys(), key=lambda f: len((freq2idx[f] + freq2offset[f]).intersection(orig_t)))[-1]
    return (freq, freq2offset[freq]) if return_offset else freq


def reindex_df(
    df: Union[pd.Series, pd.DataFrame], reference: Sequence[Union[int, float]], missing_value_policy: MissingValuePolicy
):
    """
    Reindexes a Datetime-indexed dataframe ``df`` to have the same time stamps
    as a reference sequence of timestamps. Imputes missing values with the given
    `MissingValuePolicy`.
    """
    reference = to_pd_datetime(reference)
    all_times = np.unique(np.concatenate((reference.values, df.index.values)))
    df = df.reindex(index=all_times)
    df = missing_value_policy.value(df).ffill().bfill()
    return df.loc[reference]
