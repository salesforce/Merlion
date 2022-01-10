#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from enum import Enum
from functools import partial
import logging
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
import pandas.tseries.frequencies

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
    Missing value imputation policies. Values are partial functions for
    ``pd.Series`` methods.
    """

    FFill = partial(lambda df, *args, **kwargs: getattr(df, "ffill")(*args, **kwargs))
    """Fill gap with the first value before the gap."""
    BFill = partial(lambda df, *args, **kwargs: getattr(df, "bfill")(*args, **kwargs))
    """Fill gap with the first value after the gap."""
    Nearest = partial(lambda df, *args, **kwargs: getattr(df, "interpolate")(*args, **kwargs), method="nearest")
    """Replace missing value with the value closest to it."""
    Interpolate = partial(lambda df, *args, **kwargs: getattr(df, "interpolate")(*args, **kwargs), method="time")
    """Fill in missing values by linear interpolation."""


def to_pd_datetime(timestamp):
    """
    Converts a timestamp (or list/iterable of timestamps) to pandas Datetime,
    truncated at the millisecond.
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


def to_timestamp(t):
    """
    Converts a datetime to a Unix timestamp.
    """
    if isinstance(t, (int, float)) or isinstance(t, Iterable) and all(isinstance(ti, (int, float)) for ti in t):
        return t
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
        try:
            granularity = pd.Timedelta(granularity)
        except:
            pass
        ms = np.floor(pd.tseries.frequencies.to_offset(granularity).nanos / 1e6)
    return ms / 1000


def get_gcd_timedelta(*time_stamp_lists):
    """
    Calculates all timedeltas present in any of the lists of time stamps given,
    and returns the GCD of all these timedeltas (up to units of milliseconds).
    """
    all_dt = np.concatenate([np.diff(t) for t in time_stamp_lists])
    all_dt = np.unique((all_dt * 1000).astype(np.int64))
    gcd_dt = all_dt[0]
    for dt in all_dt[1:]:
        gcd_dt = np.gcd(gcd_dt, dt)
    return gcd_dt.astype(float) / 1000


def infer_granularity(time_stamps):
    """
    Infers the granularity of a list of time stamps
    """
    freq = pd.infer_freq(to_pd_datetime(time_stamps))
    if freq is not None:
        granularity = pd.tseries.frequencies.to_offset(freq)
    else:
        granularity = pd.to_timedelta(get_gcd_timedelta(time_stamps), unit="s")
        logger.warning(f"Inferred granularity {granularity}")
    return granularity


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
