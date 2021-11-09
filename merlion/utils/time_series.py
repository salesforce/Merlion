#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from bisect import bisect_left, bisect_right
import itertools
import logging
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from merlion.utils.misc import ValIterOrderedDict
from merlion.utils.resample import (
    AggregationPolicy,
    AlignPolicy,
    MissingValuePolicy,
    infer_granularity,
    reindex_df,
    to_pd_datetime,
    to_timestamp,
)

logger = logging.getLogger(__name__)


class UnivariateTimeSeries(pd.Series):
    """
    Please read the `tutorial <examples/TimeSeries>` before reading this API doc.
    This class is a time-indexed ``pd.Series`` which represents a univariate
    time series. For the most part, it supports all the same features as
    ``pd.Series``, with the following key differences to iteration and indexing:

    1.  Iterating over a `UnivariateTimeSeries` is implemented as

        .. code-block:: python

            for timestamp, value in univariate:
                # do stuff...

        where ``timestamp`` is a Unix timestamp, and ``value`` is the
        corresponding time series value.
    2.  Integer index: ``u[i]`` yields the tuple ``(u.time_stamps[i], u.values[i])``
    3.  Slice index: ``u[i:j:k]`` yields a new
        ``UnivariateTimeSeries(u.time_stamps[i:j:k], u.values[i:j:k])``

    The class also supports the following additional features:

    1.  ``univariate.time_stamps`` returns the list of Unix timestamps, and
        ``univariate.values`` returns the list of the time series values. You
        may access the ``pd.DatetimeIndex`` directly with ``univariate.index``
        (or its ``np.ndarray`` representation with ``univariate.np_time_stamps``),
        and the ``np.ndarray`` of values with ``univariate.np_values``.
    2.  ``univariate.concat(other)`` will concatenate the UnivariateTimeSeries
        ``other`` to the right end of ``univariate``.
    3.  ``left, right = univariate.bisect(t)`` will split the univariate at the
        given timestamp ``t``.
    4.  ``window = univariate.window(t0, tf)`` will return the subset of the time
        series occurring between timestamps ``t0`` (inclusive) and ``tf``
        (non-inclusive)
    5.  ``series = univariate.to_pd()`` will convert the `UnivariateTimeSeries`
        into a regular ``pd.Series`` (for compatibility).
    6.  ``univariate = UnivariateTimeSeries.from_pd(series)`` uses a time-indexed
        ``pd.Series`` to create a `UnivariateTimeSeries` object directly.

    .. document special functions
    .. automethod:: __getitem__
    .. automethod:: __iter__
    """

    def __init__(
        self,
        time_stamps: Union[None, Sequence[Union[int, float]]],
        values: Sequence[float],
        name: str = None,
        freq="1h",
    ):
        """
        :param time_stamps: a sequence of Unix timestamps. You may specify
            ``None`` if you only have ``values`` with no specific time stamps.
        :param values: a sequence of univariate values, where ``values[i]``
            occurs at time ``time_stamps[i]``
        :param name: the name of the univariate time series
        :param freq: if ``time_stamps`` is not provided, the univariate is
            assumed to be sampled at frequency ``freq``. ``freq`` may be a
            string (e.g. ``"1h"``), timedelta, or ``int``/``float`` (in units
            of seconds).
        """
        is_pd = isinstance(values, pd.Series)
        if name is None and is_pd:
            name = values.name
        if is_pd and isinstance(values.index, pd.DatetimeIndex):
            super().__init__(values, name=name)
        else:
            if time_stamps is None:
                if isinstance(freq, (int, float)):
                    freq = pd.to_timedelta(freq, unit="s")
                else:
                    freq = to_offset(freq)
                if is_pd and values.index.dtype in ("int64", "float64"):
                    index = values.index.values * freq + np.full(len(values), pd.to_datetime(0))
                else:
                    index = pd.date_range(start=0, periods=len(values), freq=freq)
            else:
                index = to_pd_datetime(time_stamps)
            super().__init__(np.asarray(values), index=index, name=name, dtype=float)
            if len(self) >= 3 and self.index.freq is None:
                self.index.freq = pd.infer_freq(self.index)

    @property
    def np_time_stamps(self):
        """
        :rtype: np.ndarray
        :return: the ``numpy`` representation of this time series's Unix timestamps
        """
        return to_timestamp(self.index.values)

    @property
    def np_values(self):
        """
        :rtype: np.ndarray
        :return: the ``numpy`` representation of this time series's values
        """
        return super().values

    @property
    def time_stamps(self):
        """
        :rtype: List[float]
        :return: the list of Unix timestamps for the time series
        """
        return self.np_time_stamps.tolist()

    @property
    def values(self):
        """
        :rtype: List[float]
        :return: the list of values for the time series.
        """
        return self.np_values.tolist()

    @property
    def t0(self):
        """
        :rtype: float
        :return: the first timestamp in the univariate time series.
        """
        return self.np_time_stamps[0]

    @property
    def tf(self):
        """
        :rtype: float
        :return: the final timestamp in the univariate time series.
        """
        return self.np_time_stamps[-1]

    def is_empty(self):
        """
        :rtype: bool
        :return: True if the univariate is empty, False if not.
        """
        return len(self) == 0

    def __iter__(self):
        """
        The i'th item in the iterator is the tuple
        ``(self.time_stamps[i], self.values[i])``.
        """
        return itertools.starmap(lambda t, x: (t.item(), x.item()), zip(self.np_time_stamps, self.np_values))

    def __getitem__(self, i: Union[int, slice]):
        """
        :param i: integer index or slice

        :rtype: Union[Tuple[float, float], UnivariateTimeSeries]
        :return: ``(self.time_stamps[i], self.values[i])`` if ``i`` is
            an integer. ``UnivariateTimeSeries(self.time_series[i], self.values[i])``
            if ``i`` is a slice.
        """
        if isinstance(i, int):
            return self.np_time_stamps[i].item(), self.np_values[i].item()
        elif isinstance(i, slice):
            return UnivariateTimeSeries.from_pd(self.iloc[i])
        else:
            raise KeyError(
                f"Indexing a `UnivariateTimeSeries` with key {i} of "
                f"type {type(i).__name__} is not supported. Try "
                f"using loc[] or iloc[] for more complicated "
                f"indexing."
            )

    def __eq__(self, other):
        return self.time_stamps == other.time_stamps and (self.np_values == other.np_values).all()

    def copy(self, deep=True):
        """
        Copies the `UnivariateTimeSeries`. Simply a wrapper around the
        ``pd.Series.copy()`` method.
        """
        return UnivariateTimeSeries.from_pd(super().copy(deep=deep))

    def concat(self, other):
        """
        Concatenates the `UnivariateTimeSeries` ``other`` to the right of this one.
        :param UnivariateTimeSeries other: another `UnivariateTimeSeries`
        :rtype: UnivariateTimeSeries
        :return: concatenated univariate time series
        """
        return UnivariateTimeSeries.from_pd(pd.concat((self, other)), name=self.name)

    def bisect(self, t: float, t_in_left: bool = False):
        """
        Splits the time series at the point where the given timestamp occurs.

        :param t: a Unix timestamp or datetime object. Everything before time
            ``t`` is in the left split, and everything after time ``t`` is in
            the right split.
        :param t_in_left: if ``True``, ``t`` is in the left split. Otherwise,
            ``t`` is in the right split.

        :rtype: Tuple[UnivariateTimeSeries, UnivariateTimeSeries]
        :return: the left and right splits of the time series.
        """
        t = to_pd_datetime(t)
        if t_in_left:
            i = bisect_right(self.index, t)
        else:
            i = bisect_left(self.index, t)
        return self[:i], self[i:]

    def window(self, t0: float, tf: float, include_tf: bool = False):
        """
        :param t0: The timestamp/datetime at the start of the window (inclusive)
        :param tf: The timestamp/datetime at the end of the window (inclusive
            if ``include_tf`` is ``True``, non-inclusive otherwise)
        :param include_tf: Whether to include ``tf`` in the window.

        :rtype: UnivariateTimeSeries
        :return: The subset of the time series occurring between timestamps
            ``t0`` (inclusive) and ``tf`` (included if ``include_tf`` is
            ``True``, excluded otherwise).
        """
        times = self.index
        t0, tf = to_pd_datetime(t0), to_pd_datetime(tf)
        i_0 = bisect_left(times, t0)
        i_f = bisect_right(times, tf) if include_tf else bisect_left(times, tf)
        return self[i_0:i_f]

    def to_dict(self) -> Dict[float, float]:
        """
        :return: A dictionary representing the data points in the time series.
        """
        return dict(zip(self.time_stamps, self.values))

    @classmethod
    def from_dict(cls, obj: Dict[float, float], name=None):
        """
        :param obj: A dictionary of timestamp - value pairs
        :param name: the name to assign the output

        :rtype: UnivariateTimeSeries
        :return: the `UnivariateTimeSeries` represented by series.
        """
        time_stamps, values = [], []
        for point in sorted(obj.items(), key=lambda p: p[0]):
            time_stamps.append(point[0])
            values.append(point[1])
        return cls(time_stamps, values, name)

    def to_pd(self) -> pd.Series:
        """
        :return: A pandas Series representing the time series, indexed by time.
        """
        return pd.Series(self.np_values, index=self.index, name=self.name)

    @classmethod
    def from_pd(cls, series: pd.Series, name=None, freq="1h"):
        """
        :param series: a ``pd.Series``. If it has a``pd.DatetimeIndex``, we will
            use that index for the timestamps. Otherwise, we will create one at
            the specified frequency.
        :param name: the name to assign the output
        :param freq: if ``series`` is not indexed by time, this is the frequency
            at which we will assume it is sampled.

        :rtype: UnivariateTimeSeries
        :return: the `UnivariateTimeSeries` represented by series.
        """
        return cls(time_stamps=None, values=series.astype(float), name=name, freq=freq)

    def to_ts(self):
        """
        :rtype: TimeSeries
        :return: A `TimeSeries` representing this univariate time series.
        """
        if self.name is None:
            return TimeSeries([self])
        else:
            return TimeSeries({self.name: self})

    @classmethod
    def empty(cls, name=None):
        """
        :rtype: `UnivariateTimeSeries`
        :return: A Merlion `UnivariateTimeSeries` that has empty timestamps and values.
        """
        return cls([], [], name)


class TimeSeries:
    """
    Please read the `tutorial <examples/TimeSeries>` before reading this API doc.
    This class represents a general multivariate time series as a wrapper around
    a number of (optionally named) `UnivariateTimeSeries`. A `TimeSeries` object
    is initialized as ``time_series = TimeSeries(univariates)``, where
    ``univariates`` is either a list of `UnivariateTimeSeries`, or a dictionary
    mapping string names to their corresponding `UnivariateTimeSeries` objects.

    Because the individual ``univariates`` need not be sampled at the same times, an
    important concept for `TimeSeries` is *alignment*. We say that a `TimeSeries`
    is *aligned* if all of its univariates have observations sampled at the exact
    set set of times.

    One may access the `UnivariateTimeSeries` comprising this `TimeSeries` in
    four ways:

    1.  Iterate over the individual univariates using

        .. code-block:: python

            for var in time_series.univariates:
                # do stuff with each UnivariateTimeSeries var

    2.  Access an individual `UnivariateTimeSeries` by name as
        ``time_series.univariates[name]``. If you supplied unnamed univariates to
        the constructor (i.e. using a list), the name of a univariate will just
        be its index in that list.
    3.  Get the list of each univariate's name with ``time_series.names``.
    4.  Iterate over named univariates as

        .. code-block:: python

            for name, var in time_series.items():
                # do stuff

        Note that this is equivalent to iterating over
        ``zip(time_series.names, time_series.univariates)``.

    This class supports the following additional features as well:

    1.  Interoperability with ``pandas``

        -   ``df = time_series.to_pd()`` yields a time-indexed ``pd.DataFrame``,
            where each column (with the appropriate name) corresponds to a
            variable. Missing values are ``NaN``.
        -   ``time_series = TimeSeries.from_pd(df)`` takes a time-indexed
            ``pd.DataFrame`` and returns a corresponding `TimeSeries` object
            (missing values are handled appropriately). The order of
            ``time_series.univariates`` is the order of ``df.keys()``.

    2.  Automated alignment: ``aligned = time_series.align()`` resamples each of
        ``time_series.univariates`` so that they all have the same timestamps.
        By default, this is done by taking the union of all timestamps present
        in any individual univariate time series, and imputing missing values
        via interpolation. See the method documentation for details on how you
        may configure the alignment policy.
    3.  Transparent indexing and iteration for `TimeSeries` which have all
        univariates aligned (i.e. they all have the same timestamps)

        -   Get the length and shape of the time series (equal to the number of
            observations in each individual univariate). Note that if the time
            series is not aligned, we will return the length/shape of an equivalent
            ``pandas`` dataframe and emit a warning.
        -   Index ``time_series[i] = (times[i], (x1[i], ..., xn[i]))``
            (assuming ``time_series`` has ``n`` aligned univariates with timestamps
            ``times``, and ``xk = time_series.univariates[k-1].values``). Slice
            returns a `TimeSeries` object and works as one would expect.
        -   Assuming ``time_series`` has ``n`` variables, you may iterate with

            .. code-block:: python

                for t_i, (x1_i, ..., xn_i) in time_series:
                    # do stuff

            Notably, this lets you call ``times, val_vectors = zip(*time_series)``

    4.  Time-based queries for any time series

        -   Get the two sub `TimeSeries` before and after a timestamp ``t`` via
            ``left, right = time_series.bisect(t)``
        -   Get the sub `TimeSeries` between timestamps ``t0`` (inclusive) and
            ``tf`` (non-inclusive) via ``window = time_series.window(t0, tf)``

    5.  Concatenation: two `TimeSeries` may be concatenated (in time) as
        ``time_series = time_series_1 + time_series_2``.

    .. document special functions
    .. automethod:: __getitem__
    .. automethod:: __iter__
    """

    def __init__(
        self,
        univariates: Union[Mapping[Any, UnivariateTimeSeries], Iterable[UnivariateTimeSeries]],
        *,
        check_aligned=True,
    ):
        # Type/length checking of univariates
        if isinstance(univariates, Mapping):
            if not isinstance(univariates, ValIterOrderedDict):
                univariates = ValIterOrderedDict(univariates.items())
            assert all(isinstance(var, UnivariateTimeSeries) for var in univariates.values())
        elif isinstance(univariates, Iterable):
            univariates = list(univariates)
            assert all(isinstance(var, UnivariateTimeSeries) for var in univariates)

            names = [var.name for var in univariates]
            if len(set(names)) == len(names):
                names = [i if name is None else name for i, name in enumerate(names)]
                univariates = ValIterOrderedDict(zip(names, univariates))
            else:
                univariates = ValIterOrderedDict(enumerate(univariates))
        else:
            raise TypeError(
                "Expected univariates to be either a "
                "`Sequence[UnivariateTimeSeries]` or a "
                "`Mapping[Hashable, UnivariateTimeSeries]`."
            )
        assert len(univariates) > 0

        # Assign all the individual univariate series the appropriate names
        for name, var in univariates.items():
            var.name = name

        # Set self.univariates and check if they are perfectly aligned
        self.univariates = univariates
        if check_aligned and self.dim > 1:
            t = self.univariates[self.names[0]].time_stamps
            self._is_aligned = all(self.univariates[name].time_stamps == t for name in self.names[1:])
        else:
            self._is_aligned = len(univariates) <= 1

        # Raise a warning if the univariates are too mis-aligned
        if check_aligned and not self.is_aligned:
            all_t0 = [var.index[0] for var in univariates if len(var) > 0]
            all_tf = [var.index[-1] for var in univariates if len(var) > 0]
            min_elapsed = min(tf - t0 for t0, tf in zip(all_t0, all_tf))
            min_t0, max_t0 = min(all_t0), max(all_t0)
            min_tf, max_tf = min(all_tf), max(all_tf)
            if max_t0 - min_t0 > 0.1 * min_elapsed:
                logger.warning(
                    f"The earliest univariate starts at {min_t0}, but the "
                    f"latest univariate starts at {max_t0}, a difference of "
                    f"{max_t0 - min_t0}. This is more than 10% of the length "
                    f"of the shortest univariate ({min_elapsed}). You may "
                    f"want to check that the univariates cover the same "
                    f"window of time.",
                    stack_info=True,
                )
            if max_tf - min_tf > 0.1 * min_elapsed:
                logger.warning(
                    f"The earliest univariate ends at {min_tf}, but the "
                    f"latest univariate ends at {max_tf}, a difference of "
                    f"{max_tf - min_tf}. This is more than 10% of the length "
                    f"of the shortest univariate ({min_elapsed}). You may "
                    f"want to check that the univariates cover the same "
                    f"window of time.",
                    stack_info=True,
                )

    @property
    def names(self):
        """:return: The list of the names of the univariates."""
        return list(self.univariates.keys())

    def items(self):
        """:return: Iterator over ``(name, univariate)`` tuples."""
        return self.univariates.items()

    @property
    def dim(self) -> int:
        """
        :return: The dimension of the time series (the number of variables).
        """
        return len(self.univariates)

    @property
    def is_aligned(self) -> bool:
        """
        :return: Whether all individual variable time series are sampled at the
            same time stamps, i.e. they are aligned.
        """
        return self._is_aligned

    @property
    def np_time_stamps(self):
        """
        :rtype: np.ndarray
        :return: the ``numpy`` representation of this time series's Unix timestamps
        """
        return np.unique(np.concatenate([var.np_time_stamps for var in self.univariates]))

    @property
    def time_stamps(self):
        """
        :rtype: List[float]
        :return: the list of Unix timestamps for the time series
        """
        return self.np_time_stamps.tolist()

    @property
    def t0(self) -> float:
        """
        :rtype: float
        :return: the first timestamp in the time series.
        """
        return min(var.t0 for var in self.univariates)

    @property
    def tf(self) -> float:
        """
        :rtype: float
        :return: the final timestamp in the time series.
        """
        return max(var.tf for var in self.univariates)

    @staticmethod
    def _txs_to_vec(txs):
        """
        Helper function that converts [(t_1[i], x_1[i]), ..., (t_k[i], x_k[i])],
        i.e. [var[i] for var in self.univariates], into the desired output form
        (t_1[i], (x_1[i], ..., x_k[i])).
        """
        return txs[0][0], tuple(tx[1] for tx in txs)

    def __iter__(self):
        """
        Only supported if all individual variable time series are sampled at the
        same time stamps. The i'th item of the iterator is the tuple
        ``(time_stamps[i], tuple(var.values[i] for var in self.univariates))``.
        """
        if not self.is_aligned:
            raise RuntimeError(
                "The univariates comprising this time series are not aligned "
                "(they have different time stamps), but alignment is required "
                "to iterate over the time series."
            )

        return map(self._txs_to_vec, zip(*self.univariates))

    def __getitem__(self, i: Union[int, slice]):
        """
        Only supported if all individual variable time series are sampled at the
        same time stamps.

        :param i: integer index or slice.

        :rtype: Union[Tuple[float, Tuple[float]], TimeSeries]
        :return: If ``i`` is an integer, returns the tuple
            ``(time_stamps[i], tuple(var.values[i] for var in self.univariates))``.
            If ``i`` is a slice, returns the time series
            ``TimeSeries([var[i] for var in self.univariates])``
        """
        if not self.is_aligned:
            raise RuntimeError(
                "The univariates comprising this time series are not aligned "
                "(they have different time stamps), but alignment is required "
                "to index into the time series."
            )
        if isinstance(i, int):
            return self._txs_to_vec([var[i] for var in self.univariates])
        elif isinstance(i, slice):
            # ret must be aligned, so bypass the (potentially) expensive check
            univariates = ValIterOrderedDict([(k, v[i]) for k, v in self.items()])
            ret = TimeSeries(univariates, check_aligned=False)
            ret._is_aligned = True
            return ret
        else:
            raise KeyError(
                f"Indexing a `TimeSeries` with key {i} of type "
                f"{type(i).__name__} not supported. Perhaps you "
                f"meant to index into `time_series.univariates`, "
                f"rather than `time_series`?"
            )

    def is_empty(self) -> bool:
        """
        :return: whether the time series is empty
        """
        return all(len(var) == 0 for var in self.univariates)

    def squeeze(self) -> UnivariateTimeSeries:
        """
        :return: a `UnivariateTimeSeries` if the time series only
            has one univariate, otherwise returns itself, a `TimeSeries`
        """
        if self.dim == 1:
            return self.univariates[self.names[0]]
        return self

    def __len__(self):
        """
        :return: the number of observations in the time series
        """
        if not self.is_aligned:
            warning = (
                "The univariates comprising this time series are not aligned "
                "(they have different time stamps). The length returned is "
                "equal to the length of the _union_ of all time stamps present "
                "in any of the univariates."
            )
            warnings.warn(warning)
            logger.warning(warning)
            return len(self.to_pd())
        return len(self.univariates[self.names[0]])

    @property
    def shape(self) -> Tuple[int, int]:
        """
        :return: the shape of this time series, i.e. ``(self.dim, len(self))``
        """
        return self.dim, len(self)

    def __add__(self, other):
        """
        Concatenates the `TimeSeries` ``other`` to the right of this one.
        :param TimeSeries other:
        :rtype: TimeSeries
        :return: concatenated time series
        """
        assert (
            self.dim == other.dim
        ), f"Cannot concatenate a {self.dim}-dimensional time series with a {other.dim}-dimensional time series."
        assert self.names == other.names, (
            f"Cannot concatenate time series with two different sets of "
            f"variable names, {self.names} and {other.names}."
        )

        # The output time series is aligned if and only if both time series
        # are aligned, so skip the expensive check
        univariates = ValIterOrderedDict(
            [(name, ts0.concat(ts1)) for (name, ts0), ts1 in zip(self.items(), other.univariates)]
        )
        ret = TimeSeries(univariates, check_aligned=False)
        ret._is_aligned = self.is_aligned and other.is_aligned
        return ret

    def __eq__(self, other):
        if self.dim != other.dim:
            return False
        return all(u == v for u, v in zip(self.univariates, other.univariates))

    def __repr__(self):
        return repr(self.to_pd())

    def bisect(self, t: float, t_in_left: bool = False):
        """
        Splits the time series at the point where the given timestap ``t`` occurs.

        :param t: a Unix timestamp or datetime object. Everything before time
            ``t`` is in the left split, and everything after time ``t`` is in
            the right split.
        :param t_in_left: if ``True``, ``t`` is in the left split. Otherwise,
            ``t`` is in the right split.

        :rtype: Tuple[TimeSeries, TimeSeries]
        :return: the left and right splits of the time series.
        """
        left, right = ValIterOrderedDict(), ValIterOrderedDict()
        for name, var in self.items():
            left[name], right[name] = var.bisect(t, t_in_left)
        if self.is_aligned:
            left = TimeSeries(left, check_aligned=False)
            right = TimeSeries(right, check_aligned=False)
            left._is_aligned = True
            right._is_aligned = True
            return left, right
        else:
            return TimeSeries(left), TimeSeries(right)

    def window(self, t0: float, tf: float, include_tf: bool = False):
        """
        :param t0: The timestamp/datetime at the start of the window (inclusive)
        :param tf: The timestamp/datetime at the end of the window (inclusive
            if ``include_tf`` is ``True``, non-inclusive otherwise)
        :param include_tf: Whether to include ``tf`` in the window.

        :return: The subset of the time series occurring between timestamps
            ``t0`` (inclusive) and ``tf`` (included if ``include_tf`` is
            ``True``, excluded otherwise).
        :rtype: `TimeSeries`
        """
        return TimeSeries(ValIterOrderedDict([(k, var.window(t0, tf, include_tf)) for k, var in self.items()]))

    def to_pd(self) -> pd.DataFrame:
        """
        :return: A pandas DataFrame (indexed by time) which represents this time
            series. Each variable corresponds to a column of the DataFrame.
            Timestamps which are present for one variable but not another, are
            represented with NaN.
        """
        t = pd.DatetimeIndex([])
        univariates = [(name, var.to_pd()[~var.index.duplicated()]) for name, var in self.items()]
        for _, var in univariates:
            t = t.union(var.index)
        t = t.sort_values()
        df = pd.DataFrame(np.full((len(t), len(univariates)), np.nan), index=t, columns=self.names)
        for name, var in univariates:
            df.loc[var.index, name] = var[~var.index.duplicated()]
        return df

    @classmethod
    def from_pd(cls, df: Union[pd.Series, pd.DataFrame, np.ndarray], check_times=True, freq="1h"):
        """
        :param df: A pandas DataFrame with a DatetimeIndex. Each column
            corresponds to a different variable of the time series, and the
            key of column (in sorted order) give the relative order of those
            variables (in the list self.univariates). Missing values should be
            represented with ``NaN``. May also be a pandas Series for univariate
            time series.
        :param check_times: whether to check that all times in the index are
            unique (up to the millisecond) and sorted.
        :param freq: if ``df`` is not indexed by time, this is the frequency
            at which we will assume it is sampled.

        :rtype: TimeSeries
        :return: the `TimeSeries` object corresponding to ``df``.
        """
        if isinstance(df, pd.Series):
            return cls({df.name: UnivariateTimeSeries.from_pd(df[~df.isna()])})
        elif isinstance(df, np.ndarray):
            arr = df.reshape(len(df), -1).T
            ret = cls([UnivariateTimeSeries(time_stamps=None, values=v, freq=freq) for v in arr], check_aligned=False)
            ret._is_aligned = True
            return ret
        elif not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Time series is not aligned iff there are missing values
        aligned = df.shape[1] == 1 or not df.isna().any().any()

        # Make sure there are no time duplicates (by milliseconds) if desired
        dt_index = isinstance(df.index, pd.DatetimeIndex)
        if check_times:
            df = df[~df.index.duplicated()].sort_index()
            if dt_index:
                times = df.index.values.astype("datetime64[ms]").astype(np.int64)
                df = df.reindex(pd.to_datetime(np.unique(times), unit="ms"), method="bfill")

        elif not aligned and not dt_index and df.index.dtype not in ("int64", "float64"):
            raise RuntimeError(
                f"We only support instantiating time series from a "
                f"``pd.DataFrame`` with missing values when the data frame is "
                f"indexed by time, int, or float. This dataframe's index is of "
                f"type {type(df.index).__name__}"
            )

        ret = cls(
            ValIterOrderedDict(
                [(k, UnivariateTimeSeries.from_pd(ser[~ser.isna()], freq=freq)) for k, ser in df.items()]
            ),
            check_aligned=False,
        )
        ret._is_aligned = aligned
        return ret

    @classmethod
    def from_ts_list(cls, ts_list, *, check_aligned=True):
        """
        :param Iterable[TimeSeries] ts_list: iterable of time series we wish to
            form a multivariate time series with
        :param bool check_aligned: whether to check if the output time series is
            aligned
        :rtype: TimeSeries
        :return: A multivariate `TimeSeries` created from all the time series in
            the inputs.
        """
        ts_list = list(ts_list)
        all_names = [set(ts.names) for ts in ts_list]
        if all(
            len(names_i.intersection(names_j)) == 0
            for i, names_i in enumerate(all_names)
            for names_j in all_names[i + 1 :]
        ):
            univariates = ValIterOrderedDict(itertools.chain.from_iterable(ts.items() for ts in ts_list))
        else:
            univariates = list(itertools.chain.from_iterable(ts.univariates for ts in ts_list))
        return cls(univariates, check_aligned=check_aligned)

    def align(
        self,
        *,
        reference: Sequence[Union[int, float]] = None,
        granularity: Union[str, int, float] = None,
        origin: int = None,
        remove_non_overlapping=True,
        alignment_policy: AlignPolicy = None,
        aggregation_policy: AggregationPolicy = AggregationPolicy.Mean,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.Interpolate,
    ):
        """
        Aligns all the univariate time series comprising this multivariate time
        series so that they all have the same time stamps.

        :param reference: A specific set of timestamps we want the resampled
            time series to contain. Required if ``alignment_policy`` is
            `AlignPolicy.FixedReference`. Overrides other alignment policies
            if specified.
        :param granularity: The granularity (in seconds) of the resampled time
            time series. Defaults to the GCD time difference between adjacent
            elements of ``reference`` (when available) or ``time_series``
            (otherwise). Ignored if ``reference`` is given or ``alignment_policy``
            is `AlignPolicy.FixedReference`. Overrides other alignment policies
            if specified.
        :param origin: The first timestamp of the resampled time series. Only
            used if the alignment policy is AlignPolicy.FixedGranularity.
        :param remove_non_overlapping: If ``True``, we will only keep the portions
            of the univariates that overlap with each other. For example, if we
            have 3 univariates which span timestamps [0, 3600], [60, 3660], and
            [30, 3540], we will only keep timestamps in the range [60, 3540]. If
            ``False``, we will keep all timestamps produced by the resampling.
        :param alignment_policy: The policy we want to use to align the time
            time series.

            - `AlignPolicy.FixedReference` aligns each single-variable time
              series to ``reference``, a user-specified sequence of timestamps.
            - `AlignPolicy.FixedGranularity` resamples each single-variable time
              series at the same granularity, aggregating windows and imputing
              missing values as desired.
            - `AlignPolicy.OuterJoin` returns a time series with the union of
              all timestamps present in any single-variable time series.
            - `AlignPolicy.InnerJoin` returns a time series with the intersection
              of all timestamps present in all single-variable time series.
        :param aggregation_policy: The policy used to aggregate windows of adjacent
            observations when downsampling.
        :param missing_value_policy: The policy used to impute missing values
            created when upsampling.

        :rtype: TimeSeries
        :return: The resampled multivariate time series.
        """
        if self.is_empty():
            if reference is not None or granularity is not None:
                logger.warning(
                    "Attempting to align an empty time series to a set of reference time stamps or a "
                    "fixed granularity. Doing nothing."
                )
            return self.__class__.from_pd(self.to_pd())

        if reference is not None or alignment_policy is AlignPolicy.FixedReference:
            if reference is None:
                raise RuntimeError("`reference` is required when using `alignment_policy` FixedReference.")

            if alignment_policy not in [None, AlignPolicy.FixedReference]:
                logger.warning(
                    f"TimeSeries.align() received alignment policy "
                    f"{alignment_policy.name}, but a reference sequence of "
                    f"timestamps was also provided. `reference` is higher "
                    f"priority than `alignment_policy`, so we are using "
                    f"alignment policy FixedReference."
                )

            if granularity is not None:
                logger.warning(
                    "TimeSeries.align() received a granularity at which to "
                    "resample the time series, but a reference sequence of "
                    "timestamps was also provided. `reference` is higher "
                    "priority than `granularity`, so we are using alignment "
                    "policy FixedReference, not FixedGranularity."
                )

            # Align each univariate time series to the reference timestamps
            df = reindex_df(self.to_pd(), reference, missing_value_policy)
            return TimeSeries.from_pd(df, check_times=False)

        elif granularity is not None or alignment_policy is AlignPolicy.FixedGranularity:
            if alignment_policy not in [None, AlignPolicy.FixedGranularity]:
                logger.warning(
                    f"TimeSeries.align() received alignment policy "
                    f"{alignment_policy.name}, but a desired granularity at "
                    f"which to resample the time series was also received. "
                    f"`granularity` is higher priority than `alignment_policy`, "
                    f"so we are using alignment policy FixedGranularity."
                )

            # Get the granularity in seconds, if one is specified. Otherwise,
            # find the GCD granularity of  all the timedeltas that appear in any
            # of the univariate series.
            if granularity is None:
                granularity = infer_granularity(self.time_stamps)
            try:
                granularity = pd.to_timedelta(granularity, unit="s")
            except:
                granularity = to_offset(granularity)

            # Remove non-overlapping portions of univariates if desired
            df = self.to_pd()
            if remove_non_overlapping:
                t0 = max(v.index[0] for v in self.univariates if len(v) > 0)
                tf = min(v.index[-1] for v in self.univariates if len(v) > 0)
                df = df[t0:tf]

            # Resample at the desired granularity, setting the origin as needed
            if origin is None and isinstance(granularity, pd.Timedelta):
                elapsed = df.index[-1] - df.index[0]
                origin = df.index[0] + elapsed % granularity
            origin = to_pd_datetime(origin)
            new_df = df.resample(granularity, origin=origin, label="right", closed="right")

            # Apply aggregation & missing value imputation policies, and make sure we don't hallucinate new data
            new_df = aggregation_policy.value(new_df)
            new_df = missing_value_policy.value(new_df)
            new_df = new_df[df.index[0] : df.index[-1]]
            return TimeSeries.from_pd(new_df.ffill().bfill(), check_times=False)

        elif alignment_policy in [None, AlignPolicy.OuterJoin]:
            # Outer join is the union of all timestamps appearing in any of the
            # univariate time series. We just need to apply the missing value
            # policy to self.to_pd() (and bfill()/ffill() to take care of any
            # additional missing values at the start/end), and then return
            # from_pd().
            df = missing_value_policy.value(self.to_pd())
            if remove_non_overlapping:
                t0 = max(v.index[0] for v in self.univariates if len(v) > 0)
                tf = min(v.index[-1] for v in self.univariates if len(v) > 0)
                df = df[t0:tf]
            else:
                df = df.ffill().bfill()
            return TimeSeries.from_pd(df, check_times=False)

        elif alignment_policy is AlignPolicy.InnerJoin:
            # Inner join is the intersection of all the timestamps appearing in
            # all of the univariate time series. Just get the indexes of the
            # univariate sub time series where all variables are present.
            # TODO: add a resampling step instead of just indexing?
            ts = [set(var.np_time_stamps) for var in self.univariates]
            t = ts[0]
            for tprime in ts[1:]:
                t = t.intersection(tprime)
            if len(t) == 0:
                raise RuntimeError(
                    "No time stamps are shared between all variables! Try again with a different alignment policy."
                )
            t = to_pd_datetime(sorted(t))
            return TimeSeries.from_pd(self.to_pd().loc[t], check_times=False)

        else:
            raise RuntimeError(f"Alignment policy {alignment_policy.name} not supported")


def ts_csv_load(file_name: str, ms=True, n_vars=None) -> TimeSeries:
    """
    :param file_name: a csv file starting with the field timestamp followed by
        all the all variable names.
    :param ms: whether the timestamps are in milliseconds (rather than seconds)
    :return: A merlion `TimeSeries` object.
    """
    with open(file_name, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                names = line.strip().split(",")[1:]
                vars = {name: [] for name in names}
                stamps = []
                continue
            if not line:
                continue
            words = line.strip().split(",")
            stamp, vals = int(words[0]), words[1:]
            if ms:
                stamp = stamp / 1000
            stamps += [stamp]
            for name, val in zip(names, vals):
                vars[name] += [float(val)]

    return TimeSeries([UnivariateTimeSeries(stamps, vals, name) for name, vals in vars.items()][:n_vars])


def ts_to_csv(time_series: TimeSeries, file_name: str):
    """
    :param time_series: the `TimeSeries` object to write to a csv.
    :param file_name: the name to assign the csv file.
    """
    with open(file_name, "w") as f:
        header = ",".join(["timestamp"] + time_series.names)
        f.write(f"{header}\n")
        for t, x in time_series:
            vals = ",".join([str(v) for v in (int(t),) + x])
            f.write(f"{vals}\n")


def assert_equal_timedeltas(time_series: UnivariateTimeSeries, timedelta: float = None):
    """
    Checks that all time deltas in the time series are equal, either to each
    other, or a pre-specified timedelta (in seconds).
    """
    if pd.infer_freq(time_series.index) is not None:
        return
    if len(time_series) >= 2:
        timedeltas = np.diff(time_series.np_time_stamps)
        if timedelta is None:
            timedelta = timedeltas[0]
            assert timedelta > 0, "All times in the time series must be distinct."
        else:
            assert timedelta > 0

        assert (
            np.abs(timedeltas - timedeltas[0]).max() < 2e-3
        ), f"Data must be sampled with the same time difference between each element of the time series"
        assert np.abs(timedeltas[0] - timedelta) < 2e-3, (
            f"Expected data to be sampled every {timedelta} seconds, but time "
            f"series is sampled every {timedeltas[0]} seconds instead."
        )
