#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Utils for data I/O.
"""
from collections import OrderedDict
import inspect
from typing import Any, Dict, List, Mapping, Union

import numpy as np
import pandas as pd

from merlion.utils.misc import combine_signatures, parse_basic_docstring
from merlion.utils.time_series import TimeSeries


def df_to_time_series(
    df: pd.DataFrame,
    time_col: str = None,
    timestamp_unit="s",
    index_cols: Union[str, List[str]] = None,
    data_cols: Union[str, List[str]] = None,
    index_conditions: Dict[str, Any] = None,
    index_agg_average=False,
) -> TimeSeries:
    """
    Converts a general ``pandas.DataFrame`` to a `TimeSeries` object.

    This function allows a user to specify a hierarchical index to be aggregated over time. For example, the
    dataframe may contain sales volume for multiple different stores & items, differentiated by columns ``"store_id"``
    and ``"item_id"``. In this case, you should specify ``index_cols=["store_id", "item_id"]``.

    By default, we take a simple sum of all distinct values at each timestamp at each level of the hierarchical
    index. However, you may customize this behavior by specifying ``index_conditions``. Here are some examples:

    -   ``{"store_id": {"vals": [1, 8]}}`` takes the total of all sales for all items in stores 1 & 8 only.
    -   ``{"store_id": {"vals": [1, 8], "weights": [0.5, 1.3]}, "item_id": {"weights": "price"}}`` uses the
        column ``"price"`` to weight the sales of each item before summing them up (aka revenue). Then, we
        weight the price-weighted sales in store 1 by 0.5 & the price-weighted sales in store 8 by 1.3,
        before summing them together to obtain a total for each timestamp.

    If ``index_conditions`` is not specified for a particular non-temporal index column, we take a simple sum
    of all distinct values. This is also true if ``"weights"`` is not specified for a particular index key.

    :param df: the dataframe to parse
    :param time_col: the name of the column specifying time. If none is specified, the existing index is used if it
        is a ``DatetimeIndex``. Otherwise, the first column is used..
    :param timestamp_unit: if the time column is in Unix timestamps, this is the unit of the timestamp.
    :param index_cols: the columns to be interpreted as a hierarchical index, if desired.
    :param data_cols: the columns representing the actual data values of interest.
    :param index_conditions: a dict specifying how the hierarchical index should be aggregated.
    :param index_agg_average: aggregate with (weighted) average if ``True``, (weighted) sum if ``False``.
    """
    # Get the index columns
    if index_cols is None:
        index_cols = []
    elif not isinstance(index_cols, (list, tuple)):
        index_cols = [index_cols]
    if not all(c in df.columns for c in index_cols):
        raise KeyError(f"Expected each of index_cols to be in {df.columns}. Got {index_cols}.")

    # Set up a hierarchical index for the dataframe, with the timestamp first
    if time_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index([df.index] + index_cols).sort_index()
        if df.index.names[0] is None:
            df.index.set_names("time", level=0, inplace=True)
        time_col = df.index.names[0]
    else:
        if time_col is None:
            time_col = df.columns[0]
        elif time_col not in df.columns:
            raise KeyError(f"Expected time_col to be in {df.columns}. Got {time_col}.")
        df[time_col] = pd.to_datetime(df[time_col], unit=None if df[time_col].dtype == "O" else timestamp_unit)
        df = df.set_index([time_col] + index_cols).sort_index()

    # Determine the values & weights used to restrict & aggregate the dataframe
    vals_seq = [slice(None)]
    weights = pd.Series(1.0, index=df.index)
    index_conditions = index_conditions or {}
    for c in index_cols:
        cond = index_conditions.get(c, {})
        if not isinstance(cond, Mapping):
            cond = {"vals": cond}

        # Determine if we're restricting the dataframe
        vals = cond.get("vals", None)
        if vals is None:
            vals_seq.append(slice(None))
            vals = df.index.get_level_values(c).unique()
        else:
            vals_seq.append(vals)

        # Get the weights for this level of the aggregation
        w = cond.get("weights", None)
        if w is not None and isinstance(w, str):
            weights *= df[w]
        elif w is not None:
            all_vals = df.index.get_level_values(c)
            if len(w) != len(vals):
                raise ValueError(f"For index column {c}, expected weights of length {len(vals)}. Got {len(w)}.")
            w = pd.concat((pd.Series(w, index=vals), pd.Series(0, index=all_vals.unique().difference(vals))))
            weights *= w.loc[all_vals].values

    # Get only the desired columns from the dataframe
    if data_cols is not None:
        data_cols = [data_cols] if not isinstance(data_cols, (list, tuple)) else data_cols
        if not all(c in df.columns for c in data_cols):
            raise KeyError(f"Expected each of target_cols to be in {df.colums}. Got {data_cols}.")
        df = df[data_cols]

    # Restrict & aggregate the dataframe
    if len(index_cols) > 0:
        ilocs = df.index.get_locs(vals_seq)
        df = df.iloc[ilocs]
        weights = weights.iloc[ilocs]
        if index_agg_average:
            df = df.groupby(time_col).agg(lambda x: np.average(x, weights=weights.loc[x.index]))
        else:
            df = (df * weights.values.reshape(-1, 1)).groupby(time_col).sum()

    # Convert the dataframe to a time series & return it
    return TimeSeries.from_pd(df)


def data_io_decorator(func):
    """
    Decorator to standardize docstrings for data I/O functions.
    """

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Parse the docstrings of the base df_to_time_series function & decorated function.
    prefix, suffix, params = parse_basic_docstring(func.__doc__)
    base_prefix, base_suffix, base_params = parse_basic_docstring(df_to_time_series.__doc__)

    # Combine the prefixes. Base prefix starts after the first line break.
    i_lb = [i for i, line in enumerate(base_prefix) if line == ""][1]
    prefix = ("\n".join(prefix) if any([line != "" for line in prefix]) else "") + "\n".join(base_prefix[i_lb:])

    # The base docstring has no suffix, so just use the function's
    suffix = "\n".join(suffix) if any([line != "" for line in suffix]) else ""

    # Combine the parameter lists
    for param, docstring_lines in base_params.items():
        if param not in params:
            params[param] = "\n".join(docstring_lines).rstrip("\n")

    # Combine the signatures, but remove some parameters that are specific to the original (as well as kwargs).
    new_sig_params = []
    sig = combine_signatures(inspect.signature(func), inspect.signature(df_to_time_series))
    for param in sig.parameters.values():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            break
        if param.name not in ["df"]:
            new_sig_params.append(param)
    sig = sig.replace(parameters=new_sig_params)

    # Update the signature and docstring of the wrapper we are returning. Use only the params in the new signature.
    wrapper.__signature__ = sig
    params = OrderedDict((p, params[p]) for p in sig.parameters if p in params)
    wrapper.__doc__ = (prefix or "") + "\n" + "\n".join(params.values()) + "\n\n" + (suffix or "")
    return wrapper


@data_io_decorator
def csv_to_time_series(file_name: str, **kwargs) -> TimeSeries:
    """
    Reads a CSV file and converts it to a `TimeSeries` object.
    """
    return df_to_time_series(pd.read_csv(file_name), **kwargs)
