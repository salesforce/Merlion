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
from typing import List, Union

import pandas as pd

from merlion.utils.misc import combine_signatures, parse_basic_docstring
from merlion.utils.time_series import TimeSeries


def df_to_time_series(
    df: pd.DataFrame, time_col: str = None, timestamp_unit="s", data_cols: Union[str, List[str]] = None
) -> TimeSeries:
    """
    Converts a general ``pandas.DataFrame`` to a `TimeSeries` object.

    :param df: the dataframe to process
    :param time_col: the name of the column specifying time. If ``None`` is specified, the existing index
        is used if it is a ``DatetimeIndex``. Otherwise, the first column is used.
    :param timestamp_unit: if the time column is in Unix timestamps, this is the unit of the timestamp.
    :param data_cols: the columns representing the actual data values of interest.
    """
    # Set up the time index
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col is None:
            time_col = df.columns[0]
        elif time_col not in df.columns:
            raise KeyError(f"Expected `time_col` to be in {df.columns}. Got {time_col}.")
        df[time_col] = pd.to_datetime(df[time_col], unit=None if df[time_col].dtype == "O" else timestamp_unit)
        df = df.set_index(time_col)
    df = df.sort_index()

    # Get only the desired columns from the dataframe
    if data_cols is not None:
        data_cols = [data_cols] if not isinstance(data_cols, (list, tuple)) else data_cols
        if not all(c in df.columns for c in data_cols):
            raise KeyError(f"Expected each of `data_cols` to be in {df.colums}. Got {data_cols}.")
        df = df[data_cols]

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
