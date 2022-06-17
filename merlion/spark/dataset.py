#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Utils for reading & writing pyspark datasets.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import pyspark.sql
    import pyspark.sql.functions as F
    from pyspark.sql.types import DateType, StructType
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[spark]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

TSID_COL_NAME = "__ts_id"
"""
Many functions in this module rely on having a column named `TSID_COL_NAME` being in the dataset.
This column can be added manually using `add_tsid_column`, and its addition is handled automatically by `read_dataset`.
"""


def read_dataset(
    spark: pyspark.sql.SparkSession,
    path: str,
    file_format: str = "csv",
    time_col: str = None,
    index_cols: List[str] = None,
    data_cols: List[str] = None,
) -> pyspark.sql.DataFrame:
    """
    Reads a time series dataset as a pyspark Dataframe.

    :param spark: The current SparkSession.
    :param path: The path at which the dataset is stored.
    :param file_format: The file format the dataset is stored in.
    :param time_col: The name of the column which specifies timestamp. If ``None`` is provided, it is assumed to be the
        first column which is not an index column or pre-specified data column.
    :param index_cols: The columns used to index the various time series in the dataset. If ``None`` is provided, we
        assume the entire dataset is just a single time series.
    :param data_cols: The columns we will use for downstream time series tasks. If ``None`` is provided, we use all
        columns that are not a time or index column.

    :return: A pyspark dataframe with columns ``[time_col, *index_cols, *data_cols, TSID_COL_NAME]`` (in that order).
    """
    # Read the dataset into a pyspark dataframe
    df = spark.read.format(file_format).load(path, inferSchema=True, header=True)

    # Only keep the index column, data columns, and time column
    index_cols = index_cols or []
    if time_col is None:
        time_col = [c for c in df.schema.fieldNames() if c not in index_cols + (data_cols or [])][0]
    # Use all non-index non-time columns as data columns data columns are not given
    if data_cols is None or len(data_cols) == 0:
        data_cols = [c for c in df.schema.fieldNames() if c not in index_cols + [time_col]]
    assert all(col in data_cols and col not in index_cols + [time_col] for col in data_cols)

    # Get the columns in the right order & add TSID_COL_NAME to the end
    df = df.select(F.col(time_col).cast(DateType()).alias(time_col), *index_cols, *data_cols)
    return add_tsid_column(spark=spark, df=df, index_cols=index_cols)


def write_dataset(df: pyspark.sql.DataFrame, time_col: str, path: str, file_format: str = "csv"):
    """
    Writes the dataset at the specified path.

    :param df: The dataframe to save. The dataframe must have a column `TSID_COL_NAME`
        indexing the time series in the dataset (this column is automatically added by `read_dataset`).
    :param time_col: The name of the column which specifies timestamp.
    :param path: The path to save the dataset at.
    :param file_format: The file format in which to save the dataset.
    """
    df = df.sort([TSID_COL_NAME, time_col]).drop(TSID_COL_NAME)
    df.write.format(file_format).save(path, header=True, mode="overwrite")


def create_hier_dataset(
    spark: pyspark.sql.SparkSession, df: pyspark.sql.DataFrame, time_col: str = None, index_cols: List[str] = None
) -> Tuple[pyspark.sql.DataFrame, np.ndarray]:
    """
    Aggregates the time series in the dataset & appends them to the original dataset.

    :param spark: The current SparkSession.
    :param df: A pyspark dataframe containing all the data. The dataframe must have a column `TSID_COL_NAME`
        indexing the time series in the dataset (this column is automatically added by `read_dataset`).
    :param time_col: The name of the column which specifies timestamp. If ``None`` is provided, it is assumed to be the
        first column which is not an index column or pre-specified data column.
    :param index_cols: The columns used to index the various time series in the dataset. If ``None`` is provided, we
        assume the entire dataset is just a single time series. These columns define the levels of the hierarchy.
        For example, if each time series represents sales and we have ``index_cols = ["store", "item"]``, we will
        first aggregate sales for all items sold at a particular store; then we will aggregate sales for all items at
        all stores.

    :return: The dataset with additional time series corresponding to each level of the hierarchy, as well as a
        matrix specifying how the hierarchy is constructed.
    """
    # Determine which columns are which
    index_cols = [] if index_cols is None else index_cols
    extended_index_cols = index_cols + [TSID_COL_NAME]
    if time_col is None:
        non_index_cols = [c for c in df.schema.fieldNames() if c not in extended_index_cols]
        time_col = non_index_cols[0]
        data_cols = non_index_cols[1:]
    else:
        data_cols = [c for c in df.schema.fieldNames() if c not in extended_index_cols + [time_col]]

    # Create a pandas index for all the time series
    ts_index = df.groupBy(extended_index_cols).count().drop("count").toPandas()
    ts_index = ts_index.set_index(index_cols).sort_index()
    n = len(ts_index)

    # Add all higher levels of the hierarchy
    full_df = df
    hier_vecs = []
    for k in range(len(index_cols)):
        # Aggregate values of data columns over the last k+1 index column values.
        # TODO: maybe allow aggregations besides sum?
        agg_cols = index_cols if k < 0 else index_cols[: -(k + 1)]
        agg = df.groupBy([time_col] + agg_cols).agg(*[F.sum(c).alias(c) for c in data_cols])

        # Add back dummy NA values for the index columns we aggregated over, add a time series ID column,
        # concatenate the aggregated time series to the full dataframe, and compute the hierarchy vector.

        # For the top level of the hierarchy, this is easy as we just sum everything
        dummy_schema = StructType([full_df.schema[c] for c in extended_index_cols])
        if len(agg_cols) == 0:
            dummy = pd.DataFrame([[pd.NA] * len(index_cols) + [n + len(hier_vecs)]], columns=extended_index_cols)
            full_df = full_df.unionByName(agg.join(spark.createDataFrame(dummy, schema=dummy_schema)))
            hier_vecs.append(np.ones(n))
            continue

        # For lower levels of the hierarchy, we determine the membership of each grouping to create
        # the appropriate dummy entries and hierarchy vectors.
        dummy = []
        for i, (group, group_idxs) in enumerate(ts_index.groupby(agg_cols).groups.items()):
            group = [group] if len(agg_cols) == 1 else list(group)
            locs = [ts_index.index.get_loc(j) for j in group_idxs]
            dummy.append(group + [pd.NA] * (k + 1) + [n + len(hier_vecs)])
            x = np.zeros(n)
            x[locs] = 1
            hier_vecs.append(x)
        dummy = spark.createDataFrame(pd.DataFrame(dummy, columns=extended_index_cols), schema=dummy_schema)
        full_df = full_df.unionByName(agg.join(dummy, on=agg_cols))

    # Create the full hierarchy matrix, and return it along with the updated dataframe
    hier_matrix = np.concatenate([np.eye(n), np.stack(hier_vecs)])
    return full_df, hier_matrix


def add_tsid_column(
    spark: pyspark.sql.SparkSession, df: pyspark.sql.DataFrame, index_cols: List[str]
) -> pyspark.sql.DataFrame:
    """
    Adds the column `TSID_COL_NAME` to the dataframe, which assigns an integer ID to each time series in the dataset.

    :param spark: The current SparkSession.
    :param df: A pyspark dataframe containing all the data.
    :param index_cols: The columns used to index the various time series in the dataset.

    :return: The pyspark dataframe with an additional column `TSID_COL_NAME` added as the last column.
    """
    if TSID_COL_NAME in df.schema.fieldNames():
        return df

    # If no index columns are specified, we are only dealing with a single time series
    if index_cols is None or len(index_cols) == 0:
        return df.join(spark.createDataFrame(pd.DataFrame([0], columns=[TSID_COL_NAME])))

    # Compute time series IDs. Time series with any null indexes come last b/c these are aggregated time series.
    ts_index = df.groupBy(index_cols).count().drop("count").toPandas()
    null_rows = ts_index.isna().any(axis=1)
    ts_index = pd.concat(
        (
            ts_index[~null_rows].sort_values(by=index_cols, axis=0, ascending=True),
            ts_index[null_rows].sort_values(by=index_cols, axis=0, ascending=True),
        ),
        axis=0,
    )
    ts_index[TSID_COL_NAME] = np.arange(len(ts_index))

    # Add the time series ID column to the overall dataframe
    ts_index = spark.createDataFrame(ts_index)
    for i, col in enumerate(index_cols):
        pred = df[col].eqNullSafe(ts_index[col])
        condition = pred if i == 0 else condition & pred
    df = df.join(ts_index, on=condition)
    for col in index_cols:
        df = df.drop(ts_index[col])
    return df
