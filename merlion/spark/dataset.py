#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql.types import DateType

TSID_COL_NAME = "__ts_id"


def read_dataset(
    spark: pyspark.sql.SparkSession,
    path: str,
    time_col: str = None,
    index_cols: List[str] = None,
    data_cols: List[str] = None,
) -> pyspark.sql.DataFrame:
    """

    :param spark:
    :param path:
    :param time_col:
    :param index_cols:
    :param data_cols:
    :return:
    """
    # Read the dataset into a pyspark dataframe
    df = spark.read.csv(path, inferSchema=True, header=True)

    # Only keep the index column, data columns, and time column
    index_cols = index_cols or []
    if time_col is None:
        time_col = [c for c in df.schema.fieldNames() if c not in index_cols + (data_cols or [])][0]
    # Use all non-index non-time columns as data columns data columns are not given
    if data_cols is None or len(data_cols) == 0:
        data_cols = [c for c in df.schema.fieldNames() if c not in index_cols + [time_col]]
    assert all(col in data_cols and col not in index_cols + [time_col] for col in data_cols)
    df = df.select(F.col(time_col).cast(DateType()).alias(time_col), *index_cols, *data_cols)

    # If no index columns are specified, we are only dealing with a single time series
    if len(index_cols) == 0:
        return df.join(spark.createDataFrame(pd.DataFrame([0], columns=[TSID_COL_NAME])))

    # Otherwise, create a pandas dataframe indexed by the columns indexing the time series, and
    # also add a new time series ID column to the full dataset.
    ts_index = df.groupBy(index_cols).count().drop("count").sort(index_cols).toPandas()
    ts_index[TSID_COL_NAME] = np.arange(len(ts_index))
    ts_index = spark.createDataFrame(ts_index)
    for i, col in enumerate(index_cols):
        pred = df[col].eqNullSafe(ts_index[col])
        condition = pred if i == 0 else condition & pred
    return df.join(ts_index, on=condition, how="inner").drop(*[ts_index[col] for col in index_cols])


def create_hier_dataset(
    spark: pyspark.sql.SparkSession, df: pyspark.sql.DataFrame, time_col: str = None, index_cols: List[str] = None
) -> Tuple[pyspark.sql.DataFrame, np.ndarray]:
    """

    :param spark:
    :param df:
    :param time_col:
    :param index_cols:
    :return:
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
        agg = df.groupBy(agg_cols + [time_col]).agg(*[F.sum(c).alias(c) for c in data_cols])

        # Add back dummy NA values for the index columns we aggregated over, add a time series ID column,
        # concatenate the aggregated time series to the full dataframe, and compute the hierarchy vector.

        # For the top level of the hierarchy, this is easy as we just sum everything
        if len(agg_cols) == 0:
            dummy = pd.DataFrame([[pd.NA] * len(index_cols) + [n + len(hier_vecs)]], columns=extended_index_cols)
            full_df = full_df.unionByName(agg.join(spark.createDataFrame(dummy)))
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
        dummy = spark.createDataFrame(pd.DataFrame(dummy, columns=extended_index_cols))
        full_df = full_df.unionByName(agg.join(dummy, on=agg_cols))

    # Create the full hierarchy matrix, and return it along with the updated dataframe
    hier_matrix = np.concatenate([np.eye(n), np.stack(hier_vecs)])
    return full_df, hier_matrix


def write_result(df: pyspark.sql.DataFrame, time_col: str, path: str):
    """

    :param df:
    :param time_col:
    :param path:
    :return:
    """
    df = df.sort([TSID_COL_NAME, time_col]).drop(TSID_COL_NAME)
    df.write.csv(path, header=True, mode="overwrite")
