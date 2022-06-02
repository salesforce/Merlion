#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import json
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql.types import DateType, FloatType, StructField, StructType

import merlion.plot
from merlion.utils import UnivariateTimeSeries
from merlion.spark.dataset import create_hier_dataset, read_dataset, write_result, TSID_COL_NAME
from merlion.spark.pandas_udf import forecast, reconciliation
from merlion.spark.session import create_session


def plot(
    pdf: pd.DataFrame,
    forecast_pdf: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    filter_fn: Callable,
    time_col: str,
    target_col: str,
):
    # Filter the dataframes to get the desired time series
    pdf = filter_fn(pdf).set_index(time_col).sort_index()
    forecast_pdf = filter_fn(forecast_pdf).set_index(time_col).sort_index()

    # Get error bars if desired
    err_col = f"{target_col}_err"
    yhat = forecast_pdf[target_col]
    if err_col in forecast_pdf.columns and not forecast_pdf[err_col].isna().all():
        lb = yhat - np.minimum(yhat.values, forecast_pdf[err_col].values)
        ub = yhat + forecast_pdf[err_col]
    else:
        lb = ub = None

    # Plot the figure
    fig = merlion.plot.Figure(
        y_prev=UnivariateTimeSeries.from_pd(pdf[target_col]),
        yhat=UnivariateTimeSeries.from_pd(yhat),
        yhat_lb=UnivariateTimeSeries.from_pd(lb),
        yhat_ub=UnivariateTimeSeries.from_pd(ub),
    )
    return fig.plot(ax=ax, title=title)


def hierarchical_plot(
    pdf: pd.DataFrame, forecast_pdf: pd.DataFrame, time_col: str, index_cols: List[str], target_col: str
):
    k = 3
    fig, axs = plt.subplots(nrows=2, ncols=k, figsize=(15, 8))
    kwargs = dict(pdf=pdf, forecast_pdf=forecast_pdf, time_col=time_col, target_col=target_col)

    # Sort time series by how much they contribute to the total
    sub_df = pdf[~pdf.loc[:, index_cols].isna().any(axis=1)]
    totals = sub_df.groupby(index_cols + [TSID_COL_NAME]).sum().sort_values(target_col, ascending=False).reset_index()

    # Set up the top row to be a single big plot for the full aggregation
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    ax = fig.add_subplot(gs[0, :])
    title = f"{target_col} (Global Total)"
    plot(filter_fn=lambda x: x[x.loc[:, index_cols].isna().all(axis=1)], ax=ax, title=title, **kwargs)

    # Plot the k biggest contributors from the base of the hierarchy on the bottom row
    for i in range(k):
        ax = axs[1, i]
        tsid = totals.loc[i, TSID_COL_NAME]
        title = f"{target_col} ({','.join(f'{k}={totals.loc[i, k]}' for k in index_cols)})"
        plot(filter_fn=lambda x: x[x.loc[:, TSID_COL_NAME] == tsid], ax=ax, title=title, **kwargs)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_data = config["train_data"]
    time_stamps = config["time_stamps"]
    time_col = config.get("time_col", None)
    index_cols = config.get("index_cols", None) or []
    hierarchical = config.get("hierarchical", False) and len(index_cols) > 0
    target_col = config["target_col"]
    data_cols = config.get("data_cols", [target_col])

    # Set up the model with the appropriate target_seq_index
    assert target_col in data_cols
    target_seq_index = {v: i for i, v in enumerate(data_cols)}[target_col]
    model = config.get("model")
    model["target_seq_index"] = target_seq_index

    # Read the dataset as a Spark DataFrame, and process it.
    # This will add a TSID_COL_NAME column to identify each time series with a single integer.
    spark = create_session()
    df = read_dataset(spark=spark, path=train_data, time_col=time_col, index_cols=index_cols, data_cols=data_cols)
    time_col = [c for c in df.schema.fieldNames() if c not in index_cols + [TSID_COL_NAME]][0]

    # Convert to a hierarchical dataset if desired
    if hierarchical:
        df, hier_matrix = create_hier_dataset(spark=spark, df=df, time_col=time_col, index_cols=index_cols)

    # Use spark to generate forecasts for each time series in parallel
    index_fields = [df.schema[c] for c in index_cols + [TSID_COL_NAME]]
    pred_fields = [
        StructField(time_col, DateType()),
        StructField(target_col, FloatType()),
        StructField(f"{target_col}_err", FloatType()),
    ]
    output_schema = StructType(index_fields + pred_fields)
    forecast_df = df.groupBy(index_cols).applyInPandas(
        lambda pdf: forecast(pdf, index_cols, time_col, target_col, time_stamps, model), schema=output_schema
    )
    write_result(df=forecast_df, time_col=time_col, path="data/demand/raw")

    # Apply hierarchical time series reconciliation if desired
    if hierarchical:
        forecast_df = forecast_df.groupBy(time_col).applyInPandas(
            lambda pdf: reconciliation(pdf, hier_matrix, target_col), schema=output_schema
        )
        write_result(df=forecast_df, time_col=time_col, path="data/demand/reconciled")
        fig = hierarchical_plot(
            pdf=df.toPandas(),
            forecast_pdf=forecast_df.toPandas(),
            time_col=time_col,
            index_cols=index_cols,
            target_col=target_col,
        )
        plt.show()


if __name__ == "__main__":
    main()
