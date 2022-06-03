#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import json
import re
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import merlion.plot
from merlion.utils import UnivariateTimeSeries
from merlion.spark.dataset import read_dataset, create_hier_dataset
from merlion.spark.session import create_session, enable_aws_kwargs


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


def hierarchical_plot(pdf: pd.DataFrame, forecast_pdf: pd.DataFrame, index_cols: List[str], target_col: str):
    k = 3
    fig, axs = plt.subplots(nrows=2, ncols=k, figsize=(15, 8))
    kwargs = dict(pdf=pdf, forecast_pdf=forecast_pdf, time_col=pdf.columns[0], target_col=target_col)

    # Sort time series by how much they contribute to the total
    sub_df = pdf[~pdf.loc[:, index_cols].isna().any(axis=1)]
    totals = sub_df.groupby(index_cols).sum().sort_values(target_col, ascending=False).reset_index()

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
        index = totals.loc[i, index_cols]
        title = f"{target_col} ({','.join(f'{k}={totals.loc[i, k]}' for k in index_cols)})"
        plot(filter_fn=lambda x: x[(x.loc[:, index_cols] == index).all(axis=1)], ax=ax, title=title, **kwargs)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_data = config["train_data"]
    time_col = config.get("time_col", None)
    index_cols = config.get("index_cols", None) or []
    hierarchical = config.get("hierarchical", False) and len(index_cols) > 0
    target_col = config["target_col"]
    output_path = config.get("output_path", None)

    session_kwargs = {}
    if re.match("s3.*://", train_data) or re.match("s3.*://", output_path):
        session_kwargs.update(enable_aws_kwargs(credentials_provider=config.get("aws_credentials_provider")))
    spark = create_session(name="merlion-forecast-vis", **session_kwargs)

    train_df = read_dataset(spark=spark, path=train_data, time_col=time_col, index_cols=index_cols)
    if hierarchical:
        train_df, _ = create_hier_dataset(spark=spark, df=train_df, time_col=time_col, index_cols=index_cols)

    forecast_df = read_dataset(spark=spark, path=output_path, time_col=time_col, index_cols=index_cols)
    fig = hierarchical_plot(
        pdf=train_df.toPandas(), forecast_pdf=forecast_df.toPandas(), index_cols=index_cols, target_col=target_col
    )
    if args.figure is not None:
        fig.savefig(args.figure)
    plt.show()


if __name__ == "__main__":
    main()
