#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from os.path import abspath, dirname, join
import logging

from pyspark.sql.types import DateType, FloatType, StructField, StructType
from merlion.spark.dataset import create_hier_dataset, read_dataset, write_dataset, TSID_COL_NAME
from merlion.spark.pandas_udf import forecast, reconciliation


logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


def _run_job(
    spark, name: str, data_cols: list, hierarchical: bool, agg_dict: dict, predict_on_train: bool, robust: bool
):
    logger.info(f"test_spark_forecast_{name}\n{'-' * 80}")
    index_cols = ["Store", "Dept"]
    target_col = "Weekly_Sales"
    time_col = "Date"
    time_stamps = ["2012-11-02", "2012-11-09", "2012-11-16", "2012-11-23", "2012-11-30", "2012-12-07", "2012-12-14"]

    df = read_dataset(
        spark=spark,
        file_format="csv",
        path=join(rootdir, "data", "walmart", "walmart_mini_error.csv" if robust else "walmart_mini.csv"),
        index_cols=index_cols,
        time_col=time_col,
        data_cols=data_cols,
    )
    index_cols = index_cols + [TSID_COL_NAME]

    if hierarchical:
        df, hier_matrix = create_hier_dataset(
            spark=spark, df=df, time_col=time_col, index_cols=index_cols, agg_dict=agg_dict
        )

    index_fields = [df.schema[c] for c in index_cols]
    pred_fields = [
        StructField(time_col, DateType()),
        StructField(target_col, FloatType()),
        StructField(f"{target_col}_err", FloatType()),
    ]
    output_schema = StructType(index_fields + pred_fields)
    target_seq_index = {v: i for i, v in enumerate(data_cols)}[target_col]
    forecast_df = df.groupBy(index_cols).applyInPandas(
        lambda pdf: forecast(
            pdf,
            index_cols=index_cols,
            time_col=time_col,
            target_col=target_col,
            time_stamps=time_stamps,
            model=dict(name="DefaultForecaster", target_seq_index=target_seq_index),
            predict_on_train=predict_on_train,
            agg_dict=agg_dict,
        ),
        schema=output_schema,
    )
    df.unpersist()

    if hierarchical:
        forecast_df = forecast_df.groupBy(time_col).applyInPandas(
            lambda pdf: reconciliation(pdf, hier_matrix=hier_matrix, target_col=target_col), schema=output_schema
        )

    output_path = join(rootdir, "tmp", "spark", "forecast", name)
    write_dataset(df=forecast_df, time_col=time_col, path=output_path, file_format="csv")
    forecast_df.unpersist()


def test_univariate(spark_session):
    _run_job(
        spark=spark_session,
        name="univariate",
        data_cols=["Weekly_Sales"],
        hierarchical=True,
        agg_dict={},
        predict_on_train=False,
        robust=False,
    )


def test_non_hts(spark_session):
    _run_job(
        spark=spark_session,
        name="non_hts",
        data_cols=["Weekly_Sales"],
        hierarchical=False,
        agg_dict={},
        predict_on_train=False,
        robust=False,
    )


def test_multivariate(spark_session):
    _run_job(
        spark=spark_session,
        name="multivariate",
        data_cols=["Weekly_Sales", "Temperature", "CPI"],
        hierarchical=True,
        agg_dict={"Weekly_Sales": "sum", "Temperature": "mean", "CPI": "mean"},
        predict_on_train=False,
        robust=False,
    )


def test_mixed(spark_session):
    _run_job(
        spark=spark_session,
        name="mixed",
        data_cols=["Weekly_Sales", "Temperature", "CPI"],
        hierarchical=True,
        agg_dict={"Weekly_Sales": "sum"},  # only use Weekly_Sales for the target
        predict_on_train=True,
        robust=False,
    )


def test_robust(spark_session):
    _run_job(
        spark=spark_session,
        name="robust",
        data_cols=["Weekly_Sales", "Temperature", "CPI"],
        hierarchical=True,
        agg_dict={"Weekly_Sales": "sum"},  # only use Weekly_Sales for the target
        predict_on_train=True,
        robust=True,
    )
