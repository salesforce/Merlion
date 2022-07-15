#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from os.path import abspath, dirname, join
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, FloatType, StructField, StructType
from merlion.spark.dataset import read_dataset, write_dataset, TSID_COL_NAME
from merlion.spark.pandas_udf import anomaly


logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


def _run_job(name: str, data_cols: list, model: dict):
    index_cols = ["Store", "Dept"]
    time_col = "Date"
    train_test_split = "2012-06-01"
    spark = SparkSession.builder.master("local[*]").appName("unit-tests").getOrCreate()

    df = read_dataset(
        spark=spark,
        file_format="csv",
        path=join(rootdir, "data", "walmart", "walmart_mini.csv"),
        index_cols=index_cols,
        time_col=time_col,
        data_cols=data_cols,
    )
    index_cols = index_cols + [TSID_COL_NAME]

    index_fields = [df.schema[c] for c in index_cols]
    pred_fields = [StructField(time_col, DateType()), StructField("anom_score", FloatType())]
    output_schema = StructType(index_fields + pred_fields)
    anomaly_df = df.groupBy(index_cols).applyInPandas(
        lambda pdf: anomaly(
            pdf, index_cols=index_cols, time_col=time_col, train_test_split=train_test_split, model=model
        ),
        schema=output_schema,
    )

    output_path = join(rootdir, "tmp", "spark", "anomaly", name)
    write_dataset(df=anomaly_df, time_col=time_col, path=output_path, file_format="csv")


def test_univariate():
    _run_job(name="univariate", data_cols=["Weekly_Sales"], model={"name": "StatThreshold"})


def test_multivariate():
    _run_job(name="multivariate", data_cols=["Weekly_Sales", "Temperature", "CPI"], model={"name": "IsolationForest"})
