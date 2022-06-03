#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import json

from pyspark.sql.types import DateType, FloatType, StructField, StructType
from merlion.spark.dataset import read_dataset, write_result, TSID_COL_NAME
from merlion.spark.pandas_udf import anomaly
from merlion.spark.session import create_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    train_data = config["train_data"]
    train_test_split = config.get("train_test_split", None)
    time_col = config.get("time_col", None)
    index_cols = config.get("index_cols", None) or []
    data_cols = config.get("data_cols", None) or []
    output_path = config["output_path"]
    model = config.get("model")

    # Read the dataset as a Spark DataFrame, and process it.
    # This will add a TSID_COL_NAME column to identify each time series with a single integer.
    spark = create_session()
    df = read_dataset(spark=spark, path=train_data, time_col=time_col, index_cols=index_cols, data_cols=data_cols)
    df = df.where((df.item < 3) & (df.store == 1))
    time_col = [c for c in df.schema.fieldNames() if c not in index_cols + [TSID_COL_NAME]][0]

    # Use spark to predict anomaly scores for each time series in parallel
    index_fields = [df.schema[c] for c in index_cols + [TSID_COL_NAME]]
    pred_fields = [StructField(time_col, DateType()), StructField("anom_score", FloatType())]
    output_schema = StructType(index_fields + pred_fields)
    anomaly_df = df.groupBy(index_cols).applyInPandas(
        lambda pdf: anomaly(pdf, index_cols, time_col, train_test_split, model), schema=output_schema
    )

    write_result(df=anomaly_df, time_col=time_col, path=output_path)


if __name__ == "__main__":
    main()
