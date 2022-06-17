#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, FloatType, StructField, StructType
from merlion.spark.dataset import create_hier_dataset, read_dataset, write_dataset, TSID_COL_NAME
from merlion.spark.pandas_udf import forecast, reconciliation


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
    output_path = config["output_path"]

    # Set up the model with the appropriate target_seq_index
    assert target_col in data_cols
    target_seq_index = {v: i for i, v in enumerate(data_cols)}[target_col]
    model = config.get("model")
    model["target_seq_index"] = target_seq_index

    # Read the dataset as a Spark DataFrame, and process it.
    # This will add a TSID_COL_NAME column to identify each time series with a single integer.
    spark = SparkSession.builder.appName("forecast").getOrCreate()
    df = read_dataset(spark=spark, path=train_data, time_col=time_col, index_cols=index_cols, data_cols=data_cols)
    time_col = df.schema.fieldNames()[0]

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

    # Apply hierarchical time series reconciliation if desired
    if hierarchical:
        forecast_df = forecast_df.groupBy(time_col).applyInPandas(
            lambda pdf: reconciliation(pdf, hier_matrix, target_col), schema=output_schema
        )

    write_dataset(df=forecast_df, time_col=time_col, path=output_path)


if __name__ == "__main__":
    main()
