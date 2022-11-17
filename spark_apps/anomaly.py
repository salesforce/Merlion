#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import json
import re

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, FloatType, StructField, StructType
from merlion.spark.dataset import read_dataset, write_dataset, TSID_COL_NAME
from merlion.spark.pandas_udf import anomaly


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path at which the dataset is stored.")
    parser.add_argument("--output_path", required=True, help="Path at which to save output anomaly scores.")
    parser.add_argument(
        "--train_test_split", required=True, help="First timestamp in the dataset which should be used for testing."
    )
    parser.add_argument("--file_format", default="csv", help="File format of train data & output file.")
    parser.add_argument(
        "--model",
        default=json.dumps({"name": "DefaultDetector"}),
        help="JSON dict specifying the model we wish to use for anomaly detection.",
    )
    parser.add_argument(
        "--index_cols",
        default="[]",
        help="JSON list of columns used to demarcate different time series. For example, if the dataset contains sales "
        'for multiple items at different stores, this could be \'["store", "item"]\'. '
        "If not given, we assume the dataset contains only 1 time series.",
    )
    parser.add_argument(
        "--time_col",
        default=None,
        help="Name of the column containing timestamps. If not given, use the first non-index column.",
    )
    parser.add_argument(
        "--data_cols",
        default="[]",
        help="JSON list of columns to use when modeling the data. If not given, use all non-index, non-time columns.",
    )
    parser.add_argument(
        "--predict_on_train", action="store_true", help="Whether to return the model's prediction on the training data."
    )
    args = parser.parse_args()

    # Parse index_cols JSON string
    try:
        index_cols = json.loads(re.sub("'", '"', args.index_cols))
        assert isinstance(index_cols, list)
    except (json.decoder.JSONDecodeError, AssertionError) as e:
        parser.error(
            f"Expected --index_cols to be a JSON list. Got {args.index_cols}.\n" f"Caught {type(e).__name__}({e})"
        )
    else:
        args.index_cols = index_cols

    # Parse data_cols JSON string
    try:
        data_cols = json.loads(re.sub("'", '"', args.data_cols))
        assert isinstance(data_cols, list)
    except (json.decoder.JSONDecodeError, AssertionError) as e:
        parser.error(
            f"Expected --data_cols to be a JSON list if given. Got {args.data_cols}.\n"
            f"Caught {type(e).__name__}({e})"
        )
    else:
        args.data_cols = data_cols

    # Parse JSON string for the model and set the model's target_seq_index
    try:
        model = json.loads(re.sub("'", '"', args.model))
        assert isinstance(model, dict)
    except (json.decoder.JSONDecodeError, AssertionError) as e:
        parser.error(
            f"Expected --model to be a JSON dict specifying a Merlion model. Got {args.model}.\n"
            f"Caught {type(e).__name__}({e})"
        )
    else:
        args.model = model

    return args


def main():
    args = parse_args()

    # Read the dataset as a Spark DataFrame, and process it.
    # This will add a TSID_COL_NAME column to identify each time series with a single integer.
    spark = SparkSession.builder.appName("anomaly").getOrCreate()
    df = read_dataset(
        spark=spark,
        file_format=args.file_format,
        path=args.data,
        time_col=args.time_col,
        index_cols=args.index_cols,
        data_cols=args.data_cols,
    )
    if args.time_col is None:
        args.time_col = df.schema.fieldNames()[0]
    args.index_cols = args.index_cols + [TSID_COL_NAME]

    # Use spark to predict anomaly scores for each time series in parallel
    index_fields = [df.schema[c] for c in args.index_cols]
    pred_fields = [StructField(args.time_col, DateType()), StructField("anom_score", FloatType())]
    output_schema = StructType(index_fields + pred_fields)
    anomaly_df = df.groupBy(args.index_cols).applyInPandas(
        lambda pdf: anomaly(
            pdf,
            index_cols=args.index_cols,
            time_col=args.time_col,
            train_test_split=args.train_test_split,
            model=args.model,
            predict_on_train=args.predict_on_train,
        ),
        schema=output_schema,
    )

    write_dataset(df=anomaly_df, time_col=args.time_col, path=args.output_path, file_format=args.file_format)


if __name__ == "__main__":
    main()
