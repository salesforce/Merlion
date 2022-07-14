#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import json
import re
from warnings import warn

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, FloatType, StructField, StructType
from merlion.spark.dataset import create_hier_dataset, read_dataset, write_dataset, TSID_COL_NAME
from merlion.spark.pandas_udf import forecast, reconciliation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, help="Path at which the train data is stored.")
    parser.add_argument("--output_path", required=True, help="Path at which to save output forecasts.")
    parser.add_argument(
        "--time_stamps",
        required=True,
        help='JSON list of times we want to forecast, e.g. \'["2022-01-01 00:00:00", "2020-01-01 00:01:00"]\'.',
    )
    parser.add_argument("--target_col", required=True, help="Name of the column whose value we want to forecast.")
    parser.add_argument(
        "--predict_on_train", action="store_true", help="Whether to return the model's prediction on the training data."
    )
    parser.add_argument("--file_format", default="csv", help="File format of train data & output file.")
    parser.add_argument(
        "--model",
        default=json.dumps({"name": "DefaultForecaster"}),
        help="JSON dict specifying the model we wish to use for forecasting.",
    )
    parser.add_argument(
        "--index_cols",
        default="[]",
        help="JSON list of columns used to demarcate different time series. For example, if the dataset contains sales "
        'for multiple items at different stores, this could be \'["store", "item"]\'. '
        "If not given, we assume the dataset contains only 1 time series.",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=False,
        help="Whether the time series have a hierarchical structure. If true, we aggregate the time series in the "
        "dataset (by summation), in the order specified by index_cols. For example, if index_cols is "
        '\'["store", "item"]\', we first sum the sales of all items within store, and then sum the global '
        "sales of all stores and all items.",
    )
    parser.add_argument(
        "--agg_dict",
        default="{}",
        help="JSON dict indicating how different data columns should be aggregated if working with hierarchical time "
        "series. Keys are column names, values are names of standard aggregations (e.g. sum, mean, max, etc.). "
        "If a column is not specified, it is not aggregated. Note that we always sum the target column, regardless of "
        "whether it is specified. This ensures that hierarchical time series reconciliation works correctly.",
    )
    parser.add_argument(
        "--time_col",
        default=None,
        help="Name of the column containing timestamps. We use the first non-index column if one is not given.",
    )
    parser.add_argument(
        "--data_cols",
        default=None,
        help="JSON list of columns to use when modeling the data."
        "If not given, we do univariate forecasting using only target_col.",
    )
    args = parser.parse_args()

    # Parse time_stamps JSON string
    try:
        time_stamps = json.loads(re.sub("'", '"', args.time_stamps))
        assert isinstance(time_stamps, list) and len(time_stamps) > 0
    except (json.decoder.JSONDecodeError, AssertionError) as e:
        parser.error(
            f"Expected --time_stamps to be a non-empty JSON list. Got {args.time_stamps}.\n Caught {type(e).__name__}({e})"
        )
    else:
        args.time_stamps = time_stamps

    # Parse index_cols JSON string
    try:
        index_cols = json.loads(re.sub("'", '"', args.index_cols)) or []
        assert isinstance(index_cols, list)
    except (json.decoder.JSONDecodeError, AssertionError) as e:
        parser.error(
            f"Expected --index_cols to be a JSON list. Got {args.index_cols}.\n Caught {type(e).__name__}({e})"
        )
    else:
        args.index_cols = index_cols

    # Parse agg_dict JSON string
    try:
        agg_dict = json.loads(re.sub("'", '"', args.agg_dict)) or {}
        assert isinstance(agg_dict, dict)
    except (json.decoder.JSONDecodeError, AssertionError) as e:
        parser.error(f"Expected --agg_dict to be a JSON dict. Got {args.agg_dict}.\n Caught {type(e).__name__}({e})")
    else:
        if args.target_col not in agg_dict:
            agg_dict[args.target_col] = "sum"
        elif agg_dict[args.target_col] != "sum":
            warn(
                f'Expected the agg_dict to specify "sum" for target_col {args.target_col}, '
                f'but got {agg_dict[args.target_col]}. Manually changing to "sum".'
            )
            agg_dict[args.target_col] = "sum"
        args.agg_dict = agg_dict

    # Set default data_cols if needed & make sure target_col is in data_cols
    if args.data_cols is None:
        args.data_cols = [args.target_col]
    else:
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
    if args.target_col not in args.data_cols:
        parser.error(f"Expected --data_cols {args.data_cols} to contain --target_col {args.target_col}.")

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
        target_seq_index = {v: i for i, v in enumerate(args.data_cols)}[args.target_col]
        model["target_seq_index"] = target_seq_index
        args.model = model

    # Only do hierarchical forecasting if there are index columns specifying a hierarchy
    args.hierarchical = args.hierarchical and len(args.index_cols) > 0

    return args


def main():
    args = parse_args()

    # Read the dataset as a Spark DataFrame, and process it.
    # This will add a TSID_COL_NAME column to identify each time series with a single integer.
    spark = SparkSession.builder.appName("forecast").getOrCreate()
    df = read_dataset(
        spark=spark,
        file_format=args.file_format,
        path=args.train_data,
        time_col=args.time_col,
        index_cols=args.index_cols,
        data_cols=args.data_cols,
    )
    if args.time_col is None:
        args.time_col = df.schema.fieldNames()[0]

    # Convert to a hierarchical dataset if desired
    if args.hierarchical:
        df, hier_matrix = create_hier_dataset(
            spark=spark, df=df, time_col=args.time_col, index_cols=args.index_cols, agg_dict=args.agg_dict
        )

    # Use spark to generate forecasts for each time series in parallel
    index_fields = [df.schema[c] for c in args.index_cols + [TSID_COL_NAME]]
    pred_fields = [
        StructField(args.time_col, DateType()),
        StructField(args.target_col, FloatType()),
        StructField(f"{args.target_col}_err", FloatType()),
    ]
    output_schema = StructType(index_fields + pred_fields)
    forecast_df = df.groupBy(args.index_cols).applyInPandas(
        lambda pdf: forecast(
            pdf,
            index_cols=args.index_cols,
            time_col=args.time_col,
            target_col=args.target_col,
            time_stamps=args.time_stamps,
            model=args.model,
            predict_on_train=args.predict_on_train,
            agg_dict=args.agg_dict,
        ),
        schema=output_schema,
    )

    # Apply hierarchical time series reconciliation if desired
    if args.hierarchical:
        forecast_df = forecast_df.groupBy(args.time_col).applyInPandas(
            lambda pdf: reconciliation(pdf, hier_matrix=hier_matrix, target_col=args.target_col), schema=output_schema
        )

    write_dataset(df=forecast_df, time_col=args.time_col, path=args.output_path, file_format=args.file_format)


if __name__ == "__main__":
    main()
