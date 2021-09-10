#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
from collections import OrderedDict
import glob
import json
import logging
import math
import os
import re
import sys
import git
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm

from merlion.evaluate.forecast import ForecastEvaluator, ForecastMetric, ForecastEvaluatorConfig
from merlion.models.ensemble.combine import CombinerBase, Mean, ModelSelector, MetricWeightedMean
from merlion.models.ensemble.forecast import ForecasterEnsembleConfig, ForecasterEnsemble
from merlion.models.factory import ModelFactory
from merlion.models.forecast.base import ForecasterBase
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import granularity_str_to_seconds
from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.utils.resample import get_gcd_timedelta

from ts_datasets.base import BaseDataset
from ts_datasets.forecast import *

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

MERLION_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON = os.path.join(MERLION_ROOT, "conf", "benchmark_forecast.json")
DATADIR = os.path.join(MERLION_ROOT, "data")
OUTPUTDIR = os.path.join(MERLION_ROOT, "results", "forecast")


def parse_args():
    with open(CONFIG_JSON, "r") as f:
        valid_models = list(json.load(f).keys())

    parser = argparse.ArgumentParser(
        description="Script to benchmark various Merlion forecasting models on "
        "univariate forecasting task. This file assumes that "
        "you have pip installed both merlion (this repo's main "
        "package) and ts_datasets (a sub-repo)."
    )
    parser.add_argument(
        "--dataset",
        default="M4_Hourly",
        help="Name of dataset to run benchmark on. See get_dataset() "
        "in ts_datasets/ts_datasets/forecast/__init__.py for "
        "valid options.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Name of forecasting model to benchmark.",
        choices=valid_models,
    )
    parser.add_argument(
        "--hash",
        type=str,
        default=None,
        help="Unique identifier for the output file. Can be useful "
        "if doing multiple runs with the same model but different "
        "hyperparameters.",
    )
    parser.add_argument(
        "--ensemble_type",
        type=str,
        default="selector",
        help="How to combine multiple models in an ensemble",
        choices=["mean", "err_weighted_mean", "selector"],
    )
    parser.add_argument(
        "--retrain_type",
        type=str,
        default="without_retrain",
        help="Name of retrain type, should be one of the three "
        "types, without_retrain, sliding_window_retrain"
        "or expanding_window_retrain.",
        choices=["without_retrain", "sliding_window_retrain", "expanding_window_retrain"],
    )
    parser.add_argument("--n_retrain", type=int, default=0, help="Specify the number of retrain times.")
    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        default=False,
        help="Specify this option if you would like continue "
        "training your model on a dataset from a "
        "checkpoint, instead of restarting from scratch.",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to set logging level to debug.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Whether to plot the model's predictions after "
        "training on each example. Mutually exclusive "
        "with running any sort of evaluation.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        default=False,
        help="Specify this option if you want to summarize "
        "all results for a particular dataset. Note "
        "that this option only summarizes the results "
        "that have already been computed! It does not "
        "run any algorithms, aside from the one(s) given "
        "to --models (if any).",
    )

    args = parser.parse_args()

    # If not summarizing all results, we need at least one model to evaluate
    if args.summarize and args.models is None:
        args.models = []
    elif not args.summarize:
        if args.models is None:
            args.models = ["ARIMA"]
        elif len(args.models) == 0:
            parser.error("At least one model required if --summarize not given")

    return args


def get_dataset_name(dataset: BaseDataset) -> str:
    name = type(dataset).__name__
    if hasattr(dataset, "subset") and dataset.subset is not None:
        name += "_" + dataset.subset
    return name


def resolve_model_name(model_name: str):
    with open(CONFIG_JSON, "r") as f:
        config_dict = json.load(f)

    if model_name not in config_dict:
        raise NotImplementedError(
            f"Benchmarking not implemented for model {model_name}. Valid model names are {list(config_dict.keys())}"
        )

    while "alias" in config_dict[model_name]:
        model_name = config_dict[model_name]["alias"]

    return model_name


def get_model(model_name: str, dataset: BaseDataset, **kwargs) -> ForecasterBase:
    """Gets the model, configured for the specified dataset."""
    with open(CONFIG_JSON, "r") as f:
        config_dict = json.load(f)

    if model_name not in config_dict:
        raise NotImplementedError(
            f"Benchmarking not implemented for model {model_name}. Valid model names are {list(config_dict.keys())}"
        )

    while "alias" in config_dict[model_name]:
        model_name = config_dict[model_name]["alias"]

    # Load the model with default kwargs, but override with dataset-specific
    # kwargs where relevant, as well as manual kwargs
    model_configs = config_dict[model_name]["config"]
    model_type = config_dict[model_name].get("model_type", model_name)
    model_kwargs = model_configs["default"]
    model_kwargs.update(model_configs.get(type(dataset).__name__, {}))
    model_kwargs.update(kwargs)

    # Override the transform with Identity
    if "transform" in model_kwargs:
        logger.warning(
            f"Data pre-processing transforms currently not "
            f"supported for forecasting. Ignoring "
            f"transform {model_kwargs['transform']} and "
            f"using Identity instead."
        )
    model_kwargs["transform"] = TemporalResample(
        granularity=None, aggregation_policy="Mean", missing_value_policy="FFill"
    )

    return ModelFactory.create(name=model_type, **model_kwargs)


def get_combiner(ensemble_type: str) -> CombinerBase:
    if ensemble_type == "mean":
        return Mean(abs_score=False)
    elif ensemble_type == "selector":
        return ModelSelector(metric=ForecastMetric.sMAPE)
    elif ensemble_type == "err_weighted_mean":
        return MetricWeightedMean(metric=ForecastMetric.sMAPE)
    else:
        raise KeyError(f"ensemble_type {ensemble_type} not supported.")


def get_dirname(model_names: List[str], ensemble_type: str) -> str:
    dirname = "+".join(sorted(model_names))
    if len(model_names) > 1:
        dirname += "_" + ensemble_type
    return dirname


def train_model(
    model_names: List[str],
    dataset: BaseDataset,
    ensemble_type: str,
    csv: str,
    config_fname: str,
    retrain_type: str = "without_retrain",
    n_retrain: int = 10,
    load_checkpoint: bool = False,
    visualize: bool = False,
):
    """
    Trains all the model on the dataset, and evaluates its predictions for every
    horizon setting on every time series.
    """
    model_names = [resolve_model_name(m) for m in model_names]
    dirname = get_dirname(model_names, ensemble_type)
    dirname = dirname + "_" + retrain_type + str(n_retrain)
    results_dir = os.path.join(MERLION_ROOT, "results", "forecast", dirname)
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = get_dataset_name(dataset)

    # Determine where to start within the dataset if there is a checkpoint
    if os.path.isfile(csv) and load_checkpoint:
        i0 = pd.read_csv(csv).idx.max()
    else:
        i0 = -1
        with open(csv, "w") as f:
            f.write("idx,name,horizon,retrain_type,n_retrain,RMSE,sMAPE\n")

    model = None
    # loop over dataset

    is_multivariate_data = dataset[0][0].shape[1] > 1

    for i, (df, md) in enumerate(tqdm.tqdm(dataset, desc=f"{dataset_name} Dataset")):
        if i <= i0:
            continue
        trainval = md["trainval"]

        # Resample to an appropriate granularity according to metadata
        if "granularity" in md:
            dt = md["granularity"]
            df = df.resample(dt, closed="right", label="right").mean().interpolate()

        vals = TimeSeries.from_pd(df)
        # Get time-delta
        if not is_multivariate_data:
            dt = df.index[1] - df.index[0]
        else:
            dt = get_gcd_timedelta(vals.time_stamps)
            dt = pd.to_timedelta(dt, unit="s")

        # Get the train/val split
        t = trainval.index[np.argmax(~trainval)].value // 1e9
        train_vals, test_vals = vals.bisect(t, t_in_left=False)

        # Compute train_window_len and test_window_len
        train_start_timestamp = train_vals.univariates[train_vals.names[0]].time_stamps[0]
        test_start_timestamp = test_vals.univariates[test_vals.names[0]].time_stamps[0]
        train_window_len = test_start_timestamp - train_start_timestamp

        train_end_timestamp = train_vals.univariates[train_vals.names[0]].time_stamps[-1]
        test_end_timestamp = test_vals.univariates[test_vals.names[0]].time_stamps[-1]
        test_window_len = test_end_timestamp - train_end_timestamp

        # Get all the horizon conditions we want to evaluate from metadata
        if any("condition" in k and isinstance(v, list) for k, v in md.items()):
            conditions = sum([v for k, v in md.items() if "condition" in k and isinstance(v, list)], [])
            logger.debug("\n" + "=" * 80 + "\n" + df.columns[0] + "\n" + "=" * 80 + "\n")
            horizons = set()
            for condition in conditions:
                horizons.update([v for k, v in condition.items() if "horizon" in k])

        # For multivariate data, we use a horizon of 3
        elif is_multivariate_data:
            horizons = [3 * dt]

        # For univariate data, we predict the entire test data in batch
        else:
            horizons = [test_window_len]

        # loop over horizon conditions
        for horizon in horizons:
            horizon = granularity_str_to_seconds(horizon)
            max_forecast_steps = math.ceil(horizon / dt.total_seconds())
            logger.debug(f"horizon is {pd.Timedelta(seconds=horizon)} and max_forecast_steps is {max_forecast_steps}")
            if retrain_type == "without_retrain":
                retrain_freq = None
                train_window = None
                n_retrain = 0
            elif retrain_type == "sliding_window_retrain":
                retrain_freq = math.ceil(test_window_len / int(n_retrain))
                train_window = train_window_len
            elif retrain_type == "expanding_window_retrain":
                retrain_freq = math.ceil(test_window_len / int(n_retrain))
                train_window = None
            else:
                raise ValueError(
                    "the retrain_type should be without_retrain, sliding_window_retrain or expanding_window_retrain"
                )

            # Get Model
            models = [get_model(m, dataset, max_forecast_steps=max_forecast_steps) for m in model_names]
            if len(models) == 1:
                model = models[0]
            else:
                config = ForecasterEnsembleConfig(combiner=get_combiner(ensemble_type))
                model = ForecasterEnsemble(config=config, models=models)

            evaluator = ForecastEvaluator(
                model=model,
                config=ForecastEvaluatorConfig(train_window=train_window, horizon=horizon, retrain_freq=retrain_freq),
            )

            # Initialize train config
            train_kwargs = {}
            if type(model).__name__ == "AutoSarima":
                train_kwargs = {"train_config": {"enforce_stationarity": True, "enforce_invertibility": True}}

            # Get Evaluate Results
            train_result, test_pred = evaluator.get_predict(
                train_vals=train_vals, test_vals=test_vals, train_kwargs=train_kwargs, retrain_kwargs=train_kwargs
            )

            rmses = evaluator.evaluate(ground_truth=test_vals, predict=test_pred, metric=ForecastMetric.RMSE)
            smapes = evaluator.evaluate(ground_truth=test_vals, predict=test_pred, metric=ForecastMetric.sMAPE)

            # Log relevant info to the CSV
            with open(csv, "a") as f:
                f.write(f"{i},{df.columns[0]},{horizon},{retrain_type},{n_retrain},{rmses},{smapes}\n")

            # generate comparison plot
            if visualize:
                name = train_vals.names[0]
                train_time_stamps = train_vals.univariates[name].time_stamps
                fig_dir = os.path.join(results_dir, dataset_name + "_figs")
                os.makedirs(fig_dir, exist_ok=True)
                fig_dataset_dir = os.path.join(fig_dir, df.columns[0])
                os.makedirs(fig_dataset_dir, exist_ok=True)
                if train_result[0] is not None:
                    train_pred = train_result[0]
                else:
                    train_pred = TimeSeries({name: UnivariateTimeSeries(train_time_stamps, None)})
                fig_name = dirname + "_" + retrain_type + str(n_retrain) + "_" + "horizon" + str(int(horizon)) + ".png"
                plot_unrolled_compare(
                    train_vals,
                    test_vals,
                    train_pred,
                    test_pred,
                    os.path.join(fig_dataset_dir, fig_name),
                    dirname + f"(sMAPE={smapes:.4f})",
                )

            # Log relevant info to the logger
            logger.debug(f"{dirname} {retrain_type} {n_retrain} sMAPE : {smapes:.4f}\n")

    # Save full experimental config
    if model is not None:
        full_config = dict(
            model_config=model.config.to_dict(),
            evaluator_config=evaluator.config.to_dict(),
            code_version_info=get_code_version_info(),
        )

        with open(config_fname, "w") as f:
            json.dump(full_config, f, indent=2, sort_keys=True)


def get_code_version_info():
    return dict(time=str(pd.Timestamp.now()), commit=git.Repo(search_parent_directories=True).head.object.hexsha)


def plot_unrolled_compare(train_vals, test_vals, train_pred, test_pred, outputpath, title):
    truth_pd = (train_vals + test_vals).to_pd()
    truth_pd.columns = ["ground_truth"]
    pred_pd = (train_pred + test_pred).to_pd()
    pred_pd.columns = ["prediction"]
    result_pd = pd.concat([truth_pd, pred_pd], axis=1)
    plt.figure()
    plt.rcParams["savefig.dpi"] = 500
    plt.rcParams["figure.dpi"] = 500
    result_pd.plot(linewidth=0.5)
    plt.axvline(train_vals.to_pd().index[-1], color="r")
    plt.title(title)
    plt.savefig(outputpath)
    plt.clf()


def join_dfs(name2df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Joins multiple results dataframes into a single dataframe describing the
    results from all models.
    """
    full_df, lsuffix = None, ""
    shared_cols = ["idx", "name", "horizon", "retrain_type", "n_retrain"]
    for name, df in name2df.items():
        df.columns = [c if c in shared_cols else f"{c}_{name}" for c in df.columns]
        if full_df is None:
            full_df = df
        else:
            full_df = full_df.merge(df, how="outer", left_on=shared_cols, right_on=shared_cols)
    unique_cols = [c for c in full_df.columns if c not in shared_cols]
    return full_df[shared_cols + unique_cols]


def summarize_full_df(full_df: pd.DataFrame) -> pd.DataFrame:
    # Get the names of all algorithms which have full results
    algs = [col[len("sMAPE") :] for col in full_df.columns if col.startswith("sMAPE") and not full_df[col].isna().any()]

    summary_df = pd.DataFrame({alg.lstrip("_"): [] for alg in algs})

    # Compute pooled (per time series) mean/median sMAPE, RMSE
    mean_smape, med_smape, mean_rmse, med_rmse = [[] for _ in range(4)]

    for ts_name in np.unique(full_df.name):
        ts = full_df[full_df.name == ts_name]
        # append smape
        smapes = ts[[f"sMAPE{alg}" for alg in algs]]
        mean_smape.append(smapes.mean(axis=0).values)
        med_smape.append(smapes.median(axis=0).values)
        # append rmse
        rmses = ts[[f"RMSE{alg}" for alg in algs]]
        mean_rmse.append(rmses.mean(axis=0).values)
        med_rmse.append(rmses.median(axis=0).values)

    # Add mean/median loglifts to the summary dataframe
    summary_df.loc["mean_sMAPE"] = np.mean(mean_smape, axis=0)
    summary_df.loc["median_sMAPE"] = np.median(med_smape, axis=0)
    summary_df.loc["mean_RMSE"] = np.mean(mean_rmse, axis=0)
    summary_df.loc["median_RMSE"] = np.median(med_rmse, axis=0)
    return summary_df


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    dataset = get_dataset(args.dataset)
    dataset_name = get_dataset_name(dataset)

    if len(args.models) > 0:
        # Determine the name of the results CSV
        model_names = [resolve_model_name(m) for m in args.models]
        dirname = get_dirname(model_names, args.ensemble_type)
        dirname = dirname + "_" + args.retrain_type + str(args.n_retrain)
        results_dir = os.path.join(MERLION_ROOT, "results", "forecast", dirname)
        basename = dataset_name
        if args.hash is not None:
            basename += "_" + args.hash
        config_fname = f"{dataset_name}_config"
        csv = os.path.join(results_dir, f"{basename}.csv")
        config_fname = os.path.join(results_dir, f"{config_fname}.json")

        train_model(
            model_names=args.models,
            dataset=dataset,
            ensemble_type=args.ensemble_type,
            retrain_type=args.retrain_type,
            n_retrain=args.n_retrain,
            csv=csv,
            config_fname=config_fname,
            load_checkpoint=args.load_checkpoint,
            visualize=args.visualize,
        )

        # Pool the mean/medium sMAPE, RMSE for all evaluation
        # settings for each time series, and report summary
        # pooled statistics.
        df = pd.read_csv(csv)
        summary = summarize_full_df(df)
        summary.to_csv(os.path.join(results_dir, f"{basename}_summary.csv"), index=True)
        summary = summary[summary.columns[0]]
        logger.info(f"Pooled mean   sMAPE: {summary['mean_sMAPE']:.4f}")
        logger.info(f"Pooled median sMAPE: {summary['median_sMAPE']:.4f}")
        logger.info(f"Pooled mean   RMSE: {summary['mean_RMSE']:.4f}")
        logger.info(f"Pooled median RMSE: {summary['median_RMSE']:.4f}")

    # Now we summarize all results. Get all the individual CSV's as dataframes
    name2df = OrderedDict()
    prefix = f"{MERLION_ROOT}/results/forecast/*/{dataset_name}"
    csvs = glob.glob(f"{prefix}.csv") + glob.glob(f"{prefix}_*.csv")
    csvs = [c for c in csvs if not c.endswith(f"_summary.csv")]
    if len(csvs) == 0:
        raise RuntimeError(
            f"Did not find any pre-computed results files "
            f"for dataset {dataset_name}. Please run this "
            f"script on the dataset with specific algorithms "
            f"before trying to summarize their results."
        )
    for csv in sorted(csvs):
        model_name = os.path.basename(os.path.dirname(csv))
        suffix = re.search(f"(?<={dataset_name}).*(?=\\.csv)", os.path.basename(csv)).group(0)
        try:
            name2df[model_name + suffix] = pd.read_csv(csv)
        except Exception as e:
            logger.warning(f'Caught {type(e).__name__}: "{e}". Skipping csv file {csv}.')
            continue

    # Join all the dataframes into one
    dirname = os.path.join(MERLION_ROOT, "results", "forecast")
    full_df = join_dfs(name2df)
    full_df.to_csv(os.path.join(dirname, f"{dataset_name}_full.csv"), index=False)

    # Summarize the joined dataframe
    summary_df = summarize_full_df(full_df)
    summary_df.to_csv(os.path.join(dirname, f"{dataset_name}_summary.csv"), index=True)
    if args.summarize:
        print(summary_df)


if __name__ == "__main__":
    main()
