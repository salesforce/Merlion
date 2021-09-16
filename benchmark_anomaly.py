#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import argparse
import copy
import json
import logging
import os
import sys
import time
import git
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from merlion.evaluate.anomaly import (
    TSADEvaluatorConfig,
    accumulate_tsad_score,
    TSADScoreAccumulator as ScoreAcc,
    TSADEvaluator,
)
from merlion.models.anomaly.base import DetectorBase
from merlion.models.ensemble.anomaly import DetectorEnsemble
from merlion.evaluate.anomaly import TSADMetric, ScoreType
from merlion.models.factory import ModelFactory
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries
from merlion.utils.resample import to_pd_datetime

from ts_datasets.anomaly import *

logger = logging.getLogger(__name__)

# Benchmark code assumes you have created data/<dirname> symlinks to
# the root directories of all the relevant datasets
MERLION_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON = os.path.join(MERLION_ROOT, "conf", "benchmark_anomaly.json")
DATADIR = os.path.join(MERLION_ROOT, "data")


def parse_args():
    with open(CONFIG_JSON, "r") as f:
        valid_models = list(json.load(f).keys())

    parser = argparse.ArgumentParser(
        description="Script to benchmark Merlion time series anomaly detection "
        "models. This script assumes that you have pip installed "
        "both merlion (this repo's main package) and ts_datasets "
        "(a sub-repo)."
    )
    parser.add_argument(
        "--dataset",
        default="NAB_all",
        help="Name of dataset to run benchmark on. See get_dataset() "
        "in ts_datasets/ts_datasets/anomaly/__init__.py for "
        "valid options.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["DefaultDetector"],
        help="Name of model (or models in ensemble) to benchmark.",
        choices=valid_models,
    )
    parser.add_argument(
        "--retrain_freq",
        type=str,
        default="default",
        help="String (e.g. 1d, 2w, etc.) specifying how often "
        "to re-train the model before evaluating it on "
        "the next window of data. Note that re-training "
        "is unsupervised, i.e. does not use ground truth "
        "anomaly labels in any way. Default retrain_freq is "
        "1d for univariate data and None for multivariate.",
    )
    parser.add_argument(
        "--train_window",
        type=str,
        default=None,
        help="String (e.g. 30d, 6m, etc.) specifying how much "
        "data (in terms of a time window) the model "
        "should train on at any point.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="F1",
        choices=list(TSADMetric.__members__.keys()),
        help="Metric to optimize for (where relevant)",
    )
    parser.add_argument(
        "--point_adj_metric",
        type=str,
        default="PointAdjustedF1",
        choices=list(TSADMetric.__members__.keys()),
        help="Final metric to optimize for when evaluating point-adjusted performance",
    )
    parser.add_argument(
        "--pointwise_metric",
        type=str,
        default="PointwiseF1",
        choices=list(TSADMetric.__members__.keys()),
        help="Final metric to optimize for when evaluating pointwise performance",
    )
    parser.add_argument("--unsupervised", action="store_true")
    parser.add_argument(
        "--tune_on_test",
        action="store_true",
        default=False,
        help="Whether to tune the threshold on both train and "
        "test splits of the time series. Useful for "
        "metrics like Best F1, or NAB score with "
        "threshold optimization.",
    )
    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        default=False,
        help="Specify this option if you would like continue "
        "training your model on a dataset from a "
        "checkpoint, instead of restarting from scratch.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Specify this option if you would like to skip "
        "the model training phase, and simply evaluate "
        "on partial saved results.",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to enable INFO-level logs.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Whether to plot the model's predictions after "
        "training on each example. Mutually exclusive "
        "with running any sort of evaluation.",
    )
    args = parser.parse_args()
    args.metric = TSADMetric[args.metric]
    args.pointwise_metric = TSADMetric[args.pointwise_metric]
    args.visualize = args.visualize and not args.eval_only
    if args.retrain_freq.lower() in ["", "none", "null"]:
        args.retrain_freq = None
    elif args.retrain_freq != "default":
        rf = pd.to_timedelta(args.retrain_freq).total_seconds()
        if rf % (3600 * 24) == 0:
            args.retrain_freq = f"{int(rf/3600/24)}d"
        elif rf % 3600 == 0:
            args.retrain_freq = f"{int(rf/3600)}h"
        elif rf % 60 == 0:
            args.retrain_freq = f"{int(rf//60)}min"
        else:
            args.retrain_freq = f"{int(rf)}s"

    return args


def dataset_to_name(dataset: TSADBaseDataset):
    if dataset.subset is not None:
        return f"{type(dataset).__name__}_{dataset.subset}"
    return type(dataset).__name__


def dataset_to_threshold(dataset: TSADBaseDataset, tune_on_test=False):
    if isinstance(dataset, IOpsCompetition):
        return 2.25
    elif isinstance(dataset, NAB):
        return 3.5
    elif isinstance(dataset, Synthetic):
        return 2
    elif isinstance(dataset, MSL):
        return 3.0
    elif isinstance(dataset, SMAP):
        return 3.5
    elif isinstance(dataset, SMD):
        return 3 if not tune_on_test else 2.5
    elif hasattr(dataset, "default_threshold"):
        return dataset.default_threshold
    return 3


def resolve_model_name(model_name: str):
    with open(CONFIG_JSON, "r") as f:
        config_dict = json.load(f)

    if model_name not in config_dict:
        raise NotImplementedError(
            f"Benchmarking not implemented for model {model_name}. Valid model names are {list(config_dict.keys())}"
        )

    while "alias" in config_dict[model_name]:
        assert model_name != config_dict[model_name]["alias"], "Alias name cannot be the same as the model name"
        model_name = config_dict[model_name]["alias"]

    return model_name


def get_model(
    model_name: str, dataset: TSADBaseDataset, metric: TSADMetric, tune_on_test=False, unsupervised=False
) -> Tuple[DetectorBase, dict]:
    with open(CONFIG_JSON, "r") as f:
        config_dict = json.load(f)

    if model_name not in config_dict:
        raise NotImplementedError(
            f"Benchmarking not implemented for model {model_name}. Valid model names are {list(config_dict.keys())}"
        )

    while "alias" in config_dict[model_name]:
        model_name = config_dict[model_name]["alias"]

    # Load the model with default kwargs, but override with dataset-specific
    # kwargs where relevant
    model_configs = config_dict[model_name]["config"]
    model_type = config_dict[model_name].get("model_type", model_name)
    model_kwargs = model_configs["default"]
    model_kwargs.update(model_configs.get(type(dataset).__name__, {}))
    model = ModelFactory.create(name=model_type, **model_kwargs)

    # The post-rule train configs are fully specified for each dataset (where
    # relevant), with a default option if there is no dataset-specific option.
    post_rule_train_configs = config_dict[model_name].get("post_rule_train_config", {})
    d = post_rule_train_configs.get("default", {})
    d.update(post_rule_train_configs.get(type(dataset).__name__, {}))
    if len(d) == 0:
        d = copy.copy(model._default_post_rule_train_config)
    d["metric"] = None if unsupervised else metric
    d.update({"max_early_sec": dataset.max_lead_sec, "max_delay_sec": dataset.max_lag_sec})

    t = dataset_to_threshold(dataset, tune_on_test)
    model.threshold.alm_threshold = t
    d["unsup_quantile"] = None
    return model, d


def df_to_merlion(df: pd.DataFrame, md: pd.DataFrame, get_ground_truth=False, transform=None) -> TimeSeries:
    """Converts a pandas dataframe time series to the Merlion format."""
    if get_ground_truth:
        if False and "changepoint" in md.keys():
            series = md["anomaly"] | md["changepoint"]
        else:
            series = md["anomaly"]
    else:
        series = df
    time_series = TimeSeries.from_pd(series)
    if transform is not None:
        time_series = transform(time_series)
    return time_series


def train_model(
    model_name,
    metric,
    dataset,
    retrain_freq=None,
    train_window=None,
    load_checkpoint=False,
    visualize=False,
    debug=False,
    unsupervised=False,
    tune_on_test=False,
):
    """Trains a model on the time series dataset given, and save their predictions
    to a dataset."""
    resampler = None
    if isinstance(dataset, IOpsCompetition):
        resampler = TemporalResample("5min")

    model_name = resolve_model_name(model_name)
    dataset_name = dataset_to_name(dataset)
    model_dir = model_name if retrain_freq is None else f"{model_name}_{retrain_freq}"
    dirname = os.path.join("results", "anomaly", model_dir)
    csv = os.path.join(dirname, f"pred_{dataset_name}.csv.gz")
    config_fname = os.path.join(dirname, f"{dataset_name}_config.json")
    checkpoint = os.path.join(dirname, f"ckpt_{dataset_name}.txt")

    # Determine where to start within the dataset if there is a checkpoint
    i0 = 0
    if os.path.isfile(checkpoint) and os.path.isfile(csv) and load_checkpoint:
        with open(checkpoint, "r") as f:
            i0 = int(f.read().rstrip("\n"))

        # Validate & sanitize the existing CSV checkpoint
        df = pd.read_csv(csv, dtype={"trainval": bool, "idx": int})
        df = df[df["idx"] < i0]
        if set(df["idx"]) == set(range(i0)):
            df.to_csv(csv, index=False)
        else:
            i0 = 0

    model = None
    for i, (df, md) in enumerate(tqdm(dataset)):
        if i < i0:
            continue

        # Reload model & get the train / test split for this time series
        model, post_rule_train_config = get_model(
            model_name=model_name, dataset=dataset, metric=metric, tune_on_test=tune_on_test, unsupervised=unsupervised
        )
        delay = post_rule_train_config["max_early_sec"]
        train_vals = df_to_merlion(df[md.trainval], md[md.trainval], get_ground_truth=False, transform=resampler)
        test_vals = df_to_merlion(df[~md.trainval], md[~md.trainval], get_ground_truth=False, transform=resampler)
        train_anom = df_to_merlion(df[md.trainval], md[md.trainval], get_ground_truth=True)
        test_anom = df_to_merlion(df[~md.trainval], md[~md.trainval], get_ground_truth=True)

        # Set up an evaluator & get predictions
        evaluator = TSADEvaluator(
            model=model,
            config=TSADEvaluatorConfig(
                train_window=train_window,
                retrain_freq=retrain_freq,
                max_delay_sec=delay,
                max_early_sec=getattr(model.threshold, "suppress_secs", delay),
            ),
        )
        train_scores, test_scores = evaluator.get_predict(
            train_vals=train_vals,
            test_vals=test_vals,
            post_process=False,
            train_kwargs={"anomaly_labels": train_anom, "post_rule_train_config": post_rule_train_config},
        )

        # Write the model's predictions to the csv file, starting a new one
        # if we aren't loading an existing checkpoint. Scores from all time
        # series in the dataset are combined together in a single csv. Each
        # line in the csv corresponds to a point in a time series, and contains
        # the timestamp, raw anomaly score, and index of the time series.
        if not visualize:
            if i == i0 == 0:
                os.makedirs(os.path.dirname(csv), exist_ok=True)
                df = pd.DataFrame({"timestamp": [], "y": [], "trainval": [], "idx": []})
                df.to_csv(csv, index=False)

            df = pd.read_csv(csv)
            ts_df = train_scores.to_pd().append(test_scores.to_pd())
            ts_df.columns = ["y"]
            ts_df.loc[:, "timestamp"] = ts_df.index.view(int) // 1e9
            ts_df.loc[:, "trainval"] = [j < len(train_scores) for j in range(len(ts_df))]
            ts_df.loc[:, "idx"] = i
            df = df.append(ts_df, ignore_index=True)
            df.to_csv(csv, index=False)

            # Start from time series i+1 if loading a checkpoint.
            with open(checkpoint, "w") as f:
                f.write(str(i + 1))

        if visualize or debug:
            # Train the post-rule on the appropriate labels
            score = test_scores if tune_on_test else train_scores
            label = test_anom if tune_on_test else train_anom
            model.train_post_rule(
                anomaly_scores=score, anomaly_labels=label, post_rule_train_config=post_rule_train_config
            )

            # Log (many) evaluation metrics for the time series
            score_acc = evaluator.evaluate(ground_truth=test_anom, predict=model.threshold(test_scores))
            mttd = score_acc.mean_time_to_detect()
            if mttd < pd.to_timedelta(0):
                mttd = f"-{-mttd}"
            logger.info(f"\nPerformance on time series {i+1}/{len(dataset)}")
            logger.info("Revised Point-Adjusted Metrics")
            logger.info(f"F1 Score:  {score_acc.f1(score_type=ScoreType.RevisedPointAdjusted):.4f}")
            logger.info(f"Precision: {score_acc.precision(score_type=ScoreType.RevisedPointAdjusted):.4f}")
            logger.info(f"Recall:    {score_acc.recall(score_type=ScoreType.RevisedPointAdjusted):.4f}\n")
            logger.info(f"Mean Time To Detect Anomalies:  {mttd}")
            logger.info(f"Mean Detected Anomaly Duration: {score_acc.mean_detected_anomaly_duration()}")
            logger.info(f"Mean Anomaly Duration:          {score_acc.mean_anomaly_duration()}\n")

            if debug:
                logger.info(f"Pointwise metrics")
                logger.info(f"F1 Score:  {score_acc.f1(score_type=ScoreType.Pointwise):.4f}")
                logger.info(f"Precision: {score_acc.precision(score_type=ScoreType.Pointwise):.4f}")
                logger.info(f"Recall:    {score_acc.recall(score_type=ScoreType.Pointwise):.4f}\n")

                logger.info("Point-Adjusted Metrics")
                logger.info(f"F1 Score:  {score_acc.f1(score_type=ScoreType.PointAdjusted):.4f}")
                logger.info(f"Precision: {score_acc.precision(score_type=ScoreType.PointAdjusted):.4f}")
                logger.info(f"Recall:    {score_acc.recall(score_type=ScoreType.PointAdjusted):.4f}\n")

                logger.info(f"NAB Scores")
                logger.info(f"NAB score (balanced): {score_acc.nab_score():.4f}")
                logger.info(f"NAB score (low FP):   {score_acc.nab_score(fp_weight=0.22):.4f}")
                logger.info(f"NAB score (low FN):   {score_acc.nab_score(fn_weight=2.0):.4f}\n")

            if visualize:
                # Make a plot
                alarms = model.threshold(test_scores)
                fig = model.get_figure(time_series=test_vals, time_series_prev=train_vals, plot_time_series_prev=True)
                fig.anom = alarms.univariates[alarms.names[0]]
                fig, ax = fig.plot(figsize=(1800, 600))

                # Overlay windows indicating the true anomalies
                all_anom = train_anom + test_anom
                t, y = zip(*all_anom)
                y = np.asarray(y).flatten()
                splits = np.where(y[1:] != y[:-1])[0] + 1
                splits = np.concatenate(([0], splits, [len(y) - 1]))
                anom_windows = [(splits[k], splits[k + 1]) for k in range(len(splits) - 1) if y[splits[k]]]
                for i_0, i_f in anom_windows:
                    t_0 = to_pd_datetime(t[i_0])
                    t_f = to_pd_datetime(t[i_f])
                    ax.axvspan(t_0, t_f, color="#d07070", zorder=-1, alpha=0.5)
                time.sleep(2)
                plt.show()

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


def read_model_predictions(dataset: TSADBaseDataset, model_dir: str):
    """
    Returns a list of lists all_preds, where all_preds[i] is the model's raw
    anomaly scores for time series i in the dataset.
    """
    csv = os.path.join("results", "anomaly", model_dir, f"pred_{dataset_to_name(dataset)}.csv.gz")
    preds = pd.read_csv(csv, dtype={"trainval": bool, "idx": int})
    preds["timestamp"] = to_pd_datetime(preds["timestamp"])
    return [preds[preds["idx"] == i].set_index("timestamp") for i in sorted(preds["idx"].unique())]


def evaluate_predictions(
    model_names,
    dataset,
    all_model_preds,
    metric: TSADMetric,
    pointwise_metric: TSADMetric,
    point_adj_metric: TSADMetric,
    tune_on_test=False,
    unsupervised=False,
    debug=False,
):

    scores_rpa, scores_pw, scores_pa = [], [], []
    use_ucr_eval = isinstance(dataset, UCR) and (unsupervised or not tune_on_test)
    for i, (true, md) in enumerate(tqdm(dataset)):
        # Get time series for the train & test splits of the ground truth
        idx = ~md.trainval if tune_on_test else md.trainval
        true_train = df_to_merlion(true[idx], md[idx], get_ground_truth=True)
        true_test = df_to_merlion(true[~md.trainval], md[~md.trainval], get_ground_truth=True)

        for acc_id, (simple_threshold, opt_metric, scores) in enumerate(
            [
                (use_ucr_eval and not tune_on_test, metric, scores_rpa),
                (True, pointwise_metric, scores_pw),
                (True, point_adj_metric, scores_pa),
            ]
        ):
            if acc_id > 0 and use_ucr_eval:
                scores_pw = scores_rpa
                scores_pa = scores_rpa
                continue
            # For each model, load its raw anomaly scores for the i'th time series
            # as a UnivariateTimeSeries, and collect all the models' scores as a
            # TimeSeries. Do this for both the train and test splits.
            if i >= min(len(p) for p in all_model_preds):
                break
            pred = [model_preds[i] for model_preds in all_model_preds]
            pred_train = [p[~p["trainval"]] if tune_on_test else p[p["trainval"]] for p in pred]
            pred_train = [TimeSeries.from_pd(p["y"]) for p in pred_train]
            pred_test = [p[~p["trainval"]] for p in pred]
            pred_test = [TimeSeries.from_pd(p["y"]) for p in pred_test]

            # Train each model's post rule on the train split
            models = []
            for name, train, og_pred in zip(model_names, pred_train, pred):
                m, prtc = get_model(
                    model_name=name,
                    dataset=dataset,
                    metric=opt_metric,
                    tune_on_test=tune_on_test,
                    unsupervised=unsupervised,
                )
                m.config.enable_threshold = len(model_names) == 1
                if simple_threshold:
                    m.threshold = m.threshold.to_simple_threshold()
                if tune_on_test and not unsupervised:
                    m.calibrator.train(TimeSeries.from_pd(og_pred["y"][og_pred["trainval"]]))
                m.train_post_rule(anomaly_scores=train, anomaly_labels=true_train, post_rule_train_config=prtc)
                models.append(m)

            # Get the lead & lag time for the dataset
            early, delay = dataset.max_lead_sec, dataset.max_lag_sec
            if early is None:
                leads = [getattr(m.threshold, "suppress_secs", delay) for m in models]
                leads = [dt for dt in leads if dt is not None]
                early = None if len(leads) == 0 else max(leads)

            # No further training if we only have 1 model
            if len(models) == 1:
                model = models[0]
                pred_test_raw = pred_test[0]

            # If we have multiple models, train an ensemble model
            else:
                threshold = dataset_to_threshold(dataset, tune_on_test)
                ensemble_threshold_train_config = dict(
                    metric=opt_metric if tune_on_test else None,
                    max_early_sec=early,
                    max_delay_sec=delay,
                    unsup_quantile=None,
                )

                # Train the ensemble and its post-rule on the current time series
                model = DetectorEnsemble(models=models)
                use_m = [len(p) > 1 for p in zip(models, pred_train)]
                pred_train = [m.post_rule(p) for m, p, use in zip(models, pred_train, use_m) if use]
                pred_test = [m.post_rule(p) for m, p, use in zip(models, pred_test, use_m) if use]
                pred_train = model.train_combiner(pred_train, true_train)
                if simple_threshold:
                    model.threshold = model.threshold.to_simple_threshold()
                model.threshold.alm_threshold = threshold
                model.train_post_rule(pred_train, true_train, ensemble_threshold_train_config)
                pred_test_raw = model.combiner(pred_test, true_test)

            # For UCR dataset, the evaluation just checks whether the point with the highest
            # anomaly score is anomalous or not.
            if acc_id == 0 and use_ucr_eval and not unsupervised:
                df = pred_test_raw.to_pd()
                df[np.abs(df) < df.max()] = 0
                pred_test = TimeSeries.from_pd(df)
            else:
                pred_test = model.post_rule(pred_test_raw)

            # Compute the individual components comprising various scores.
            score = accumulate_tsad_score(true_test, pred_test, max_early_sec=early, max_delay_sec=delay)

            # Make sure all time series have exactly one detection for UCR dataset (either 1 TP, or 1 FN & 1 FP).
            if acc_id == 0 and use_ucr_eval:
                n_anom = score.num_tp_anom + score.num_fn_anom
                if n_anom == 0:
                    score.num_tp_anom, score.num_fn_anom, score.num_fp = 0, 0, 0
                elif score.num_tp_anom > 0:
                    score.num_tp_anom, score.num_fn_anom, score.num_fp = 1, 0, 0
                else:
                    score.num_tp_anom, score.num_fn_anom, score.num_fp = 0, 1, 1
            scores.append(score)

    # Aggregate statistics from full dataset
    score_rpa = sum(scores_rpa, ScoreAcc())
    score_pw = sum(scores_pw, ScoreAcc())
    score_pa = sum(scores_pa, ScoreAcc())

    # Determine if it's better to have all negatives for each time series if
    # using the test data in a supervised way.
    if tune_on_test and not unsupervised:
        # Convert true positives to false negatives, and remove all false positives.
        # Keep the updated version if it improves F1 score.
        for s in sorted(scores_rpa, key=lambda x: x.num_fp, reverse=True):
            stype = ScoreType.RevisedPointAdjusted
            sprime = copy.deepcopy(score_rpa)
            sprime.num_tp_anom -= s.num_tp_anom
            sprime.num_fn_anom += s.num_tp_anom
            sprime.num_fp -= s.num_fp
            sprime.tp_score -= s.tp_score
            sprime.fp_score -= s.fp_score
            if score_rpa.f1(stype) < sprime.f1(stype):
                # Update anomaly durations
                for duration, delay in zip(s.tp_anom_durations, s.tp_detection_delays):
                    sprime.tp_anom_durations.remove(duration)
                    sprime.tp_detection_delays.remove(delay)
                score_rpa = sprime

        # Repeat for pointwise scores
        for s in sorted(scores_pw, key=lambda x: x.num_fp, reverse=True):
            stype = ScoreType.Pointwise
            sprime = copy.deepcopy(score_pw)
            sprime.num_tp_pointwise -= s.num_tp_pointwise
            sprime.num_fn_pointwise += s.num_tp_pointwise
            sprime.num_fp -= s.num_fp
            if score_pw.f1(stype) < sprime.f1(stype):
                score_pw = sprime

        # Repeat for point-adjusted scores
        for s in sorted(scores_pa, key=lambda x: x.num_fp, reverse=True):
            stype = ScoreType.PointAdjusted
            sprime = copy.deepcopy(score_pa)
            sprime.num_tp_point_adj -= s.num_tp_point_adj
            sprime.num_fn_point_adj += s.num_tp_point_adj
            sprime.num_fp -= s.num_fp
            if score_pa.f1(stype) < sprime.f1(stype):
                score_pa = sprime

    # Compute MTTD & report F1, precision, and recall
    mttd = score_rpa.mean_time_to_detect()
    if mttd < pd.to_timedelta(0):
        mttd = f"-{-mttd}"
    print()
    print("Revised point-adjusted metrics")
    print(f"F1 score:  {score_rpa.f1(ScoreType.RevisedPointAdjusted):.4f}")
    print(f"Precision: {score_rpa.precision(ScoreType.RevisedPointAdjusted):.4f}")
    print(f"Recall:    {score_rpa.recall(ScoreType.RevisedPointAdjusted):.4f}")
    print()
    print(f"Mean Time To Detect Anomalies:  {mttd}")
    print(f"Mean Detected Anomaly Duration: {score_rpa.mean_detected_anomaly_duration()}")
    print(f"Mean Anomaly Duration:          {score_rpa.mean_anomaly_duration()}")
    print()
    if debug:
        print("Pointwise metrics")
        print(f"F1 score:  {score_pw.f1(ScoreType.Pointwise):.4f}")
        print(f"Precision: {score_pw.precision(ScoreType.Pointwise):.4f}")
        print(f"Recall:    {score_pw.recall(ScoreType.Pointwise):.4f}")
        print()
        print("Point-adjusted metrics")
        print(f"F1 score:  {score_pa.f1(ScoreType.PointAdjusted):.4f}")
        print(f"Precision: {score_pa.precision(ScoreType.PointAdjusted):.4f}")
        print(f"Recall:    {score_pa.recall(ScoreType.PointAdjusted):.4f}")
        print()
        print("NAB Scores")
        print(f"NAB Score (balanced):       {score_rpa.nab_score():.4f}")
        print(f"NAB Score (high precision): {score_rpa.nab_score(fp_weight=0.22):.4f}")
        print(f"NAB Score (high recall):    {score_rpa.nab_score(fn_weight=2.0):.4f}")
        print()

    return score_rpa, score_pw, score_pa


def main():
    args = parse_args()
    level = logging.INFO if args.debug or args.visualize else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=level
    )
    dataset = get_dataset(args.dataset)
    retrain_freq, train_window = args.retrain_freq, args.train_window
    univariate = dataset[0][0].shape[1] == 1
    if retrain_freq == "default":
        retrain_freq = "1d" if univariate else None
        desc = "univariate" if univariate else "multivariate"
        logger.warning(f"Setting retrain_freq = {retrain_freq} for {desc} dataset {type(dataset).__name__}")

    for model_name in args.models:
        if not args.eval_only:
            print(f"Training model {model_name}...")
            train_model(
                model_name=model_name,
                dataset=dataset,
                metric=args.metric,
                tune_on_test=args.tune_on_test,
                unsupervised=args.unsupervised,
                debug=args.debug,
                visualize=args.visualize,
                load_checkpoint=args.load_checkpoint,
                retrain_freq=retrain_freq,
                train_window=train_window,
            )

    # Read in & evaluate the models' predictions
    if args.visualize:
        logger.info("Skipping evaluation because --visualize flag was given.")
    else:
        model_names = [resolve_model_name(name) for name in args.models]
        model_dirs = [name if retrain_freq is None else f"{name}_{retrain_freq}" for name in model_names]
        all_model_preds = [read_model_predictions(dataset=dataset, model_dir=model_dir) for model_dir in model_dirs]
        score_acc, pw_score_acc, pa_score_acc = evaluate_predictions(
            model_names=args.models,
            dataset=dataset,
            all_model_preds=all_model_preds,
            debug=args.debug,
            metric=args.metric,
            point_adj_metric=args.point_adj_metric,
            pointwise_metric=args.pointwise_metric,
            tune_on_test=args.tune_on_test,
            unsupervised=args.unsupervised,
        )

        model_name = "+".join(sorted(resolve_model_name(m) for m in args.models))
        summary = os.path.join("results", "anomaly", f"{dataset_to_name(dataset)}_summary.csv")
        if os.path.exists(summary):
            df = pd.read_csv(summary, index_col=0)
        else:
            df = pd.DataFrame()
        if retrain_freq:
            model_name += f"_{retrain_freq}"
        if args.unsupervised:
            model_name += " (Unsupervised)"
        if args.tune_on_test:
            model_name += " (Use Test Data)"
        df.loc[model_name, "Precision"] = score_acc.precision(ScoreType.RevisedPointAdjusted)
        df.loc[model_name, "Recall"] = score_acc.recall(ScoreType.RevisedPointAdjusted)
        df.loc[model_name, "F1"] = score_acc.f1(ScoreType.RevisedPointAdjusted)
        df.loc[model_name, "Mean Time to Detect"] = score_acc.mean_time_to_detect()
        df.loc[model_name, "PA Precision"] = pa_score_acc.precision(ScoreType.PointAdjusted)
        df.loc[model_name, "PA Recall"] = pa_score_acc.recall(ScoreType.PointAdjusted)
        df.loc[model_name, "PA F1"] = pa_score_acc.f1(ScoreType.PointAdjusted)
        df.loc[model_name, "PW Precision"] = pw_score_acc.precision(ScoreType.Pointwise)
        df.loc[model_name, "PW Recall"] = pw_score_acc.recall(ScoreType.Pointwise)
        df.loc[model_name, "PW F1"] = pw_score_acc.f1(ScoreType.Pointwise)

        df = df.loc[sorted(df.index)]
        df.to_csv(summary, index=True)


if __name__ == "__main__":
    main()
