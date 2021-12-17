#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""Ensembles of forecasters."""
import copy
import logging
from typing import List, Optional, Tuple, Union

from scipy.stats import norm
from tqdm import tqdm

from merlion.models.ensemble.base import EnsembleConfig, EnsembleTrainConfig, EnsembleBase
from merlion.models.ensemble.combine import Mean
from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.utils.time_series import to_pd_datetime, TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class ForecasterEnsembleConfig(ForecasterConfig, EnsembleConfig):
    """
    Config class for an ensemble of forecasters.
    """

    _default_combiner = Mean(abs_score=False)

    def __init__(self, max_forecast_steps=None, verbose=False, **kwargs):
        self.verbose = verbose
        super().__init__(max_forecast_steps=max_forecast_steps, **kwargs)


class ForecasterEnsemble(EnsembleBase, ForecasterBase):
    """
    Class representing an ensemble of multiple forecasting models.
    """

    models: List[ForecasterBase]
    config_class = ForecasterEnsembleConfig

    _default_train_config = EnsembleTrainConfig(valid_frac=0.2)

    def __init__(self, config: ForecasterEnsembleConfig = None, models: List[ForecasterBase] = None):
        super().__init__(config=config, models=models)
        for model in self.models:
            assert isinstance(model, ForecasterBase), (
                f"Expected all models in {type(self).__name__} to be anomaly "
                f"detectors, but got a {type(model).__name__}."
            )

    def train_pre_process(
        self, train_data: TimeSeries, require_even_sampling: bool, require_univariate: bool
    ) -> TimeSeries:
        idxs = [model.target_seq_index for model in self.models]
        if any(i is not None for i in idxs):
            self.config.target_seq_index = [i for i in idxs if i is not None][0]
            assert all(i in [None, self.target_seq_index] for i in idxs), (
                f"All individual forecasters must have the same target_seq_index "
                f"to be used in a ForecasterEnsemble, but got the following "
                f"target_seq_idx values: {idxs}"
            )
        return super().train_pre_process(
            train_data=train_data, require_even_sampling=require_even_sampling, require_univariate=require_univariate
        )

    def train(
        self, train_data: TimeSeries, train_config: EnsembleTrainConfig = None
    ) -> Tuple[Optional[TimeSeries], None]:
        if train_config is None:
            train_config = copy.deepcopy(self._default_train_config)
        full_train = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)
        train, valid = self.train_valid_split(full_train, train_config)

        per_model_train_configs = train_config.per_model_train_configs
        if per_model_train_configs is None:
            per_model_train_configs = [None] * len(self.models)
        assert len(per_model_train_configs) == len(self.models), (
            f"You must provide the same number of per-model train configs "
            f"as models, but received received {len(per_model_train_configs)} "
            f"train configs for an ensemble with {len(self.models)} models"
        )

        # Train individual models on the training data
        preds, errs = [], []
        for i, (model, cfg) in enumerate(zip(self.models, per_model_train_configs)):
            logger.info(f"Training model {i+1}/{len(self.models)} ({type(model).__name__})...")
            try:
                pred, err = model.train(train, train_config=cfg)
                preds.append(pred)
                errs.append(err)
            except TypeError as e:
                if "'NoneType' object is not subscriptable" in str(e):
                    raise RuntimeError(
                        f"train() method of {type(model).__name__} model "
                        f"does not return its fitted predictions for the "
                        f"training data. Therefore, this model cannot be "
                        f"used in a forecaster ensemble."
                    )
                else:
                    raise e

        # Train the combiner on the validation data
        try:
            if train is valid:
                k = train.names[self.target_seq_index]
                combined = self.train_combiner(preds, train.univariates[k].to_ts())
            else:
                logger.info("Evaluating validation performance...")
                h = self.get_max_common_horizon()
                k = valid.names[self.target_seq_index]
                if h is None:
                    preds = [model.forecast(valid.time_stamps)[0] for model in self.models]
                else:
                    # evaluate using prediction windows of size h
                    prev = train
                    t0, tf = valid.t0, valid.tf
                    valid_windows = []
                    preds = [[] for _ in self.models]
                    pbar = tqdm(total=int(tf - t0), desc="Validation", disable=not self.config.verbose)
                    while t0 < tf:
                        next_tf = to_pd_datetime(prev.tf) + h
                        dt = int((next_tf - to_pd_datetime(prev.tf)).total_seconds())
                        pbar.update(min(dt, int(tf - t0)))
                        window = valid.window(to_pd_datetime(t0), next_tf, include_tf=False)
                        t0 += (next_tf - to_pd_datetime(prev.tf)).total_seconds()
                        if window.is_empty():
                            continue
                        valid_windows.append(window.univariates[k].to_ts())
                        for i, model in enumerate(self.models):
                            preds[i].append(model.forecast(window.time_stamps, prev)[0])
                        prev = prev + window
                    valid = valid_windows
                    pbar.close()

                combined = self.train_combiner(preds, valid)
        except AssertionError as e:
            if "None of `all_model_outs` can be `None`" in str(e):
                nones = [m for m, p in zip(self.models, preds) if p is None]
                raise RuntimeError(
                    f"Training a {type(self.combiner).__name__} combiner "
                    f"fitted model predictions for the training data, but "
                    f"the following models' train() method doesn't return any: "
                    f"{', '.join(type(m).__name__ for m in nones)}"
                )
            else:
                raise e

        # No need to re-train if we didn't use a validation split
        if train is valid:
            err = None if any(e is None for e in errs) else self.combiner(errs, None)
            return combined, err

        # Re-train on the full data if we used a validation split
        full_preds, full_errs = [], []
        for i, (model, cfg) in enumerate(zip(self.models, per_model_train_configs)):
            logger.info(f"Re-training model {i+1}/{len(self.models)} ({type(model).__name__}) on full data...")
            model.reset()
            pred, err = model.train(full_train, train_config=cfg)
            full_preds.append(pred)
            full_errs.append(err)
        err = None if any(e is None for e in full_errs) else self.combiner(full_errs, None)
        if not all(self.models_used):
            used = [f"{i+1} ({type(m).__name__})" for i, (m, u) in enumerate(zip(self.models, self.models_used)) if u]
            logger.info(f"Models used (of {len(self.models)}): {', '.join(used)}")
        return self.combiner(full_preds, None), err

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Union[Tuple[TimeSeries, Union[TimeSeries, None]], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        if time_series_prev is not None:
            time_series_prev = self.transform(time_series_prev)

        preds, errs = [], []
        for model, used in zip(self.models, self.models_used):
            if used:
                pred, err = model.forecast(
                    time_stamps=time_stamps,
                    time_series_prev=time_series_prev,
                    return_iqr=False,
                    return_prev=return_prev,
                )
                preds.append(pred)
                errs.append(err)

        pred = self.combiner(preds, None)
        err = None if any(e is None for e in errs) else self.combiner(errs, None)
        if return_iqr:
            if err is None:
                raise RuntimeError(
                    f"ForecasterEnsemble of {[type(m).__name__ for m in self.models]}"
                    f"does not support uncertainty estimates"
                )
            name = pred.names[0]
            yhat = pred.univariates[name].to_pd()
            err = err.univariates[err.names[0]].to_pd()
            lb = UnivariateTimeSeries(
                name=f"{name}_lower", time_stamps=time_stamps, values=yhat + norm.ppf(0.25) * err
            ).to_ts()
            ub = UnivariateTimeSeries(
                name=f"{name}_upper", time_stamps=time_stamps, values=yhat + norm.ppf(0.75) * err
            ).to_ts()
            return pred, lb, ub

        return pred, err
