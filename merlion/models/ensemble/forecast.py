#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""Ensembles of forecasters."""
import logging
import traceback
from typing import List, Optional, Tuple, Union

import pandas as pd

from merlion.evaluate.forecast import ForecastEvaluator, ForecastEvaluatorConfig
from merlion.models.ensemble.base import EnsembleConfig, EnsembleTrainConfig, EnsembleBase
from merlion.models.ensemble.combine import Mean
from merlion.models.forecast.base import ForecasterBase, ForecasterWithExogConfig, ForecasterWithExogBase
from merlion.utils.time_series import TimeSeries

logger = logging.getLogger(__name__)


class ForecasterEnsembleConfig(ForecasterWithExogConfig, EnsembleConfig):
    """
    Config class for an ensemble of forecasters.
    """

    _default_combiner = Mean(abs_score=False)

    def __init__(self, max_forecast_steps=None, target_seq_index=None, verbose=False, **kwargs):
        self.verbose = verbose
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=None, **kwargs)
        # Override the target_seq_index of all individual models after everything has been initialized
        # FIXME: doesn't work if models have heterogeneous transforms which change the dim of the input time series
        self.target_seq_index = target_seq_index
        if self.models is not None:
            assert all(model.target_seq_index == self.target_seq_index for model in self.models)

    @property
    def target_seq_index(self):
        return self._target_seq_index

    @target_seq_index.setter
    def target_seq_index(self, target_seq_index):
        if self.models is not None:
            # Get the target_seq_index from the models if None is given
            if target_seq_index is None:
                non_none_idxs = [m.target_seq_index for m in self.models if m.target_seq_index is not None]
                if len(non_none_idxs) > 0:
                    target_seq_index = non_none_idxs[0]
                assert all(m.target_seq_index in [None, target_seq_index] for m in self.models), (
                    f"Attempted to infer target_seq_index from the individual models in the ensemble, but "
                    f"not all models have the same target_seq_index. Got {[m.target_seq_index for m in self.models]}"
                )
            # Only override the target_seq_index from the models if there is one
            if target_seq_index is not None:
                for model in self.models:
                    model.config.target_seq_index = target_seq_index
        # Save the ensemble-level target_seq_index as a private variable
        self._target_seq_index = target_seq_index


class ForecasterEnsemble(EnsembleBase, ForecasterWithExogBase):
    """
    Class representing an ensemble of multiple forecasting models.
    """

    models: List[ForecasterBase]
    config_class = ForecasterEnsembleConfig

    _default_train_config = EnsembleTrainConfig(valid_frac=0.2)

    @property
    def require_even_sampling(self) -> bool:
        return False

    def __init__(self, config: ForecasterEnsembleConfig = None, models: List[ForecasterBase] = None):
        super().__init__(config=config, models=models)
        for model in self.models:
            assert isinstance(
                model, ForecasterBase
            ), f"Expected all models in {type(self).__name__} to be forecasters, but got a {type(model).__name__}."
            model.config.invert_transform = True

    def train_pre_process(self, train_data: TimeSeries) -> TimeSeries:
        idxs = [model.target_seq_index for model in self.models]
        if any(i is not None for i in idxs):
            self.config.target_seq_index = [i for i in idxs if i is not None][0]
            assert all(i in [None, self.target_seq_index] for i in idxs), (
                f"All individual forecasters must have the same target_seq_index "
                f"to be used in a ForecasterEnsemble, but got the following "
                f"target_seq_idx values: {idxs}"
            )
        return super().train_pre_process(train_data=train_data)

    def resample_time_stamps(self, time_stamps: Union[int, List[int]], time_series_prev: TimeSeries = None):
        return time_stamps

    def train_combiner(self, all_model_outs: List[TimeSeries], target: TimeSeries, **kwargs) -> TimeSeries:
        return super().train_combiner(all_model_outs, target, target_seq_index=self.target_seq_index, **kwargs)

    def _train_with_exog(
        self, train_data: pd.DataFrame, train_config: EnsembleTrainConfig = None, exog_data: TimeSeries = None
    ) -> Tuple[Optional[TimeSeries], None]:
        full_train = TimeSeries.from_pd(train_data)
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
                pred, err = model.train(train, train_config=cfg, exog_data=exog_data)
                preds.append(pred)
                errs.append(err)
            except Exception:
                logger.warning(
                    f"Caught an exception while training model {i+1}/{len(self.models)} ({type(model).__name__}). "
                    f"Model will not be used. {traceback.format_exc()}"
                )
                self.combiner.set_model_used(i, False)
                preds.append(None)
                errs.append(None)

        # Train the combiner on the validation data
        if train is valid:
            combined = self.train_combiner(preds, train)
        else:
            logger.info("Evaluating validation performance...")
            h = self.get_max_common_horizon()
            preds = []
            for i, model in enumerate(self.models):
                pred = None
                try:
                    if self.combiner.get_model_used(i):
                        evaluator = ForecastEvaluator(model=model, config=ForecastEvaluatorConfig(horizon=h))
                        _, pred = evaluator.get_predict(train_vals=train, test_vals=valid, pretrained=True)
                except Exception:
                    logger.warning(
                        f"Caught an exception while evaluating model {i + 1}/{len(self.models)} "
                        f"({type(model).__name__}). Model will not be used. {traceback.format_exc()}"
                    )
                    self.combiner.set_model_used(i, False)
                preds.append(pred)
            combined = self.train_combiner(preds, valid)

        # No need to re-train if we didn't use a validation split
        if train is valid:
            err = None if any(e is None for e in errs) else self.combiner(errs, None)
            return combined, err

        # Re-train on the full data if we used a validation split
        full_preds, full_errs = [], []
        for i, (model, used, cfg) in enumerate(zip(self.models, self.models_used, per_model_train_configs)):
            model.reset()
            if used:
                logger.info(f"Re-training model {i+1}/{len(self.models)} ({type(model).__name__}) on full data...")
                pred, err = model.train(full_train, train_config=cfg, exog_data=exog_data)
            else:
                pred, err = None, None
            full_preds.append(pred)
            full_errs.append(err)
        if any(used and e is None for used, e in zip(self.models_used, full_errs)):
            err = None
        else:
            err = self.combiner(full_errs, None)
        return self.combiner(full_preds, None), err

    def _forecast_with_exog(
        self,
        time_stamps: List[int],
        time_series_prev: pd.DataFrame = None,
        return_prev=False,
        exog_data: pd.DataFrame = None,
        exog_data_prev: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        preds, errs = [], []
        time_series_prev = TimeSeries.from_pd(time_series_prev)
        if exog_data is not None:
            exog_data = pd.concat((exog_data_prev, exog_data)) if exog_data_prev is not None else exog_data
            exog_data = TimeSeries.from_pd(exog_data)
        for model, used in zip(self.models, self.models_used):
            if used:
                pred, err = model.forecast(
                    time_stamps=time_stamps,
                    time_series_prev=time_series_prev,
                    exog_data=exog_data,
                    return_prev=return_prev,
                )
                preds.append(pred)
                errs.append(err)

        pred = self.combiner(preds, None).to_pd()
        err = None if any(e is None for e in errs) else self.combiner(errs, None).to_pd()
        return pred, err
