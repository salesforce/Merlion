#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Ensembles of anomaly detectors.
"""
import logging
import traceback
from typing import List

import pandas as pd

from merlion.evaluate.anomaly import TSADMetric, TSADEvaluator, TSADEvaluatorConfig
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.models.ensemble.base import EnsembleConfig, EnsembleTrainConfig, EnsembleBase
from merlion.models.ensemble.combine import Mean
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class DetectorEnsembleConfig(DetectorConfig, EnsembleConfig):
    """
    Config class for an ensemble of anomaly detectors.
    """

    _default_combiner = Mean(abs_score=True)

    @property
    def _default_threshold(self):
        if self.per_model_threshold:
            return None
        return AggregateAlarms(alm_threshold=3.0, abs_score=True)

    @property
    def per_model_threshold(self):
        """
        :return: whether to apply the thresholding rules of each individual
            model, before combining their outputs. Only done if doing model
            selection.
        """
        from merlion.models.ensemble.combine import ModelSelector

        return isinstance(self.combiner, ModelSelector) and not self.enable_threshold

    def __init__(self, enable_calibrator=False, **kwargs):
        """
        :param enable_calibrator: Whether to enable calibration of the ensemble
            anomaly score. ``False`` by default.
        :param kwargs: Any additional kwargs for `EnsembleConfig` or `DetectorConfig`
        """
        super().__init__(enable_calibrator=enable_calibrator, **kwargs)


class DetectorEnsembleTrainConfig(EnsembleTrainConfig):
    """
    Config object describing how to train an ensemble of anomaly detectors.
    """

    def __init__(self, valid_frac=0.0, per_model_train_configs=None, per_model_post_rule_train_configs=None):
        """
        :param valid_frac: fraction of training data to use for validation.
        :param per_model_train_configs: list of train configs to use for individual models, one per model.
            ``None`` means that you use the default for all models. Specifying ``None`` for an individual
            model means that you use the default for that model.
        :param per_model_post_rule_train_configs: list of post-rule train configs to use for individual models, one per
            model. ``None`` means that you use the default for all models. Specifying ``None`` for an individual
            model means that you use the default for that model.
        """
        super().__init__(valid_frac=valid_frac, per_model_train_configs=per_model_train_configs)
        self.per_model_post_rule_train_configs = per_model_post_rule_train_configs


class DetectorEnsemble(EnsembleBase, DetectorBase):
    """
    Class representing an ensemble of multiple anomaly detection models.
    """

    models: List[DetectorBase]
    config_class = DetectorEnsembleConfig

    def __init__(self, config: DetectorEnsembleConfig = None, models: List[DetectorBase] = None):
        super().__init__(config=config, models=models)
        for model in self.models:
            assert isinstance(model, DetectorBase), (
                f"Expected all models in {type(self).__name__} to be anomaly "
                f"detectors, but got a {type(model).__name__}."
            )
            model.config.enable_threshold = self.per_model_threshold

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    @property
    def _default_post_rule_train_config(self):
        return dict(metric=TSADMetric.F1, unsup_quantile=None)

    @property
    def _default_train_config(self):
        return DetectorEnsembleTrainConfig()

    @property
    def per_model_threshold(self):
        """
        :return: whether to apply the threshold rule of each individual model
            before aggregating their anomaly scores.
        """
        return self.config.per_model_threshold

    def _train(
        self,
        train_data: TimeSeries,
        train_config: DetectorEnsembleTrainConfig = None,
        anomaly_labels: TimeSeries = None,
    ) -> TimeSeries:
        """
        Trains each anomaly detector in the ensemble unsupervised, and each of
        their post-rules supervised (if labels are given).

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param train_config: `DetectorEnsembleTrainConfig` for ensemble training.
        :param anomaly_labels: a `TimeSeries` indicating which timestamps are anomalous. Optional.

        :return: A `TimeSeries` of the ensemble's anomaly scores on the training data.
        """
        train, valid = self.train_valid_split(train_data, train_config)
        if valid is not None:
            logger.warning("Using a train/validation split to train a DetectorEnsemble is not recommended!")

        train_cfgs = train_config.per_model_train_configs
        if train_cfgs is None:
            train_cfgs = [None] * len(self.models)
        assert len(train_cfgs) == len(self.models), (
            f"You must provide the same number of per-model train configs as models, but received received"
            f"{len(train_cfgs)} train configs for an ensemble with {len(self.models)} models."
        )

        pr_cfgs = train_config.per_model_post_rule_train_configs
        if pr_cfgs is None:
            pr_cfgs = [None] * len(self.models)
        assert len(pr_cfgs) == len(self.models), (
            f"You must provide the same number of per-model post-rule train configs as models, but received "
            f"{len(pr_cfgs)} post-rule train configs for an ensemble with {len(self.models)} models."
        )

        # Train each model individually, with its own train config & post-rule train config
        all_scores = []
        eval_cfg = TSADEvaluatorConfig(retrain_freq=None, cadence=self.get_max_common_horizon(train))
        # TODO: parallelize me
        for i, (model, cfg, pr_cfg) in enumerate(zip(self.models, train_cfgs, pr_cfgs)):
            try:
                train_kwargs = dict(train_config=cfg, anomaly_labels=anomaly_labels, post_rule_train_config=pr_cfg)
                train_scores, valid_scores = TSADEvaluator(model=model, config=eval_cfg).get_predict(
                    train_vals=train, test_vals=valid, train_kwargs=train_kwargs, post_process=True
                )
                scores = train_scores if valid is None else valid_scores
            except Exception:
                logger.warning(
                    f"Caught an exception while training model {i + 1}/{len(self.models)} ({type(model).__name__}). "
                    f"Model will not be used. {traceback.format_exc()}"
                )
                self.combiner.set_model_used(i, False)
                scores = None
            all_scores.append(scores)

        # Train combiner on train data if there is no validation data
        if valid is None:
            return self.train_combiner(all_scores, anomaly_labels)

        # Otherwise, train the combiner on the validation data, and re-train the models on the full data
        self.train_combiner(all_scores, anomaly_labels.bisect(t=valid.time_stamps[0], t_in_left=False)[1])
        all_scores = []
        # TODO: parallelize me
        for i, (model, cfg, pr_cfg, used) in enumerate(zip(self.models, train_cfgs, pr_cfgs, self.models_used)):
            model.reset()
            if used:
                logger.info(f"Re-training model {i+1}/{len(self.models)} ({type(model).__name__}) on full data...")
                train_kwargs = dict(train_config=cfg, anomaly_labels=anomaly_labels, post_rule_train_config=pr_cfg)
                train_scores = model.train(train_data, **train_kwargs)
                train_scores = model.post_rule(train_scores)
            else:
                train_scores = None
            all_scores.append(train_scores)
        return self.combiner(all_scores, anomaly_labels)

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        time_series, time_series_prev = TimeSeries.from_pd(time_series), TimeSeries.from_pd(time_series_prev)
        y = [
            model.get_anomaly_label(time_series, time_series_prev)
            for model, used in zip(self.models, self.models_used)
            if used
        ]
        return self.combiner(y, time_series).to_pd()
