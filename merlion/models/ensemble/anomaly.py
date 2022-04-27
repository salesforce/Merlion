#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Ensembles of anomaly detectors.
"""
import copy
import logging
from typing import List

import pandas as pd

from merlion.evaluate.anomaly import TSADMetric
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
    _default_train_config = DetectorEnsembleTrainConfig()

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
    def per_model_threshold(self):
        """
        :return: whether to apply the threshold rule of each individual model
            before aggregating their anomaly scores.
        """
        return self.config.per_model_threshold

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        raise NotImplementedError("_train() is not meant to be called for DetectorEnsemble")

    def train(
        self,
        train_data: TimeSeries,
        anomaly_labels: TimeSeries = None,
        train_config: DetectorEnsembleTrainConfig = None,
        post_rule_train_config=None,
    ) -> TimeSeries:
        """
        Trains each anomaly detector in the ensemble unsupervised, and each of
        their post-rules supervised (if labels are given).

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param anomaly_labels: a `TimeSeries` indicating which timestamps are anomalous. Optional.
        :param train_config: `DetectorEnsembleTrainConfig` for ensemble training.
        :param post_rule_train_config: the post-rule train config to use for the ensemble-level post-rule.

        :return: A `TimeSeries` of the ensemble's anomaly scores on the training data.
        """
        if train_config is None:
            train_config = copy.deepcopy(self._default_train_config)
        full_train = self.train_pre_process(train_data)
        train, valid = self.train_valid_split(full_train, train_config)
        if train is not valid:
            logger.warning("Using a train/validation split to train a DetectorEnsemble is not recommended!")

        per_model_train_configs = train_config.per_model_train_configs
        if per_model_train_configs is None:
            per_model_train_configs = [None] * len(self.models)
        assert len(per_model_train_configs) == len(self.models), (
            f"You must provide the same number of per-model train configs "
            f"as models, but received received {len(per_model_train_configs)} "
            f"train configs for an ensemble with {len(self.models)} models"
        )

        # Train each model individually, with its own post-rule train config
        per_model_post_rule_train_configs = train_config.per_model_post_rule_train_configs
        if per_model_post_rule_train_configs is None:
            per_model_post_rule_train_configs = [None] * len(self.models)
        assert len(per_model_post_rule_train_configs) == len(self.models), (
            f"You must provide the same number of per-model post-rule train "
            f"configs as models, but received {len(per_model_post_rule_train_configs)} "
            f"post-rule train configs for an ensemble with {len(self.models)} models."
        )
        all_train_scores = []
        for model, cfg, pr_cfg in zip(self.models, per_model_train_configs, per_model_post_rule_train_configs):
            train_scores = model.train(
                train_data=train, anomaly_labels=anomaly_labels, train_config=cfg, post_rule_train_config=pr_cfg
            )
            train_scores = model.post_rule(train_scores)
            all_train_scores.append(train_scores)

        # Train combiner on validation data if there is any, otherwise use train data
        if train is valid:
            combined = self.train_combiner(all_train_scores, anomaly_labels)
        else:
            # Train combiner on the validation data
            valid = self.truncate_valid_data(valid)
            all_valid_scores = [m.get_anomaly_label(valid) for m in self.models]
            self.train_combiner(all_valid_scores, anomaly_labels)

            # Re-train models on the full data
            all_train_scores = []
            for model, cfg in zip(self.models, per_model_post_rule_train_configs):
                model.reset()
                train_scores = model.train(
                    train_data=full_train, anomaly_labels=anomaly_labels, post_rule_train_config=cfg
                )
                train_scores = model.post_rule(train_scores)
                all_train_scores.append(train_scores)
            combined = self.combiner(all_train_scores, anomaly_labels)

        # Train the model-level post-rule
        self.train_post_rule(combined, anomaly_labels, post_rule_train_config)
        return combined

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        time_series, time_series_prev = TimeSeries.from_pd(time_series), TimeSeries.from_pd(time_series_prev)
        y = [
            model.get_anomaly_label(time_series, time_series_prev)
            for model, used in zip(self.models, self.models_used)
            if used
        ]
        return self.combiner(y, time_series).to_pd()
