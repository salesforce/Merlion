#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for ensembles of models.
"""
from abc import ABC
import copy
import json
import logging
import dill
import os
from typing import Dict, List, Tuple, Union

import pandas as pd
from pandas.tseries.frequencies import to_offset

from merlion.models.base import ModelBase, Config
from merlion.models.ensemble.combine import CombinerBase, CombinerFactory, Mean
from merlion.models.factory import ModelFactory
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class EnsembleConfig(Config):
    """
    An ensemble config contains the configs of each individual model in the ensemble,
    as well as the combiner object to combine those models' outputs.
    """

    _default_combiner = Mean(abs_score=False)

    def __init__(
        self, model_configs: List[Tuple[str, Union[Config, Dict]]] = None, combiner: CombinerBase = None, **kwargs
    ):
        """
        :param model_configs: A list of ``(class_name, config)`` tuples, where
            ``class_name`` is the name of the model's class (as you would
            provide to the `ModelFactory`), and ``config`` is its config or a
            dict. Note that ``model_configs`` is not serialized by
            `EnsembleConfig.to_dict`! The individual models are handled by
            `EnsembleBase.save`. If ``model_configs`` is not provided, you are
            expected to provide the ``models`` directly when initializing the
            `EnsembleBase`.
        :param combiner: The combiner object to combine the outputs of the
            models in the ensemble.
        :param kwargs: Any additional kwargs for `Config`
        """
        super().__init__(**kwargs)
        if combiner is None:
            self.combiner = copy.deepcopy(self._default_combiner)
        elif isinstance(combiner, dict):
            self.combiner = CombinerFactory.create(**combiner)
        else:
            self.combiner = combiner

        if model_configs is not None:
            model_configs = [
                (name, copy.deepcopy(config))
                if isinstance(config, Config)
                else (name, ModelFactory.get_model_class(name).config_class.from_dict(config))
                for name, config in model_configs
            ]
        self.model_configs = model_configs

    def to_dict(self, _skipped_keys=None):
        config_dict = super().to_dict(_skipped_keys)
        model_configs = config_dict["model_configs"]
        if model_configs is not None:
            config_dict["model_configs"] = [(name, config.to_dict()) for name, config in model_configs]
        return config_dict


class EnsembleTrainConfig:
    """
    Config object describing how to train an ensemble.
    """

    def __init__(self, valid_frac, per_model_train_configs=None):
        """
        :param valid_frac: fraction of training data to use for validation.
        :param per_model_train_configs: list of train configs to use for
            individual models, one per model. ``None`` means that you use
            the default for all models. Specifying ``None`` for an individual
            model means that you use the default for that model.
        """
        assert 0 <= valid_frac < 1
        self.valid_frac = valid_frac
        self.per_model_train_configs = per_model_train_configs


class EnsembleBase(ModelBase, ABC):
    """
    An abstract class representing an ensemble of multiple models.
    """

    models: List[ModelBase]
    config_class = EnsembleConfig

    _default_train_config = EnsembleTrainConfig(valid_frac=0.0)

    def __init__(self, config: EnsembleConfig = None, models: List[ModelBase] = None):
        """
        Initializes the ensemble according to the specified config.

        :param config: The ensemble's config
        :param models: The models in the ensemble. Only provide this argument if
            you did not specify ``config.model_configs``, and you want to
            initialize an ensemble from models that have already been
            constructed.
        """
        msg = (
            "When initializing an ensemble, you must either provide the dict "
            "`model_configs` (mapping each model's name to its config) when "
            "creating the `DetectorEnsembleConfig`, or provide a list of "
            "`models` to the constructor of `EnsembleBase`."
        )
        config = self.config_class() if config is None else config
        if config.model_configs is None and models is None:
            raise RuntimeError(f"{msg} Received neither.")
        elif config.model_configs is not None and models is not None:
            logger.warning(f"{msg} Received both. Overriding `model_configs` with the configs belonging to `models`.")

        if models is not None:
            models = [copy.deepcopy(model) for model in models]
            config.model_configs = [(type(model).__name__, model.config) for model in models]
        else:
            models = [ModelFactory.create(name, **config.to_dict()) for name, config in config.model_configs]

        super().__init__(config)
        self.models = models

    def train_valid_split(
        self, transformed_train_data: TimeSeries, train_config: EnsembleTrainConfig
    ) -> Tuple[TimeSeries, TimeSeries]:
        valid_frac = train_config.valid_frac
        if valid_frac == 0 or not self.combiner.requires_training:
            return transformed_train_data, transformed_train_data

        t0 = transformed_train_data.t0
        tf = transformed_train_data.tf

        return transformed_train_data.bisect(t0 + (tf - t0) * (1 - valid_frac))

    def get_max_common_horizon(self):
        horizons = []
        for model in self.models:
            dt = getattr(model, "timedelta", None)
            n = getattr(model, "max_forecast_steps", None)
            if dt is not None and n is not None:
                try:
                    h = pd.to_timedelta(dt * n, unit="s")
                except:
                    h = to_offset(dt * n)
                horizons.append(h)
        if all(h is None for h in horizons):
            return None
        return min([h for h in horizons if h is not None])

    def truncate_valid_data(self, transformed_valid_data: TimeSeries):
        tf = transformed_valid_data.tf
        max_model_tfs = [tf]
        for model in self.models:
            t0 = getattr(model, "last_train_time", None)
            dt = getattr(model, "timedelta", None)
            n = getattr(model, "max_forecast_steps", None)
            if all(x is not None for x in [t0, dt, n]):
                max_model_tfs.append(t0 + dt * n)

        tf = min(max_model_tfs)
        return transformed_valid_data.bisect(tf, t_in_left=True)[0]

    def train_combiner(self, all_model_outs: List[TimeSeries], target: TimeSeries) -> TimeSeries:
        return self.combiner.train(all_model_outs, target)

    @property
    def combiner(self) -> CombinerBase:
        """
        :return: the object used to combine model outputs.
        """
        return self.config.combiner

    def reset(self):
        for model in self.models:
            model.reset()

    @property
    def models_used(self):
        if self.combiner.n_models is not None:
            return self.combiner.models_used
        else:
            return [True] * len(self.models)

    def save(self, dirname: str, save_only_used_models=False, **save_config):
        """
        Saves the ensemble of models.

        :param dirname: directory to save the ensemble to
        :param save_only_used_models: whether to save only the models that are
            actually used by the ensemble.
        :param save_config: additional save config arguments
        """
        state_dict = self.__getstate__()
        state_dict.pop("models")  # to remove from the state dict
        config_dict = self.config.to_dict()
        config_dict.pop("model_configs", None)  # should save/load models directly

        # create the directory if needed & save each individual model
        os.makedirs(dirname, exist_ok=True)
        paths = []
        for i, (model, used) in enumerate(zip(self.models, self.models_used)):
            if used or not save_only_used_models:
                path = os.path.abspath(os.path.join(dirname, str(i)))
                paths.append(path)
                model.save(path)
            else:
                paths.append(None)

        # Add model paths to the config dict, and save it
        config_dict["model_paths"] = [(type(m).__name__, p) for m, p in zip(self.models, paths)]
        with open(os.path.join(dirname, self.config_class.filename), "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        # Save the remaining ensemble state
        filename = os.path.join(dirname, self.filename)
        self._save_state(
            state_dict=state_dict, filename=filename, save_only_used_models=save_only_used_models, **save_config
        )

    @classmethod
    def load(cls, dirname: str, **kwargs):
        # Read the config dict from json
        config_path = os.path.join(dirname, cls.config_class.filename)
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Load all the models from the config dict
        model_paths = config_dict.pop("model_paths")
        models = [ModelFactory.load(name=name, model_path=path) for name, path in model_paths]

        # Load the state dict
        with open(os.path.join(dirname, cls.filename), "rb") as f:
            state_dict = dill.load(f)

        return cls._from_config_state_dicts(config_dict, state_dict, models, **kwargs)

    @classmethod
    def _from_config_state_dicts(cls, config_dict, state_dict, models, **kwargs):
        # Use the config to initialize the model & then load it
        config, model_kwargs = cls.config_class.from_dict(config_dict, return_unused_kwargs=True, **kwargs)
        ensemble = cls(config=config, models=models)
        ensemble._load_state(state_dict, **model_kwargs)

        return ensemble

    def to_bytes(self, save_only_used_models=False, **save_config):
        """
        Converts the entire ensemble to a single byte object.

        :param save_only_used_models: whether to save only the models that are
            actually used by the ensemble.
        :param save_config: additional save config arguments
        :return: bytes object representing the model.
        """
        state_dict = self.__getstate__()
        state_dict.pop("models")
        config_dict = self.config.to_dict()
        config_dict.pop("model_configs")
        state_dict = self._save_state(state_dict, **save_config)
        class_name = self.__class__.__name__

        model_tuples = [
            model._to_serializable_comps()
            for model, used in zip(self.models, self.models_used)
            if used or not save_only_used_models
        ]

        return dill.dumps((class_name, config_dict, state_dict, model_tuples))

    @classmethod
    def from_bytes(cls, obj, **kwargs):
        """
        Creates a fully specified model from a byte object

        :param obj: byte object to convert into a model
        :return: `EnsembleBase` object loaded from ``obj``
        """
        name, config_dict, state_dict, model_tuples = dill.loads(obj)
        models = [
            ModelFactory.get_model_class(model_tuple[0])._from_config_state_dicts(*model_tuple[1:])
            for model_tuple in model_tuples
        ]
        return cls._from_config_state_dicts(config_dict, state_dict, models, **kwargs)
