#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for ensembles of models.
"""
import copy
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from merlion.models.base import ModelBase, Config
from merlion.models.ensemble.combine import CombinerBase, CombinerFactory, Mean
from merlion.models.factory import ModelFactory
from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


class EnsembleConfig(Config):
    """
    An ensemble config contains the each individual model in the ensemble, as well as the Combiner object
    to combine those models' outputs. The rationale behind placing the model objects in the EnsembleConfig
    (rather than in the Ensemble itself) is discussed in more detail in the documentation for `LayeredModel`.
    """

    _default_combiner = Mean(abs_score=False)
    models: List[ModelBase]

    def __init__(self, models: List[Union[ModelBase, Dict]] = None, combiner: CombinerBase = None, **kwargs):
        """
        :param models: A list of models or dicts representing them.
        :param combiner: The `CombinerBase` object to combine the outputs of the models in the ensemble.
        :param kwargs: Any additional kwargs for `Config`
        """
        super().__init__(**kwargs)
        if combiner is None:
            self.combiner = copy.deepcopy(self._default_combiner)
        elif isinstance(combiner, dict):
            self.combiner = CombinerFactory.create(**combiner)
        else:
            self.combiner = combiner

        if models is not None:
            models = [ModelFactory.create(**m) if isinstance(m, dict) else copy.deepcopy(m) for m in models]
        self.models = models

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"models"}))
        if "models" not in _skipped_keys:
            if self.models is None:
                models = None
            else:
                models = [None if m is None else dict(name=type(m).__name__, **m.config.to_dict()) for m in self.models]
            config_dict["models"] = models
        return config_dict

    def __copy__(self):
        config_dict = super().to_dict(_skipped_keys={"models"})
        config_dict["models"] = self.models
        return self.from_dict(config_dict)

    def __deepcopy__(self, memodict={}):
        copied = copy.copy(self)
        copied.models = copy.deepcopy(self.models)
        return copied


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


class EnsembleBase(ModelBase, metaclass=AutodocABCMeta):
    """
    An abstract class representing an ensemble of multiple models.
    """

    config_class = EnsembleConfig
    _default_train_config = EnsembleTrainConfig(valid_frac=0.0)

    def __init__(self, config: EnsembleConfig = None, models: List[ModelBase] = None):
        """
        :param config: The ensemble's config
        :param models: The models in the ensemble. Only provide this argument if you did not specify ``config.models``.
        """
        msg = f"Expected exactly one of `config.models` or `models` when creating a {type(self).__name__}."
        if config is None and models is None:
            raise RuntimeError(f"{msg} Received neither.")
        elif config is not None and models is not None:
            if config.models is None:
                config.models = models
            else:
                raise RuntimeError(f"{msg} Received both.")
        elif config is None:
            config = self.config_class(models=models)
        super().__init__(config=config)

    @property
    def models(self):
        return self.config.models

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
        i = np.argmin([pd.to_datetime(0) + h for h in horizons if h is not None])
        return horizons[i]

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

    def __getstate__(self):
        state = super().__getstate__()
        if self.models is None:
            state["models"] = None
        else:
            state["models"] = [None if model is None else model.__getstate__() for model in self.models]
        return state

    def __setstate__(self, state):
        if "models" in state:
            model_states = state.pop("models")
            if self.models is None and model_states is not None:
                raise ValueError(f"`{type(self).__name__}.models` is None, but received a non-None `models` state.")
            elif self.models is None or model_states is None:
                self.config.models = None
            else:
                for i, (model, model_state) in enumerate(zip(self.models, model_states)):
                    if model is None and model_state is not None:
                        raise ValueError(f"One of the Ensemble models is None, but received a non-None model state.")
                    elif model is None or model_state is None:
                        self.models[i] = None
                    else:
                        model.__setstate__(model_state)
        super().__setstate__(state)

    def save(self, dirname: str, save_only_used_models=False, **save_config):
        """
        Saves the ensemble of models.

        :param dirname: directory to save the ensemble to
        :param save_only_used_models: whether to save only the models that are actually used by the ensemble.
        :param save_config: additional save config arguments
        """
        super().save(dirname=dirname, save_only_used_models=save_only_used_models, **save_config)

    def _save_state(
        self, state_dict: Dict[str, Any], filename: str = None, save_only_used_models=False, **save_config
    ) -> Dict[str, Any]:
        """
        Saves the model's state to the the specified file, or just modifies the state_dict as needed.

        :param state_dict: The state dict to save.
        :param filename: The name of the file to save the model to.
        :param save_only_used_models: whether to save only the models that are actually used by the ensemble.
        :param save_config: additional configurations (if needed)
        :return: The state dict to save.
        """
        state_dict.pop("config", None)  # don't save the model's config in binary
        if self.models is not None:
            model_states = []
            for model, model_state, model_used in zip(self.models, state_dict["models"], self.models_used):
                if save_only_used_models and not model_used:
                    model_states.append(None)
                else:
                    model_states.append(
                        model._save_state(model_state, None, save_only_used_models=save_only_used_models, **save_config)
                    )
            state_dict["models"] = model_states
        return super()._save_state(state_dict, filename, **save_config)

    def to_bytes(self, save_only_used_models=False, **save_config):
        """
        Converts the entire model state and configuration to a single byte object.

        :param save_only_used_models: whether to save only the models that are actually used by the ensemble.
        :param save_config: additional configurations (if needed)
        """
        return super().to_bytes(save_only_used_models=save_only_used_models, **save_config)
