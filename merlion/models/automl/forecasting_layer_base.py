#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import json
import os
from abc import ABC
from copy import deepcopy
from os.path import join
from typing import Tuple, Optional, Union, List

import dill

from merlion.models.automl.layer_mixin import LayerMixIn
from merlion.models.factory import ModelFactory
from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries


class ForecasterAutoMLBase(ForecasterBase, LayerMixIn, ABC):
    """
    Base Implementation of AutoML Layer Logic.

    Custom `train` and `forecast` methods that call rely on implementations of `LayerMixIn` to perform the training and
    forecasting procedures.

    Note: Layer models don't have a config but any calls to their config will bubble down to the underlying model. This
    may be a blessing or a curse.
    """

    def __init__(self, model: ForecasterBase, **kwargs):
        """
        Assume config also inherits ForecastConfig
        """
        if isinstance(model, dict):
            model = ModelFactory.create(**{**model, **kwargs})
        self.model = model

    def reset(self):
        self.model.reset()
        self.__init__(self.model)

    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        original_train_data = train_data
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)

        candidate_thetas = self.generate_theta(train_data)
        # need to call evaluate_theta on original training data since evaluate_theta often trains another model
        # and therefore we might be applying transform twice
        theta, model, train_result = self.evaluate_theta(candidate_thetas, original_train_data, train_config)
        if model:
            self.model = model
            return train_result
        else:
            model = deepcopy(self.model)
            model.reset()
            self.set_theta(model, theta, train_data)
            self.model = model
            return self.model.train(original_train_data, train_config)

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Union[Tuple[TimeSeries, Optional[TimeSeries]], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        return self.model.forecast(time_stamps, time_series_prev, return_iqr, return_prev)

    def save(self, dirname: str, **save_config):
        state_dict = self.__getstate__()
        state_dict.pop("model")
        model_path = os.path.abspath(join(dirname, self.filename))
        config_dict = dict()

        # create the directory if needed
        os.makedirs(dirname, exist_ok=True)

        underlying_model_path = os.path.abspath(os.path.join(dirname, "model"))
        self.model.save(underlying_model_path)
        config_dict["model_name"] = type(self.model).__name__

        with open(os.path.join(dirname, self.config_class.filename), "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        # Save the model state
        self._save_state(state_dict, model_path, **save_config)

    @classmethod
    def load(cls, dirname: str, **kwargs):
        # Read the config dict from json
        config_path = os.path.join(dirname, cls.config_class.filename)
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        model_name = config_dict.pop("model_name")
        model = ModelFactory.load(model_name, os.path.abspath(os.path.join(dirname, "model")))

        # Load the state dict
        with open(os.path.join(dirname, cls.filename), "rb") as f:
            state_dict = dill.load(f)

        return cls._from_config_state_dicts(state_dict, model, **kwargs)

    @classmethod
    def _from_config_state_dicts(cls, state_dict, model, **kwargs):
        model = cls(model)
        model._load_state(state_dict, **kwargs)

        return model

    def __getattr__(self, attr):
        try:
            return getattr(self.model, attr)
        except AttributeError:
            try:
                return getattr(self.model.config, attr)
            except AttributeError:
                raise AttributeError(f"Attribute {attr} not found in underlying class {type(self.model)}")
