#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from abc import ABC
from copy import deepcopy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from merlion.models.automl.layer_mixin import LayerMixIn
from merlion.models.factory import ModelFactory
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.utils import TimeSeries


class ForecasterAutoMLConfig(ForecasterConfig):
    def __init__(self, model, **kwargs):
        if isinstance(model, dict):
            model = ModelFactory.create(**{**model, **kwargs})
        self.model = model
        kwargs["max_forecast_steps"] = getattr(model, "max_forecast_steps", None)
        kwargs["target_seq_index"] = getattr(model, "target_seq_index", None)
        super().__init__(**kwargs)

    @property
    def max_forecast_steps(self):
        return self.model.max_forecast_steps

    @max_forecast_steps.setter
    def max_forecast_steps(self, m):
        self.model.config.max_forecast_steps = m

    @property
    def target_seq_index(self):
        return self.model.target_seq_index

    @target_seq_index.setter
    def target_seq_index(self, i):
        self.model.config.target_seq_index = i

    def __getattr__(self, attr):
        try:
            return getattr(self.model, attr)
        except AttributeError:
            try:
                return getattr(self.model.config, attr)
            except AttributeError:
                raise AttributeError(f"Attribute {attr} not found in underlying class {type(self.model)}")

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"model"}))
        config_dict["model"] = {"name": type(self.model).__name__, **self.model.config.to_dict()}
        return config_dict

    def __copy__(self):
        config_dict = super().to_dict(_skipped_keys={"model"})
        model = deepcopy(self.model)
        return self.__class__(model=model, **config_dict)


class ForecasterAutoMLBase(ForecasterBase, LayerMixIn, ABC):
    """
    Base Implementation of AutoML Layer Logic.

    Custom `train` and `forecast` methods that call rely on implementations of `LayerMixIn` to perform the training and
    forecasting procedures.

    Note: Layer models don't have a config but any calls to their config will bubble down to the underlying model. This
    may be a blessing or a curse.
    """

    config_class = ForecasterAutoMLConfig

    def __init__(self, *, model: ForecasterBase = None, config: ForecasterAutoMLConfig = None, **kwargs):
        """
        Assume config also inherits ForecastConfig
        """
        msg = f"Expected at exactly one of config or model when initializing Layer model {type(self).__name__}."
        if config is None and model is None:
            raise RuntimeError(f"{msg} Received neither.")
        elif config is not None and model is not None:
            raise RuntimeError(f"{msg} Received both.")
        elif config is None:
            config = self.config_class(model=model, **kwargs)
        super().__init__(config=config)

    def reset(self):
        self.model.reset()
        self.__init__(config=self.config)

    @property
    def model(self):
        return self.config.model

    @model.setter
    def model(self, model):
        self.config.model = model

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

    def __getstate__(self):
        state = super().__getstate__()
        state["model"] = (type(self.model).__name__, self.model.to_bytes())
        return state

    def __setstate__(self, state):
        if "model" in state:
            model_name, model_bytes = state.pop("model")
            self.model = ModelFactory.get_model_class(model_name).from_bytes(model_bytes)
        super().__setstate__(state)

    def __getattr__(self, attr):
        try:
            return getattr(self.model, attr)
        except AttributeError:
            try:
                return getattr(self.model.config, attr)
            except AttributeError:
                raise AttributeError(f"Attribute {attr} not found in underlying class {type(self.model)}")
