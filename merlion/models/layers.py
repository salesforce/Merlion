#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for _layered_ models. These are models which act as a wrapper around another model, often with additional
functionality. This is the basis for :py:mod:`default models <merlion.models.defaults>` and
:py:mod:`AutoML models <merlion.models.automl>`_.
"""
from copy import deepcopy
from typing import Any, Dict, Union

from merlion.models.base import Config, ModelBase
from merlion.models.factory import ModelFactory
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta


class LayeredModelConfig(Config):
    def __init__(self, model: Union[ModelBase, Dict], model_kwargs=None, **kwargs):
        """
        :param model: The model being wrapped, or a ``dict`` representing it.
        :param model_kwargs: Keyword arguments used specifically to initialize the underlying model. Only used if
            ``model`` is a ``dict``.
        :param kwargs: Any other keyword arguments (e.g. for initializing a base class). If ``model`` is a ``dict``,
            we will also try to pass these arguments when creating the actual underlying model.
        """
        # Model-specific kwargs override kwargs when creating the model.
        model_kwargs = {} if model_kwargs is None else model_kwargs
        if isinstance(model, dict):
            model, kwargs = ModelFactory.create(**{**model, **kwargs, **model_kwargs, "return_unused_kwargs": True})
        self.model = model
        super().__init__(**kwargs)

        # If no model was created, reserve unused kwargs to try initializing the model with
        if model is None:
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in self.to_dict()}
            self.model_kwargs = {**extra_kwargs, **model_kwargs}

    @property
    def model(self):
        """
        The underlying model which this layer wraps.
        """
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.model_kwargs = {}

    @property
    def depth(self):
        """
        The depth of the layered model, i.e. the number of layers until we hit a base model. Base models have depth 0.
        """
        return 0 if self.model is None else self.model.config.depth + 1

    @property
    def base_model(self):
        """
        The base model at the heart of the full layered model.
        """
        model = self.model
        for _ in range(self.depth - 1):
            model = model.model
        return model

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"model"}))
        if not self.model_kwargs and "model_kwargs" in config_dict:
            config_dict["model_kwargs"] = None
        if "model" not in _skipped_keys:
            if self.model is None:
                config_dict["model"] = None
            else:
                config_dict["model"] = dict(name=type(self.model).__name__, **self.model.config.to_dict())
        return config_dict

    def __copy__(self):
        config_dict = super().to_dict(_skipped_keys={"model"})
        return self.__class__(model=deepcopy(self.model), **config_dict)

    def __getattr__(self, item):
        base_model = self.base_model
        is_detector_attr = isinstance(base_model, DetectorBase) and hasattr(DetectorConfig, item)
        is_forecaster_attr = isinstance(base_model, ForecasterBase) and hasattr(ForecasterConfig, item)
        if is_detector_attr or is_forecaster_attr:
            return getattr(base_model.config, item)
        return self.__getattribute__(item)


class LayeredModel(ModelBase, metaclass=AutodocABCMeta):
    """
    Abstract class implementing a model that wraps around another internal model.
    """

    config_class = LayeredModelConfig
    require_even_sampling = False
    require_univariate = False

    def __init__(self, config: LayeredModelConfig = None, model: ModelBase = None, **kwargs):
        msg = f"Expected exactly one of `config.model` or `model` when creating LayeredModel {type(self).__name__}."
        if config is None and model is None:
            raise RuntimeError(f"{msg} Received neither.")
        elif config is not None and model is not None:
            if config.model is None:
                config.model = model
            else:
                raise RuntimeError(f"{msg} Received both.")
        elif config is None:
            config = self.config_class(model=model, **kwargs)
        super().__init__(config=config)

    @property
    def model(self):
        return self.config.model

    @model.setter
    def model(self, model):
        self.config.model = model

    @property
    def base_model(self):
        return self.config.base_model

    def reset(self):
        self.model.reset()
        self.__init__(config=self.config)

    def __getstate__(self):
        state = super().__getstate__()
        state["model"] = self.model.__getstate__()
        return state

    def __setstate__(self, state):
        if "model" in state:
            model_state = state.pop("model")
            self.model.__setstate__(model_state)
        super().__setstate__(state)

    def _save_state(self, state_dict: Dict[str, Any], filename: str = None, **save_config) -> Dict[str, Any]:
        # don't save config for any of the sub-models
        sub_state = state_dict
        for d in range(self.config.depth):
            sub_state.pop("config", None)
            sub_state = sub_state["model"]
        return super()._save_state(state_dict, filename, **save_config)

    def __getattr__(self, item):
        """
        We can get callable attributes from the base model, as well as attributes from `ForecasterBase` or
        `DetectorBase` if the base model is (respectively) a forecaster or anomaly detector.
        """
        base_model = self.base_model
        attr = getattr(base_model, item, None)
        is_detector_attr = isinstance(base_model, DetectorBase) and hasattr(DetectorBase, item)
        is_forecaster_attr = isinstance(base_model, ForecasterBase) and hasattr(ForecasterBase, item)
        if callable(attr) or is_detector_attr or is_forecaster_attr:
            return attr
        return self.__getattribute__(item)

    def train(self, train_data: TimeSeries, **kwargs):
        train_data = self.train_pre_process(
            train_data, require_even_sampling=self.require_even_sampling, require_univariate=self.require_univariate
        )
        return self.model.train(train_data, **kwargs)

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        if not isinstance(self.base_model, DetectorBase):
            raise NotImplementedError(f"Base model is a {type(self.base_model)}, which is not an anomaly detector.")
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
        return self.model.get_anomaly_score(time_series, time_series_prev)

    def forecast(self, time_stamps, time_series_prev: TimeSeries = None, *args, **kwargs):
        if not isinstance(self.base_model, ForecasterBase):
            raise NotImplementedError(f"Base model is a {type(self.base_model)}, which is not a forecaster.")
        if time_series_prev is not None:
            time_series_prev = self.transform(time_series_prev)
        return self.model.forecast(time_stamps, time_series_prev, *args, **kwargs)
