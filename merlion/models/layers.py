#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for layered models. These are models which act as a wrapper around another model, often with additional
functionality. This is the basis for `default models <merlion.models.defaults>`_ and
`AutoML models <merlion.models.automl>`_.
"""
import copy
import inspect
import logging
from typing import Any, Dict, List, Union

import pandas as pd

from merlion.models.base import Config, ModelBase
from merlion.models.factory import ModelFactory
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig, ForecasterExogBase, ForecasterExogConfig
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta, call_with_accepted_kwargs

logger = logging.getLogger(__name__)

_DETECTOR_MEMBERS = dict(inspect.getmembers(DetectorConfig)).keys()
_FORECASTER_MEMBERS = dict(inspect.getmembers(ForecasterConfig)).keys()
_FORECASTER_EXOG_MEMBERS = dict(inspect.getmembers(ForecasterExogConfig)).keys()


def _is_detector_attr(base_model, attr):
    return isinstance(base_model, DetectorBase) and attr in _DETECTOR_MEMBERS


def _is_forecaster_attr(base_model, attr):
    is_member = isinstance(base_model, ForecasterBase) and attr in _FORECASTER_MEMBERS
    return is_member or (isinstance(base_model, ForecasterExogBase) and attr in _FORECASTER_EXOG_MEMBERS)


class LayeredModelConfig(Config):
    """
    Config object for a `LayeredModel`. See `LayeredModel` documentation for more details.
    """

    def __init__(self, model: Union[ModelBase, Dict], model_kwargs=None, **kwargs):
        """
        :param model: The model being wrapped, or a dict representing it.
        :param model_kwargs: Keyword arguments used specifically to initialize the underlying model. Only used if
            ``model`` is a dict. Will override keys in the ``model`` dict if specified.
        :param kwargs: Any other keyword arguments (e.g. for initializing a base class). If ``model`` is a dict,
            we will also try to pass these arguments when creating the actual underlying model. However, they will
            not override arguments in either the ``model`` dict or ``model_kwargs`` dict.
        """
        # Model-specific kwargs override kwargs when creating the model.
        model_kwargs = {} if model_kwargs is None else model_kwargs
        if isinstance(model, dict):
            model.update(
                {
                    k: v
                    for k, v in kwargs.items()
                    if k not in model and k not in model_kwargs and k not in _LAYERED_MEMBERS
                }
            )
            model, extra_kwargs = ModelFactory.create(**{**model, **model_kwargs, "return_unused_kwargs": True})
            kwargs.update(extra_kwargs)
        self.model = model
        self.model_kwargs = {}
        super().__init__(**kwargs)

        # Reserve unused kwargs to initialize the model with (useful if model is None, and can be helpful for reset())
        model_kwargs = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in {**kwargs, **model_kwargs}.items()}
        self.model_kwargs = self._remove_used_kwargs(self.to_dict(), model_kwargs)

    @property
    def base_model(self):
        """
        The base model at the heart of the full layered model.
        """
        model = self.model
        while isinstance(model, LayeredModel):
            model = model.model
        return model

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"model"}))
        # Serialize only the model's config (the model itself is serialized separately)
        if "model" not in _skipped_keys:
            if self.model is None:
                config_dict["model"] = None
            else:
                config_dict["model"] = dict(name=type(self.model).__name__, **self.model.config.to_dict(_skipped_keys))
        # Don't serialize any of the used keys from model_kwargs
        if "model_kwargs" in config_dict:
            config_dict["model_kwargs"] = self._remove_used_kwargs(config_dict, config_dict["model_kwargs"])
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], return_unused_kwargs=False, dim=None, **kwargs):
        config, kwargs = super().from_dict(config_dict=config_dict, return_unused_kwargs=True, dim=dim, **kwargs)
        if config.model is None:
            base_class_members = set(_DETECTOR_MEMBERS).union(_FORECASTER_MEMBERS).union(_FORECASTER_EXOG_MEMBERS)
            used = {k: v for k, v in kwargs.items() if k in base_class_members}
            config.model_kwargs.update(used)
            kwargs = {k: v for k, v in kwargs.items() if k not in used}

        if len(kwargs) > 0 and not return_unused_kwargs:
            logger.warning(f"Unused kwargs: {kwargs}", stack_info=True)
        elif return_unused_kwargs:
            return config, kwargs
        return config

    @staticmethod
    def _remove_used_kwargs(config_dict, kwargs):
        used_keys = set()  # Removes kwargs which have already been used by given config dict
        while isinstance(config_dict, dict):
            used_keys = used_keys.union(config_dict.keys())
            config_dict = config_dict.get("model", None)
        return {k: v for k, v in kwargs.items() if k not in used_keys}

    def __copy__(self):
        config_dict = super().to_dict(_skipped_keys={"model"})
        config_dict["model"] = self.model
        return self.from_dict(config_dict)

    def __deepcopy__(self, memodict={}):
        config_dict = super().to_dict(_skipped_keys={"model"})
        config_dict["model"] = copy.deepcopy(self.model)
        return self.from_dict(config_dict)

    def __getattr__(self, item):
        if item in ["model", "base_model"]:
            return super().__getattribute__(item)
        base_model = self.base_model
        if _is_detector_attr(base_model, item) or _is_forecaster_attr(base_model, item):
            return getattr(base_model.config, item)
        elif base_model is None and item in self.model_kwargs:
            return self.model_kwargs.get(item)
        return self.__getattribute__(item)

    def __setattr__(self, key, value):
        if hasattr(self, "model") and hasattr(self.model, "config"):
            base = self.base_model
            if key not in _LAYERED_MEMBERS and (_is_detector_attr(base, key) or _is_forecaster_attr(base, key)):
                return setattr(base.config, key, value)
        return super().__setattr__(key, value)

    def get_unused_kwargs(self, **kwargs):
        config = self
        valid_keys = {"model"}.union(config.to_dict(_skipped_keys={"model"}).keys())
        while isinstance(config, LayeredModelConfig) and config.model is not None:
            config = config.model.config
            valid_keys = valid_keys.union(config.to_dict(_skipped_keys={"model"}).keys())
        return {k: v for k, v in kwargs.items() if k not in valid_keys}


_LAYERED_MEMBERS = dict(inspect.getmembers(LayeredModelConfig)).keys()


class LayeredModel(ModelBase, metaclass=AutodocABCMeta):
    """
    Abstract class implementing a model which wraps around another internal model.

    The actual underlying model is stored in ``model.config.model``, and ``model.model`` is a property which references
    this. This is to allow the model to retain the initializer ``LayeredModel(config)``, and to ensure that various
    attributes do not become de-synchronized (e.g. if we were to store ``config.model_config`` and ``model.model``
    separately).

    We define the *base model* as the non-layered model at the base of the overall model hierarchy.

    The layered model is allowed to access any callable attribute of the base model,
    e.g. ``model.set_seasonality(...)`` resolves to``model.base_model.set_seasonality(...)`` for a `SeasonalityModel`.
    If the base model is a forecaster, the layered model will automatically inherit from `ForecasterBase`; similarly
    for `DetectorBase` or `ForecastingDetectorBase`. The abstract methods (``forecast`` and ``get_anomaly_score``)
    are overridden to call the underlying model.

    If the base model is a forecaster, the top-level config ``model.config`` does not duplicate attributes of the
    underlying forecaster config (e.g. ``max_forecast_steps`` or ``target_seq_index``). Instead,
    ``model.config.max_forecast_steps`` will resolve to ``model.config.base_model.max_forecast_steps``.
    As a result, you will only need to specify this parameter once. The same holds true for `DetectorConfig` attributes
    (e.g. ``threshold`` or ``calibrator``) when the base model is an anomaly detector.

    .. note::

        For the time being, every layer of the model is allowed to have its own ``transform``. However, after the
        model is trained, the entire transform will be composed as a single `TransformSequence` and will be owned by
        the base model.
    """

    config_class = LayeredModelConfig

    def __new__(cls, config: LayeredModelConfig = None, model: ModelBase = None, **kwargs):
        # Dynamically inherit from the appropriate kind of base model.
        # However, this creates a new class that isn't registered anywhere with pickle/dill. This causes
        # serialization problems, especially when using models with multiprocessing. So we maintain this
        # class (cls) as a class attribute _original_cls of the new, dynamically created class. This is
        # used by the __reduce__ method when pickling a LayeredModel.
        original_cls = cls
        config = cls._resolve_args(config=config, model=model, **kwargs)
        if isinstance(config.model, ForecastingDetectorBase):
            cls = cls.__class__(cls.__name__, (cls, LayeredForecastingDetector), {})
            setattr(cls, "_original_cls", original_cls)
        elif isinstance(config.model, ForecasterBase):
            cls = cls.__class__(cls.__name__, (cls, LayeredForecaster), {})
            setattr(cls, "_original_cls", original_cls)
        elif isinstance(config.model, DetectorBase):
            cls = cls.__class__(cls.__name__, (cls, LayeredDetector), {})
            setattr(cls, "_original_cls", original_cls)
        return super().__new__(cls)

    def __init__(self, config: LayeredModelConfig = None, model: ModelBase = None, **kwargs):
        super().__init__(config=self._resolve_args(config=config, model=model, **kwargs))

    @classmethod
    def _resolve_args(cls, config: LayeredModelConfig, model: ModelBase, **kwargs):
        if config is None and model is None:
            raise RuntimeError(
                f"Expected at least one of `config` or `model` when creating {cls.__name__}. Received neither."
            )
        elif config is not None and model is not None:
            if config.model is None:
                if isinstance(model, dict):
                    model = ModelFactory.create(**model)
                config = copy.copy(config)
                config.model = model
            else:
                raise RuntimeError(
                    f"Expected at most one of `config.model` or `model` when creating {cls.__name__}. Received both."
                )
        elif config is None:
            config = cls.config_class(model=model, **kwargs)
        return config

    @property
    def _pandas_train(self):
        return self.model._pandas_train

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    @property
    def model(self):
        return self.config.model

    @model.setter
    def model(self, model):
        self.config.model = model

    @property
    def base_model(self):
        return self.config.base_model

    @property
    def train_data(self):
        return None if self.model is None else self.model.train_data

    @train_data.setter
    def train_data(self, train_data):
        if self.model is not None:
            self.model.train_data = train_data

    @property
    def _default_train_config(self):
        return self.model._default_train_config

    def reset(self):
        self.model.reset()
        self.__init__(config=self.config)

    def __getstate__(self):
        state = super().__getstate__()
        state["model"] = None if self.model is None else self.model.__getstate__()
        return state

    def __setstate__(self, state):
        if "model" in state:
            model_state = state.pop("model")
            if self.model is None and model_state is not None:
                raise ValueError(f"{type(self).__name__}.model is None, but received a non-None model state.")
            elif self.model is None or model_state is None:
                self.model = None
            else:
                self.model.__setstate__(model_state)
        super().__setstate__(state)

    def __reduce__(self):
        state_dict = self.__getstate__()
        config = state_dict.pop("config")
        return getattr(self.__class__, "_original_cls", self.__class__), (config,), state_dict

    def _save_state(self, state_dict: Dict[str, Any], filename: str = None, **save_config) -> Dict[str, Any]:
        state_dict.pop("config", None)  # don't save the model's config in binary
        if self.model is not None:
            state_dict["model"] = self.model._save_state(state_dict["model"], filename=None, **save_config)
        return super()._save_state(state_dict, filename, **save_config)

    def __getattr__(self, item):
        """
        We can get callable attributes from the base model.
        """
        base_model = self.base_model
        attr = getattr(base_model, item, None)
        if callable(attr):
            return attr
        return self.__getattribute__(item)

    def _train(self, train_data: pd.DataFrame, train_config=None, **kwargs):
        """
        Trains the underlying model. May be overridden, e.g. for AutoML.

        :param train_data: the data to train on.
        :param train_config: the train config of the underlying model (optional).
        """
        return call_with_accepted_kwargs(self.model._train, train_data=train_data, train_config=train_config, **kwargs)

    def train_pre_process(self, train_data: TimeSeries, **kwargs) -> TimeSeries:
        # Push the layered model transform to the owned model, but make sure we only resample once.
        has_resample = False
        transforms = []
        for t in TransformSequence([self.transform, self.model.transform]).transforms:
            if isinstance(t, TemporalResample):
                if not has_resample:
                    transforms.append(t)
                has_resample = True
            else:
                transforms.append(t)
        self.transform = Identity()
        self.model.transform = TransformSequence(transforms)

        # Return the result of calling the underlying model's train_pre_process()
        train_data = super().train_pre_process(train_data)
        return call_with_accepted_kwargs(self.model.train_pre_process, train_data=train_data, **kwargs)

    def train_post_process(self, train_result, **kwargs):
        # All post_processing is handled by the underlying model
        return call_with_accepted_kwargs(self.model.train_post_process, train_result=train_result, **kwargs)


class LayeredDetector(LayeredModel, DetectorBase):
    """
    Base class for a layered anomaly detector. Only to be used as a subclass.
    """

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        raise NotImplementedError("Layered model _get_anomaly_score() should not be called.")

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None, **kwargs) -> TimeSeries:
        kwargs.update(time_series=time_series, time_series_prev=time_series_prev)
        return call_with_accepted_kwargs(self.model.get_anomaly_score, **kwargs)


class LayeredForecaster(LayeredModel, ForecasterBase):
    """
    Base class for a layered forecaster. Only to be used as a subclass.
    """

    def _train_with_exog(self, train_data: pd.DataFrame, train_config=None, exog_data: pd.DataFrame = None, **kwargs):
        kwargs.update(train_data=train_data, train_config=train_config, exog_data=exog_data)
        return call_with_accepted_kwargs(self.model._train_with_exog, **kwargs)

    def _forecast(self, time_stamps: List[int], time_series_prev: TimeSeries = None, return_prev=False):
        raise NotImplementedError("Layered model _forecast() should not be called.")

    def forecast(self, time_stamps, time_series_prev: TimeSeries = None, **kwargs):
        kwargs.update(time_stamps=time_stamps, time_series_prev=time_series_prev)
        return call_with_accepted_kwargs(self.model.forecast, **kwargs)


class LayeredForecastingDetector(LayeredForecaster, LayeredDetector, ForecastingDetectorBase):
    """
    Base class for a layered forecasting detector. Only to be used as a subclass.
    """
