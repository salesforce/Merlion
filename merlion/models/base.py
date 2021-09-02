#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains the base classes for all models.
"""
from abc import abstractmethod
from copy import deepcopy
import json
import logging
import os
from os.path import abspath, join
from typing import Any, Dict, Optional, Tuple

import dill

from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.transform.normalize import Rescale, MeanVarNormalize
from merlion.transform.sequence import TransformSequence
from merlion.utils.time_series import assert_equal_timedeltas, TimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


def override_config(config, config_dict, return_unused_kwargs=False, **kwargs):
    """
    :meta private:
    """
    to_remove = []
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            to_remove.append(key)

    for key in to_remove:
        kwargs.pop(key)

    for key, value in config_dict.items():
        if key not in kwargs and not hasattr(config, key):
            kwargs[key] = value

    if len(kwargs) > 0 and not return_unused_kwargs:
        logger.warning(f"Unused kwargs: {kwargs}", stack_info=True)
    elif return_unused_kwargs:
        return config, kwargs
    return config


class Config(object):
    """
    Abstract class which defines a model config.
    """

    filename = "config.json"
    _default_transform = Identity()

    def __init__(self, transform: TransformBase = None, **kwargs):
        """
        :param transform: Transformation to pre-process input time series.
        """
        super().__init__()
        if transform is None:
            self.transform = deepcopy(self._default_transform)
        elif isinstance(transform, dict):
            self.transform = TransformFactory.create(**transform)
        else:
            self.transform = transform

    def to_dict(self, _skipped_keys=None):
        """
        :return: dict with keyword arguments used to initialize the config class.
        """
        config_dict = {}
        skipped_keys = set() if _skipped_keys is None else _skipped_keys
        for key, value in self.__dict__.items():
            k_strip = key.lstrip("_")
            key = k_strip if hasattr(self, k_strip) else key
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            if key not in skipped_keys:
                config_dict[key] = deepcopy(value)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], return_unused_kwargs=False, **kwargs):
        """
        Constructs a `Config` from a Python dictionary of parameters.

        :param config_dict: dict that will be used to instantiate this object.
        :param return_unused_kwargs: whether to return any unused keyword args.
        :param kwargs: any additional parameters to set (overriding config_dict).

        :return: `Config` object initialized from the dict.
        """
        config = cls(**config_dict)
        return override_config(
            config=config, config_dict=config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs
        )

    def __copy__(self):
        return self.from_dict(self.to_dict())

    def __deepcopy__(self, memodict={}):
        return self.__copy__()


class NormalizingConfig(Config):
    """
    Model config where the transform must return normalized values. Applies
    additional normalization after the initial data pre-processing transform.
    """

    def __init__(self, normalize: Rescale = None, **kwargs):
        """
        :param normalize: Pre-trained normalization transformation (optional).
        """
        super().__init__(**kwargs)
        if normalize is None:
            self.normalize = MeanVarNormalize()
        elif isinstance(normalize, dict):
            self.normalize = TransformFactory.create(**normalize)
        else:
            self.normalize = normalize

    @property
    def full_transform(self):
        """
        Returns the full transform, including the pre-processing step, lags, and
        final mean/variance normalization.
        """
        return TransformSequence([self.transform, self.normalize])

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        """
        Set the pre-processing transform. Also resets the mean/variance
        normalization, since the new transform could change these, and the
        new mean/variance may need to be re-learned.
        """
        self._transform = transform
        self.normalize = MeanVarNormalize()


class ModelBase(metaclass=AutodocABCMeta):
    """
    Abstract base class for models.
    """

    filename = "model.pkl"
    config_class = Config
    _default_train_config = None

    def __init__(self, config: Config):
        assert isinstance(config, self.config_class)
        self.config = deepcopy(config)
        self.last_train_time = None
        self.timedelta = None

    def reset(self):
        """
        Resets the model's internal state.
        """
        self.__init__(self.config)

    def __getstate__(self):
        return {k: deepcopy(v) for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        for name, value in state.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'. "
                    f"'{name}' is an invalid kwarg for the load() method."
                )

    @property
    def transform(self):
        """
        :return: The data pre-processing transform to apply on any time series,
            before giving it to the model.
        """
        return getattr(self.config, "full_transform", self.config.transform)

    @transform.setter
    def transform(self, transform):
        self.config.transform = transform

    def train_pre_process(
        self, train_data: TimeSeries, require_even_sampling: bool, require_univariate: bool
    ) -> TimeSeries:
        """
        Applies pre-processing steps common for training most models.

        :param train_data: the original time series of training data
        :param require_even_sampling: whether the model assumes that training
            data is sampled at a fixed frequency
        :param require_univariate: whether the model only works with univariate
            time series

        :return: the training data, after any necessary pre-processing has been applied
        """
        self.transform.train(train_data)
        train_data = self.transform(train_data)

        # Make sure the training data is univariate & all timestamps are equally
        # spaced (this is a key assumption for ARIMA)
        if require_univariate and train_data.dim != 1:
            raise RuntimeError(
                f"Transform {self.transform} transforms data into a {train_data.dim}-"
                f"variate time series, but model {type(self).__name__} can "
                f"only handle uni-variate time series. Change the transform."
            )

        if require_even_sampling:
            assert_equal_timedeltas(train_data.univariates[train_data.names[0]])
            assert train_data.is_aligned

        t = train_data.time_stamps
        self.timedelta = t[1] - t[0]
        self.last_train_time = t[-1]
        return train_data

    def transform_time_series(
        self, time_series: TimeSeries, time_series_prev: TimeSeries = None
    ) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        """
        Applies the model's pre-processing transform to ``time_series`` and
        ``time_series_prev``.

        :param time_series: The time series
        :param time_series_prev: A time series of context, immediately preceding
            ``time_series``. Optional.

        :return: The transformed ``time_series``.
        """
        t0 = time_series.t0
        if time_series_prev is not None:
            time_series = time_series_prev + time_series
            time_series_prev, time_series = self.transform(time_series).bisect(t0, t_in_left=False)
        else:
            time_series = self.transform(time_series)
        return time_series, time_series_prev

    @abstractmethod
    def train(self, train_data: TimeSeries, train_config=None):
        """
        Trains the model on the specified time series, optionally with some
        additional implementation-specific config options ``train_config``.

        :param train_data: a `TimeSeries` to use as a training set
        :param train_config: additional configurations (if needed)
        """
        raise NotImplementedError

    def _save_state(self, state_dict: Dict[str, Any], filename: str = None, **save_config) -> Dict[str, Any]:
        """
        Saves the model's state to the the specified file. If you override this
        method, please also override _load_state(). By default, the model's state
        dict is just serialized using dill.

        :param state_dict: The state dict to save.
        :param filename: The name of the file to save the model to.
        :param save_config: additional configurations (if needed)
        :return: The state dict to save.
        """
        if "config" in state_dict:  # don't save the config
            state_dict.pop("config")
        if filename is not None:
            with open(filename, "wb") as f:
                dill.dump(state_dict, f)

        return state_dict

    def save(self, dirname: str, **save_config):
        """
        :param dirname: directory to save the model & its config
        :param save_config: additional configurations (if needed)
        """
        state_dict = self.__getstate__()
        config_dict = self.config.to_dict()
        model_path = abspath(join(dirname, self.filename))
        config_dict["model_path"] = model_path

        # create the directory if needed
        os.makedirs(dirname, exist_ok=True)

        # Save the config dict
        with open(join(dirname, self.config_class.filename), "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        # Save the model state
        self._save_state(state_dict, model_path, **save_config)

    def _load_state(self, state_dict: Dict[str, Any], **kwargs):
        """
        Loads the model's state from the specified file. Override this method if
        you have overridden _save_state(). By default, the model's state dict is
        loaded from a file (serialized by dill), and the state is set.

        :param filename: serialized file containing the model's state.
        :param kwargs: any additional keyword arguments to set manually in the
            state dict (after loading it).
        """
        if "config" in state_dict:  # don't re-set the config
            state_dict.pop("config")
        self.__setstate__(state_dict)

    @classmethod
    def _load_state_dict(cls, model_path: str):
        with open(model_path, "rb") as f:
            state_dict = dill.load(f)
        return state_dict

    @classmethod
    def load(cls, dirname: str, **kwargs):
        """
        :param dirname: directory to load model (and config) from
        :param kwargs: config params to override manually
        :return: `ModelBase` object loaded from file
        """
        # Load the config
        config_path = join(dirname, cls.config_class.filename)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        model_path = config_dict.pop("model_path")
        # Load the state
        state_dict = cls._load_state_dict(model_path)

        return cls._from_config_state_dicts(config_dict, state_dict, **kwargs)

    @classmethod
    def _from_config_state_dicts(cls, config_dict, state_dict, **kwargs):
        """
        Initializes a model from the config and state dictionaries used to
        save it.

        :param config_dict: Dictionary used to initialize the config.
        :param state_dict: Dictionary used to load the model state.
        :param kwargs: config params to override manually
        :return: `ModelBase` object loaded from file
        """
        config, model_kwargs = cls.config_class.from_dict(config_dict, return_unused_kwargs=True, **kwargs)
        model = cls(config)
        model._load_state(state_dict, **model_kwargs)

        return model

    def to_bytes(self, **save_config):
        """
        Converts the entire model state and configuration to a single byte object.

        :return: bytes object representing the model.
        """
        return dill.dumps(self._to_serializable_comps(**save_config))

    def _to_serializable_comps(self, **save_config):
        state_dict = self.__getstate__()
        config_dict = self.config.to_dict()
        state_dict = self._save_state(state_dict, **save_config)
        class_name = self.__class__.__name__
        return class_name, config_dict, state_dict

    @classmethod
    def from_bytes(cls, obj, **kwargs):
        """
        Creates a fully specified model from a byte object

        :param obj: byte object to convert into a model
        :return: ModelBase object loaded from ``obj``
        """
        name, config_dict, state_dict = dill.loads(obj)
        return cls._from_config_state_dicts(config_dict, state_dict, **kwargs)

    def __copy__(self):
        new_model = self.__class__(deepcopy(self.config))
        state_dict = self.__getstate__()
        state_dict.pop("config", None)
        new_model.__setstate__(state_dict)
        return new_model

    def __deepcopy__(self, memodict={}):
        return self.__copy__()


class ModelWrapper(ModelBase, metaclass=AutodocABCMeta):
    """
    Abstract class implementing a model that wraps around another internal model.
    """

    filename = "model"

    def __init__(self, config: Config, model: ModelBase = None):
        super().__init__(config)
        self.model = model

    def save(self, dirname: str, **save_config):
        config_dict = self.config.to_dict()
        config_dict["model_type"] = type(self.model).__name__
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, self.config_class.filename), "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        self.model.save(os.path.join(dirname, self.filename), **save_config)

    @classmethod
    def load(cls, dirname: str, **kwargs):
        from merlion.models.factory import ModelFactory

        config_path = os.path.join(dirname, cls.config_class.filename)
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        model_type = config_dict.pop("model_type")
        model = ModelFactory.load(model_type, os.path.join(dirname, cls.filename))
        return cls._from_config_state_dicts(config_dict, model, **kwargs)

    @classmethod
    def _from_config_state_dicts(cls, config_dict, model, **kwargs):
        config = cls.config_class.from_dict(config_dict)
        ret = cls(config=config)
        ret.model = model
        return ret

    def to_bytes(self, **save_config):
        config_dict = self.config.to_dict()
        model_tuple = self.model._to_serializable_comps(**save_config)
        class_name = type(self).__name__
        return dill.dumps((class_name, config_dict, model_tuple))

    @classmethod
    def from_bytes(cls, obj, **kwargs):
        from merlion.models.factory import ModelFactory

        class_name, config_dict, model_tuple = dill.loads(obj)
        model = [ModelFactory.get_model_class(model_tuple[0])._from_config_state_dicts(*model_tuple[1:])]
        return cls._from_config_state_dicts(config_dict, model, **kwargs)
