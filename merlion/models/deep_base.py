#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains the base classes for all deep learning models.
"""
import io
import json
import copy
import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from abc import abstractmethod
from enum import Enum

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

from merlion.models.base import Config, ModelBase
from merlion.plot import Figure
from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.utils.misc import initializer
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy

logger = logging.getLogger(__name__)


class Optimizer(Enum):
    Adam = torch.optim.Adam
    AdamW = torch.optim.AdamW
    SGD = torch.optim.SGD
    Adagrad = torch.optim.Adagrad
    RMSprop = torch.optim.RMSprop


class LossFunction(Enum):
    mse = nn.MSELoss
    l1 = nn.L1Loss
    huber = nn.HuberLoss


class DeepConfig(Config):
    """
    Config object used to define a deep learning (pytorch) model
    """

    @initializer
    def __init__(
        self,
        batch_size: int = 32,
        num_epochs: int = 10,
        optimizer: Union[str, Optimizer] = Optimizer.Adam,
        loss_fn: Union[str, LossFunction] = LossFunction.mse,
        clip_gradient: Optional[float] = None,
        use_gpu: bool = False,
        ts_encoding: Union[None, str] = "h",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        validation_rate: float = 0.2,
        early_stop_patience: Union[None, int] = None,
        **kwargs,
    ):
        """
        :param batch_size: Batch size of a batch for stochastic training of deep models
        :param num_epochs: Total number of epochs for training.
        :param optimizer: The optimizer for learning the parameters of the deep learning models. The value of optimizer
            can be ``Adam``, ``AdamW``, ``SGD``, ``Adagrad``, ``RMSprop``.
        :param loss_fn: Loss function for optimizing deep learning models. The value of loss_fn can be `mse` for l2 loss,
            `l1` for l1 loss, `huber` for huber loss.
        :param clip_gradient: Clipping gradient norm of model parameters before updating. If `clip_gradient is None`,
            then the gradient will not be clipped.
        :param use_gpu: Whether to use gpu for training deep models. If ``use_gpu = True`` while thre is no GPU device,
            the model will use CPU for training instead.
        :param ts_encoding: whether the timestamp should be encoded to a float vector, which can be used
            for training deep learning based time series models; if ``None``, the timestamp is not encoded.
            If not ``None``, it represents the frequency for time features encoding options:[s:secondly, t:minutely, h:hourly,
            d:daily, b:business days, w:weekly, m:monthly]
        :param lr: Learning rate for optimizing deep learning models.
        :param weight_decay: Weight decay (L2 penalty) (default: 0)
        :param validation_rate: Data percent of validation set splitted from training data
        :param early_stop_patience: Number of epochs with no improvement after which training will be stopped for
            early stopping function. If `early_stop_patience = None`, the training process will not stop early.
        """
        super().__init__(**kwargs)

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[str, Optimizer]):
        if isinstance(optimizer, str):
            valid = set(Optimizer.__members__.keys())
            if optimizer not in valid:
                raise KeyError(f"{optimizer} is not a valid optimizer that supported. Valid optimizers are: {valid}")
            optimizer = Optimizer[optimizer]
        self._optimizer = optimizer

    @property
    def loss_fn(self) -> LossFunction:
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn: Union[str, LossFunction]):
        if isinstance(loss_fn, str):
            valid = set(LossFunction.__members__.keys())
            if loss_fn not in valid:
                raise KeyError(f"{loss_fn} is not a valid loss that supported. Valid optimizers are: {valid}")
            loss_fn = LossFunction[loss_fn]
        self._loss_fn = loss_fn


class TorchModel(nn.Module):
    """
    Abstract base class for Pytorch deep learning models
    """

    def __init__(self, config: DeepConfig):
        super(TorchModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, past, *args, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device


class DeepModelBase(ModelBase):
    """
    Abstract base class for all deep learning models
    """

    config_class = DeepConfig
    deep_model_class = TorchModel

    def __init__(self, config: DeepConfig):
        super().__init__(config)
        self.deep_model = None

    @abstractmethod
    def _create_model(self):
        """
        Create and initialize deep models and neccessary components for training
        """

        self.deep_model = self.deep_model_class(self.config)

        self.optimizer = self.config.optimizer.value(
            self.deep_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.loss_fn = self.config.loss_fn.value()

    @abstractmethod
    def _get_batch_model_loss_and_outputs(self, batch):
        """
        Calculate optimizing loss and get the output of the deep_model,  given a batch of data
        """
        raise NotImplementedError

    def to_gpu(self):
        """
        Move deep model to GPU
        """
        if torch.cuda.is_available():
            if self.deep_model is not None:
                device = torch.device("cuda")
                self.deep_model = self.deep_model.to(device)
        else:
            logger.warning("GPU not available, using CPU instead")
            self.to_cpu()

    def to_cpu(self):
        """
        Move deep model to CPU
        """
        if self.deep_model is not None:
            device = torch.device("cpu")
            self.deep_model = self.deep_model.to(device)

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        deep_model = state.pop("deep_model", None)
        optimizer = state.pop("optimizer", None)
        loss_fn = state.pop("loss_fn", None)

        state = copy.deepcopy(state)

        if deep_model is not None:
            state["deep_model_state_dict"] = deep_model.state_dict()

        return state

    def __setstate__(self, state):
        deep_model_state_dict = state.pop("deep_model_state_dict", None)
        super().__setstate__(state)

        if deep_model_state_dict:
            if self.deep_model is None:
                self._create_model()

            device = self.deep_model.device

            buffer = io.BytesIO()
            torch.save(deep_model_state_dict, buffer)
            buffer.seek(0)

            self.deep_model.load_state_dict(torch.load(buffer, map_location=device))
