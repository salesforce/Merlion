#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Base class for Deep Learning Models
"""
import copy
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from abc import abstractmethod
from enum import Enum
import pdb

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
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        num_epochs: int = 10,
        optimizer: Union[str, Optimizer] = Optimizer.Adam,
        loss_fn: Union[str, LossFunction] = LossFunction.mse,
        clip_gradient: Optional[float] = None,
        use_gpu: bool = False,
        ts_encoding: Union[None, str] = "h",
        validation_rate: float = 0.2,
        early_stop_patience: Union[None, int] = None,
        **kwargs,
    ):
        """
        :param lr: The learning rate for training
        :param batch_size: The batch size for training
        :param num_epochs: The number of traning epochs
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
    Base class for a deep learning model
    """

    config_class = DeepConfig
    deep_model_class = TorchModel

    def __init__(self, config: DeepConfig):
        super().__init__(config)
        self.deep_model = None

    @abstractmethod
    def _create_model(self):
        """
        Create and initialize deep models
        """
        raise NotImplementedError

    @abstractmethod
    def _init_optimizer(self):
        """
        Initialize the optimizer for training
        """
        raise NotImplementedError

    @abstractmethod
    def _init_loss_fn(self):
        """
        Initialize loss functions for training
        """
        raise NotImplementedError

    @abstractmethod
    def _get_batch_model_loss_and_outputs(self, batch):
        """
        Calculate loss and get prediction given a batch
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
            deep_model = deep_model.to(torch.device("cpu"))
            state["deep_model"] = copy.deepcopy(deep_model.state_dict())

        return state

    def __setstate__(self, state):
        deep_model_state_dict = state.pop("forest", None)
        super().__setstate__(state)

        pdb.set_trace()
        if deep_model_state_dict:
            self._create_model()
            self.to_cpu()
            self.deep_model.load_state_dict(deep_model_state_dict)
