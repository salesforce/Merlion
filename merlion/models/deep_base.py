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


class DeepConfig(Config):
    """
    Config Object used to define a deep learning (pytorch) model
    """

    @initializer
    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 256,
        num_epochs: int = 10,
        optim_name: str = "adam",
        criterion: str = "mse",
        **kwargs
    ):
        """
        :param lr: The learning rate for training
        :param batch_size: The batch size for training
        :param num_epochs: The number of traning epochs
        """
        super().__init__(**kwargs)
        self.device = None


class TorchModel(nn.Module):
    def __init__(self, config: DeepConfig):
        super(TorchModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, past, *args, **kwargs):
        raise NotImplementedError


class DeepModelBase(ModelBase):
    """
    Base class for a deep learning model
    """

    config_class = DeepConfig
    deep_model_class = TorchModel
    deep_model = None

    def __init__(self, config: DeepConfig):
        super().__init__(config)

    @abstractmethod
    def _create_model(self):
        raise NotImplementedError

    @abstractmethod
    def _deep_batch_iter(self, batch, config: DeepConfig):
        """
        Calculate loss and get prediction given a batch
        """
        raise NotImplementedError

    def to_gpu(self):
        """
        Set device to GPU, and move deep model to GPU
        Currently we only support single GPU training
        """
        if torch.cuda.is_available():
            self.config.device = torch.device("cuda")
            self.deep_model = self.deep_model.to(self.config.device)
        else:
            logger.warning("GPU not available, using CPU instead...")
            self.to_cpu()

    def to_cpu(self):
        """
        Set device to CPU and move deep model to CPU
        """
        if self.config.device is None:
            self.config.device = torch.device("cpu")

        if self.deep_model is not None:
            self.deep_model = self.deep_model.to(self.config.device)

    def save_model(self, model_path: str = None):
        raise NotImplementedError

    def load_model(self, model_path: str = None):
        raise NotImplementedError
