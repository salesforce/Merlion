#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Implementation of Deep AR: 
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting: https://arxiv.org/abs/2012.07436
    Code adapted from https://github.com/thuml/Autoformer. 
"""
import pdb
import copy
import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from typing import List, Optional, Tuple, Union
from abc import abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

from merlion.utils.misc import initializer

from merlion.models.base import NormalizingConfig
from merlion.models.deep_base import TorchModel, LossFunction
from merlion.models.forecast.deep_base import DeepForecasterConfig, DeepForecaster


logger = logging.getLogger(__name__)


class DeepARConfig(DeepForecasterConfig, NormalizingConfig):
    """
    Config object for informer forecaster
    """

    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        hidden_size: Union[int, None] = 32,
        num_hidden_layers: int = 2,
        loss_fn: Union[str, LossFunction] = LossFunction.guassian_nll,
        **kwargs
    ):
        """
        :param n_past: # of past steps used for forecasting future.
        :param max_forecast_steps:  Max # of steps we would like to forecast for.
        :param hidden_size: hidden_size of the LSTM layers
        :param num_hidden_layers: # of hidden layers in LSTM
        """

        super().__init__(n_past=n_past, max_forecast_steps=max_forecast_steps, loss_fn="guassian_nll", **kwargs)


class DeepARModel(TorchModel):
    """
    Implementaion of Deep AR model
    """

    def __init__(self, config: DeepARConfig):
        super().__init__(config)
        output_size = input_size = config.dim

        if config.hidden_size is None:
            hidden_size = int(4 * (1 + math.pow(math.log(config.dim), 4)))
        else:
            hidden_size = config.hidden_size

        self.rnn = nn.LSTM(
            input_size,
            hidden_size=hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size * 2),
        )

    def forward(self, past, past_timestamp, future_timestamp, sample_only=True, **kwargs):
        x, _ = self.rnn(past)
        decoder_out = self.decoder(x)

        mu, log_sigma = torch.split(decoder_out, self.config.dim, dim=-1)
        sigma = torch.log(1 + torch.exp(log_sigma)) + 1e-06

        output = mu + torch.randn_like(mu, dtype=torch.float, device=self.device) * sigma

        return output if sample_only else output, mu, sigma


class DeepARForecaster(DeepForecaster):
    """
    Implementaion of Deep AR model forecaster
    """

    config_class = DeepARConfig
    deep_model_class = DeepARModel

    def __init__(self, config: DeepARConfig):
        super().__init__(config)

    def _get_batch_model_loss_and_outputs(self, batch):
        config = self.config
        device = self.deep_model.device
        past, past_timestamp, future, future_timestamp = batch

        model_output, mu, sigma = self.deep_model(past, past_timestamp, future_timestamp, sample_only=False)

        if future is None:
            return None, model_output, None

        if config.target_seq_index is None and self.support_multivariate_output:
            target_future = future
        else:
            target_idx = config.target_seq_index
            target_future = future[:, :, target_idx : target_idx + 1]

        loss = self.loss_fn(mu, target_future, torch.square(sigma))
        return loss, model_output, target_future
