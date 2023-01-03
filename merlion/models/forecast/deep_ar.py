#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Implementation of Deep AR
"""
import copy
import logging
import math

import numpy as np
import pandas as pd

from typing import List, Optional, Tuple, Union

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
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks: https://arxiv.org/abs/1704.04110
    """

    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        hidden_size: Union[int, None] = 32,
        num_hidden_layers: int = 2,
        lags_seq: List[int] = [1],
        num_prediction_samples: int = 10,
        loss_fn: Union[str, LossFunction] = LossFunction.guassian_nll,
        **kwargs,
    ):
        """
        :param n_past: # of past steps used for forecasting future.
        :param max_forecast_steps:  Max # of steps we would like to forecast for.
        :param hidden_size: hidden_size of the LSTM layers
        :param num_hidden_layers: # of hidden layers in LSTM
        :param lags_seq: Indices of the lagged observations that the RNN takes as input. For example,
            ``[1]`` indicates that the RNN only takes the observation at time ``t-1`` to produce the
            output for time ``t``.
        :param num_prediction_samples: # of samples to produce the forecasting
        """

        super().__init__(n_past=n_past, max_forecast_steps=max_forecast_steps, loss_fn=loss_fn, **kwargs)


class DeepARModel(TorchModel):
    """
    Implementaion of Deep AR model
    """

    def __init__(self, config: DeepARConfig):
        super().__init__(config)

        assert len(config.lags_seq) > 0, "lags_seq must not be empty!"
        self.lags_seq = config.lags_seq
        self.n_past = config.n_past
        self.n_context = config.n_past - max(self.lags_seq)
        self.max_forecast_steps = config.max_forecast_steps
        self.output_size = config.dim

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}

        input_size = len(self.lags_seq) * self.output_size + freq_map[config.ts_encoding]

        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]
        self.num_prediction_samples = config.num_prediction_samples

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

        self.distr_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size * 2),
        )

        self.loss_fn = self.config.loss_fn.value()

    @staticmethod
    def get_lagged_subsequences(
        sequence,
        sequence_length,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than n_past, found lag {max(indices)} " f"while n_past is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def unroll_encoder(self, past, past_timestamp, future_timestamp, future=None):
        if future_timestamp is None or future is None:
            time_features = past_timestamp[:, (self.n_past - self.n_context) :, :]
            sequence = past
            sequence_length = self.n_past
            subsequences_length = self.n_context
        else:
            time_features = torch.cat((past_timestamp[:, (self.n_past - self.n_context) :, :], future_timestamp), dim=1)
            sequence = torch.cat((past, future), dim=1)
            sequence_length = self.n_past + self.max_forecast_steps
            subsequences_length = self.n_context + self.max_forecast_steps

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        input_lags = lags.reshape((-1, subsequences_length, len(self.lags_seq) * self.output_size))
        rnn_inputs = torch.cat((input_lags, time_features), dim=-1)
        outputs, states = self.rnn(rnn_inputs)

        return outputs, states

    def calculate_loss(self, past, past_timestamp, future, future_timestamp):
        rnn_outputs, _ = self.unroll_encoder(past, past_timestamp, future_timestamp, future)
        distr_proj_out = self.distr_proj(rnn_outputs)

        mu, log_sigma = torch.split(distr_proj_out, self.config.dim, dim=-1)
        sigma = torch.log(1 + torch.exp(log_sigma)) + 1e-07

        target_future = torch.cat((past[:, (self.n_past - self.n_context) :, :], future), dim=1)

        loss = self.loss_fn(mu, target_future, torch.square(sigma))

        return loss

    def sampling_decoder(self, past, time_features, begin_states):
        repeated_past = past.repeat_interleave(
            repeats=self.num_prediction_samples,
            dim=0,
        )

        repeated_time_features = time_features.repeat_interleave(repeats=self.num_prediction_samples, dim=0)

        repeated_states = [s.repeat_interleave(repeats=self.num_prediction_samples, dim=1) for s in begin_states]

        future_samples = []

        for k in range(self.max_forecast_steps):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past,
                sequence_length=self.n_past + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )
            input_lags = lags.reshape(-1, 1, len(self.lags_seq) * self.output_size)

            decoder_input = torch.cat((input_lags, repeated_time_features[:, k : k + 1, :]), dim=-1)
            rnn_outputs, repeated_states = self.rnn(decoder_input, repeated_states)

            distr_proj_out = self.distr_proj(rnn_outputs)

            mu, log_sigma = torch.split(distr_proj_out, self.config.dim, dim=-1)
            sigma = torch.log(1 + torch.exp(log_sigma)) + 1e-07

            new_samples = mu + torch.randn_like(mu, dtype=torch.float, device=self.device) * sigma

            repeated_past = torch.cat((repeated_past, new_samples), dim=1)
            future_samples.append(new_samples)

        samples = torch.cat(future_samples, dim=1)

        return samples.reshape((-1, self.num_prediction_samples, self.max_forecast_steps, self.output_size))

    def forward(self, past, past_timestamp, future_timestamp, mean_samples=True):
        _, states = self.unroll_encoder(past, past_timestamp, future_timestamp)

        forecast_samples = self.sampling_decoder(
            past=past,
            time_features=future_timestamp,
            begin_states=states,
        )

        target_idx = self.config.target_seq_index
        if target_idx is not None:
            forecast_samples = forecast_samples[:, :, :, target_idx : target_idx + 1]

        return forecast_samples.mean(dim=1) if mean_samples else forecast_samples


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
        past, past_timestamp, future, future_timestamp = batch

        model_output = self.deep_model(past, past_timestamp, future_timestamp)

        if future is None:
            return None, model_output, None

        # Calcuating the loss with maximum likelihood,
        # which is seperate from the sampling procedure of deep AR models
        loss = self.deep_model.calculate_loss(past, past_timestamp, future, future_timestamp)

        if self.target_seq_index is not None:
            future = future[:, :, self.target_seq_index : self.target_seq_index + 1]

        return loss, model_output, future
