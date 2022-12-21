#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Implementation of Autoformer: 
    Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting: https://arxiv.org/abs/2106.13008 
    Code adapted from https://github.com/thuml/Autoformer. 
"""
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


from merlion.models.base import NormalizingConfig
from merlion.models.deep_base import TorchModel
from merlion.models.forecast.deep_base import DeepForecasterConfig, DeepForecaster

from merlion.models.utils.nn_modules import (
    AutoCorrelation,
    AutoCorrelationLayer,
    SeriesDecomposeBlock,
    SeasonalLayernorm,
    DataEmbeddingWoPos,
)

from merlion.models.utils.nn_modules.enc_dec_autoformer import Encoder, Decoder, EncoderLayer, DecoderLayer


from merlion.utils.misc import initializer

logger = logging.getLogger(__name__)


class AutoformerConfig(DeepForecasterConfig, NormalizingConfig):
    """
    Config object for autoformer forecaster
    """

    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        moving_avg: int = 25,
        encoder_input_size: int = None,
        decoder_input_size: int = None,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        factor: int = 3,
        model_dim: int = 512,
        embed: str = "timeF",
        dropout: float = 0.05,
        activation: str = "gelu",
        n_heads: int = 8,
        fcn_dim: int = 2048,
        **kwargs
    ):
        """
        :param n_past: # of past steps used for forecasting future.
        :param max_forecast_steps:  Max # of steps we would like to forecast for.
        :param moving_avg: Window size of moving average for Autoformer.
        :param encoder_input_size: Input size of encoder. If `encoder_input_size = None`, then the model will automatically use `config.dim`,
            which is the dimension of the input data.
        :param decoder_input_size: Input size of decoder. If `decoder_input_size = None`, then the model will automatically use `config.dim`,
            which is the dimension of the input data.
        :param num_encoder_layers: Number of encoder layers.
        :param num_decoder_layers: Number of decoder layers.
        :param factor: Attention factor.
        :param model_dim: Dimension of the model.
        :param embed: Time feature encoding type, options include `timeF`, `fixed` and `learned`.
        :param dropout: dropout rate.
        :param activation: Activation function, can be `gelu`, `relu`, `sigmoid`, etc.
        :param n_heads: Number of heads of the model.
        :param fcn_dim: Hidden dimension of the MLP layer in the model.
        """

        super().__init__(n_past=n_past, max_forecast_steps=max_forecast_steps, **kwargs)


class AutoformerModel(TorchModel):
    """
    Implementaion of Autoformer Deep Torch Model
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__(config)

        if config.dim is not None:
            config.encoder_input_size = config.dim if config.encoder_input_size is None else config.encoder_input_size
            config.decoder_input_size = (
                config.encoder_input_size if config.decoder_input_size is None else config.decoder_input_size
            )

        if config.target_seq_index is None:
            config.c_out = config.encoder_input_size
        else:
            copnfig.c_out = 1

        self.n_past = config.n_past
        self.start_token_len = config.start_token_len
        self.max_forecast_steps = config.max_forecast_steps

        kernel_size = config.moving_avg
        self.decomp = SeriesDecomposeBlock(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbeddingWoPos(
            config.encoder_input_size, config.model_dim, config.embed, config.ts_encoding, config.dropout
        )

        self.dec_embedding = DataEmbeddingWoPos(
            config.decoder_input_size, config.model_dim, config.embed, config.ts_encoding, config.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.model_dim,
                        config.n_heads,
                    ),
                    config.model_dim,
                    config.fcn_dim,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.num_encoder_layers)
            ],
            norm_layer=SeasonalLayernorm(config.model_dim),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.model_dim,
                        config.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.model_dim,
                        config.n_heads,
                    ),
                    config.model_dim,
                    config.c_out,
                    config.fcn_dim,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.num_decoder_layers)
            ],
            norm_layer=SeasonalLayernorm(config.model_dim),
            projection=nn.Linear(config.model_dim, config.c_out, bias=True),
        )

    def forward(
        self,
        past,
        past_timestamp,
        future_timestamp,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        **kwargs
    ):
        config = self.config

        future_timestamp = torch.cat(
            [past_timestamp[:, (past_timestamp.shape[1] - self.start_token_len) :], future_timestamp], dim=1
        )

        # decomp init
        mean = torch.mean(past, dim=1).unsqueeze(1).repeat(1, self.max_forecast_steps, 1)
        zeros = torch.zeros(
            [past.shape[0], self.max_forecast_steps, past.shape[2]], dtype=torch.float, device=self.device
        )
        seasonal_init, trend_init = self.decomp(past)
        # decoder input
        trend_init = torch.cat([trend_init[:, (trend_init.shape[1] - self.start_token_len) :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, (seasonal_init.shape[1] - self.start_token_len) :, :], zeros], dim=1
        )

        # enc
        enc_out = self.enc_embedding(past, past_timestamp)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, future_timestamp)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.max_forecast_steps :, :]  # [B, L, D]


class AutoformerForecaster(DeepForecaster):
    """
    Implementaion of Autoformer deep forecaster
    """

    config_class = AutoformerConfig
    deep_model_class = AutoformerModel

    def __init__(self, config: AutoformerConfig):
        super().__init__(config)
