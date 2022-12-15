#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Implementation of Informer: 
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting: https://arxiv.org/abs/2012.07436
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

from merlion.utils.misc import initializer

from merlion.models.base import NormalizingConfig
from merlion.models.deep_base import TorchModel
from merlion.models.forecast.deep_base import DeepForecasterConfig, DeepForecaster

from merlion.models.utils.nn_modules import ProbAttention, AttentionLayer, DataEmbedding, ConvLayer
from merlion.models.utils.nn_modules.enc_dec_transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)


logger = logging.getLogger(__name__)


class InformerConfig(DeepForecasterConfig, NormalizingConfig):
    """
    Config object for informer forecaster
    """

    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        enc_in: int = None,
        dec_in: int = None,
        e_layers: int = 2,
        d_layers: int = 1,
        factor: int = 3,
        d_model: int = 512,
        embed: str = "timeF",
        dropout: float = 0.05,
        activation: str = "gelu",
        n_heads: int = 8,
        d_ff: int = 2048,
        distil: bool = True,
        **kwargs
    ):
        """
        :param n_past: # of past steps used for forecasting future.
        :param max_forecast_steps:  Max # of steps we would like to forecast for.
        :param enc_in: Input size of encoder. If `enc_in = None`, then the model will automatically use `config.dim`,
            which is the dimension of the input data.
        :param dec_in: Input size of decoder. If `dec_in = None`, then the model will automatically use `config.dim`,
            which is the dimension of the input data.
        :param e_layers: Number of encoder layers.
        :param d_layers: Number of decoder layers.
        :param factor: Attention factor.
        :param d_model: Dimension of the model.
        :param embed: Time feature encoding type, options include `timeF`, `fixed` and `learned`.
        :param dropout: dropout rate.
        :param activation: Activation function, can be `gelu`, `relu`, `sigmoid`, etc.
        :param n_heads: Number of heads of the model.
        :param d_ff: Hidden dimension of the MLP layer in the model.
        :param distil: whether to use distilling in the encoder of the model.
        """

        super().__init__(n_past=n_past, max_forecast_steps=max_forecast_steps, **kwargs)


class InformerModel(TorchModel):
    """
    Implementaion of informer Deep Torch Model
    """

    def __init__(self, config: InformerConfig):
        super().__init__(config)

        if config.dim is not None:
            config.enc_in = config.dim if config.enc_in is None else config.enc_in
            config.dec_in = config.enc_in if config.dec_in is None else config.dec_in

        if config.target_seq_index is None:
            config.c_out = config.enc_in
        else:
            config.c_out = 1

        self.n_past = config.n_past
        self.start_token_len = config.start_token_len
        self.max_forecast_steps = config.max_forecast_steps

        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.embed, config.ts_encoding, config.dropout
        )

        self.dec_embedding = DataEmbedding(
            config.dec_in, config.d_model, config.embed, config.ts_encoding, config.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            [ConvLayer(config.d_model) for l in range(config.e_layers - 1)] if config.distil else None,
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
            projection=nn.Linear(config.d_model, config.c_out, bias=True),
        )

        self.config = config

    def forward(
        self,
        past,
        past_timestamp,
        future,
        future_timestamp,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        **kwargs
    ):
        config = self.config

        # if future is None, we only need to do inference
        if future is None:
            start_token = past[:, past.shape[1] - config.start_token_len :]
            dec_inp = torch.zeros(past.shape[0], config.max_forecast_steps, config.dec_in).float().to(self.device)
            dec_inp = torch.cat([start_token, dec_inp], dim=1)
        else:
            dec_inp = torch.zeros_like(future[:, -config.max_forecast_steps :, :]).float().to(self.device)
            dec_inp = torch.cat([future[:, : config.start_token_len, :], dec_inp], dim=1)

        enc_out = self.enc_embedding(past, past_timestamp)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(dec_inp, future_timestamp)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        return dec_out[:, -self.max_forecast_steps :, :]


class InformerForecaster(DeepForecaster):
    """
    Implementaion of Informer deep forecaster
    """

    config_class = InformerConfig
    deep_model_class = InformerModel

    def __init__(self, config: InformerConfig):
        super().__init__(config)
