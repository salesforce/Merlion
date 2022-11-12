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

from merlion.models.deep_base import TorchModel
from merlion.models.forecast.deep_models.deep_forecast_base import DeepForecasterConfig, DeepForecaster
from merlion.models.forecast.deep_models.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from merlion.models.forecast.deep_models.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from merlion.models.forecast.deep_models.layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)


from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.utils.misc import initializer
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy

logger = logging.getLogger(__name__)


class AutoformerConfig(DeepForecasterConfig):
    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        dim: int = None,
        start_token_len: int = 0,
        moving_avg: int = 25,
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
        **kwargs
    ):
        if enc_in is None:
            self.enc_in = dim

        super().__init__(
            n_past=n_past, max_forecast_steps=max_forecast_steps, start_token_len=start_token_len, dim=dim, **kwargs
        )
        # TODO: fix this later
        # Setting proper output decoder dimension
        if self.dec_in is None:
            if self.target_seq_index is None:
                self.dec_in = self.enc_in
            else:
                self.dec_in = 1


class AutoformerModel(TorchModel):
    def __init__(self, config: AutoformerConfig):
        super().__init__(config)

        self.n_past = config.n_past
        self.start_token_len = config.start_token_len
        self.max_forecast_steps = config.max_forecast_steps

        # TODO: FIXME, the best way to get dimenion

        self.dim = config.dim = config.enc_in

        kernel_size = config.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            config.enc_in, config.d_model, config.embed, config.ts_freq, config.dropout
        )

        self.dec_embedding = DataEmbedding_wo_pos(
            config.dec_in, config.d_model, config.embed, config.ts_freq, config.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=my_Layernorm(config.d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.dim,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.d_layers)
            ],
            norm_layer=my_Layernorm(config.d_model),
            projection=nn.Linear(config.d_model, config.dim, bias=True),
        )

    def forward(
        self,
        past,
        past_timestamp,
        past_dec,
        past_timestamp_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        **kwargs
    ):

        # decomp init
        mean = torch.mean(past, dim=1).unsqueeze(1).repeat(1, self.max_forecast_steps, 1)
        zeros = torch.zeros([past_dec.shape[0], self.max_forecast_steps, past_dec.shape[2]], device=past.device)
        seasonal_init, trend_init = self.decomp(past)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.start_token_len :, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.start_token_len :, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(past, past_timestamp)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, past_timestamp_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.max_forecast_steps :, :]  # [B, L, D]


class AutoformerForecaster(DeepForecaster):
    config_class = AutoformerConfig
    deep_model_class = AutoformerModel

    def __init__(self, config: AutoformerConfig):
        super().__init__(config)
