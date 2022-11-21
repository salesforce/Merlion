#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Implementation of Transformer for time series data 
    Code adapted from https://github.com/thuml/Autoformer
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


from merlion.models.base import NormalizingConfig
from merlion.models.deep_base import TorchModel
from merlion.models.forecast.deep_models.deep_forecast_base import DeepForecasterConfig, DeepForecaster
from merlion.models.forecast.deep_models.layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from merlion.models.forecast.deep_models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from merlion.models.forecast.deep_models.layers.Embed import DataEmbedding


from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.utils.misc import initializer
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy

logger = logging.getLogger(__name__)


class TransformerConfig(DeepForecasterConfig, NormalizingConfig):
    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        start_token_len: int = 0,
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
        super().__init__(
            n_past=n_past, max_forecast_steps=max_forecast_steps, start_token_len=start_token_len, **kwargs
        )


class TransformerModel(TorchModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        if config.dim is not None:
            config.enc_in = config.dim if config.enc_in is None else config.enc_in
            config.dec_in = config.enc_in if config.dec_in is None else config.dec_in

        if config.target_seq_index is None:
            config.c_out = config.enc_in
        else:
            copnfig.c_out = 1

        self.n_past = config.n_past
        self.start_token_len = config.start_token_len
        self.max_forecast_steps = config.max_forecast_steps

        self.enc_embedding = DataEmbedding(config.enc_in, config.d_model, config.embed, config.ts_freq, config.dropout)

        self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.embed, config.ts_freq, config.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout, output_attention=False),
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
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model,
                        config.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout, output_attention=False),
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
        enc_out = self.enc_embedding(past, past_timestamp)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(past_dec, past_timestamp_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        return dec_out[:, -self.max_forecast_steps :, :]


class TransformerForecaster(DeepForecaster):
    config_class = TransformerConfig
    deep_model_class = TransformerModel

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
