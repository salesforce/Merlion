#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Implementation of ETSformer.
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

from merlion.models.utils.nn_modules import ETSEmbedding
from merlion.models.utils.nn_modules.enc_dec_etsformer import EncoderLayer, Encoder, DecoderLayer, Decoder

from merlion.utils.misc import initializer

logger = logging.getLogger(__name__)


class ETSformerConfig(DeepForecasterConfig, NormalizingConfig):
    """
    ETSformer: Exponential Smoothing Transformers for Time-series Forecasting: https://arxiv.org/abs/2202.01381
    Code adapted from https://github.com/salesforce/ETSformer.
    """

    @initializer
    def __init__(
        self,
        n_past,
        max_forecast_steps: int = None,
        encoder_input_size: int = None,
        decoder_input_size: int = None,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        model_dim: int = 512,
        dropout: float = 0.2,
        n_heads: int = 8,
        fcn_dim: int = 2048,
        top_K: int = 1,  # Top-K Fourier bases
        sigma=0.2,
        **kwargs
    ):
        """
        :param n_past: # of past steps used for forecasting future.
        :param max_forecast_steps:  Max # of steps we would like to forecast for.
        :param encoder_input_size: Input size of encoder. If `encoder_input_size = None`, then the model will automatically use `config.dim`,
            which is the dimension of the input data.
        :param decoder_input_size: Input size of decoder. If `decoder_input_size = None`, then the model will automatically use `config.dim`,
            which is the dimension of the input data.
        :param num_encoder_layers: Number of encoder layers.
        :param num_decoder_layers: Number of decoder layers.
        :param model_dim: Dimension of the model.
        :param dropout: dropout rate.
        :param n_heads: Number of heads of the model.
        :param fcn_dim: Hidden dimension of the MLP layer in the model.
        :param top_K: Top-K Frequent Fourier basis.
        :param sigma: Standard derivation for ETS input data transform.
        """
        super().__init__(n_past=n_past, max_forecast_steps=max_forecast_steps, **kwargs)
        assert self.start_token_len == 0, "No need of start token for ETSformer!"


class ETSformerModel(TorchModel):
    """
    Implementaion of ETSformer deep torch model.
    """

    def __init__(self, config: ETSformerConfig):
        super().__init__(config)

        assert (
            config.num_encoder_layers == config.num_decoder_layers
        ), "The number of encoder and decoder layers must be equal!"
        if config.dim is not None:
            config.encoder_input_size = config.dim if config.encoder_input_size is None else config.encoder_input_size
            config.decoder_input_size = (
                config.encoder_input_size if config.decoder_input_size is None else config.decoder_input_size
            )

        config.c_out = config.encoder_input_size

        self.n_past = config.n_past
        self.max_forecast_steps = config.max_forecast_steps

        self.enc_embedding = ETSEmbedding(config.encoder_input_size, config.model_dim, dropout=config.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    config.model_dim,
                    config.n_heads,
                    config.c_out,
                    config.n_past,
                    config.max_forecast_steps,
                    config.top_K,
                    dim_feedforward=config.fcn_dim,
                    dropout=config.dropout,
                    output_attention=False,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    config.model_dim,
                    config.n_heads,
                    config.c_out,
                    config.max_forecast_steps,
                    dropout=config.dropout,
                    output_attention=False,
                )
                for _ in range(config.num_decoder_layers)
            ],
        )

    def forward(
        self,
        past,
        past_timestamp,
        future_timestamp,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        attention=False,
        **kwargs
    ):
        with torch.no_grad():
            if self.training:
                past = self.transform(past)
        res = self.enc_embedding(past)
        level, growths, seasons, season_attns, growth_attns = self.encoder(res, past, attn_mask=enc_self_mask)

        growth, season, growth_dampings = self.decoder(growths, seasons)

        preds = level[:, -1:] + growth + season

        # maybe remove later
        if attention:
            decoder_growth_attns = []
            for growth_attn, growth_damping in zip(growth_attns, growth_dampings):
                decoder_growth_attns.append(torch.einsum("bth,oh->bhot", [growth_attn.squeeze(-1), growth_damping]))

            season_attns = torch.stack(season_attns, dim=0)[:, :, -self.pred_len :]
            season_attns = reduce(season_attns, "l b d o t -> b o t", reduction="mean")
            decoder_growth_attns = torch.stack(decoder_growth_attns, dim=0)[:, :, -self.pred_len :]
            decoder_growth_attns = reduce(decoder_growth_attns, "l b d o t -> b o t", reduction="mean")
            return preds, season_attns, decoder_growth_attns

        if self.config.target_seq_index is not None:
            return preds[:, :, :1]
        else:
            return preds

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.config.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.config.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.config.sigma)


class ETSformerForecaster(DeepForecaster):
    """
    Implementaion of ETSformer deep forecaster.
    """

    config_class = ETSformerConfig
    deep_model_class = ETSformerModel

    def __init__(self, config: ETSformerConfig):
        super().__init__(config)
