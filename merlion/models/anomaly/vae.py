#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The VAE-based anomaly detector for multivariate time series
"""
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.misc import ProgressBar, initializer
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset


class VAEConfig(DetectorConfig, NormalizingConfig):
    """
    Configuration class for VAE. The normalization is inherited from `NormalizingConfig`.
    The input data will be standardized automatically.
    """

    _default_threshold = AggregateAlarms(alm_threshold=2.5, abs_score=True)

    @initializer
    def __init__(
        self,
        encoder_hidden_sizes: Sequence[int] = (25, 10, 5),
        decoder_hidden_sizes: Sequence[int] = (5, 10, 25),
        latent_size: int = 5,
        sequence_len: int = 1,
        kld_weight: float = 1.0,
        dropout_rate: float = 0.0,
        num_eval_samples: int = 10,
        lr: float = 1e-3,
        batch_size: int = 1024,
        num_epochs: int = 10,
        **kwargs
    ):
        """
        :param encoder_hidden_sizes: The hidden layer sizes of the MLP encoder
        :param decoder_hidden_sizes: The hidden layer sizes of the MLP decoder
        :param latent_size: The latent size
        :param sequence_len: The input series length, e.g., input = [x(t-sequence_len+1)...,x(t-1),x(t)]
        :param kld_weight: The regularization weight for the KL divergence term
        :param dropout_rate: The dropout rate for the encoder and decoder
        :param num_eval_samples: The number of sampled latent variables during prediction
        :param lr: The learning rate during training
        :param batch_size: The batch size during training
        :param num_epochs: The number of training epochs
        """
        super().__init__(**kwargs)


class VAE(DetectorBase):
    """
    The VAE-based multivariate time series anomaly detector.
    This detector utilizes a variational autoencoder to infer the correlations between
    different time series and estimate the distribution of the reconstruction errors
    for anomaly detection.

    - paper: `Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. 2013.
      <https://arxiv.org/abs/1312.6114>`_
    """

    config_class = VAEConfig

    def __init__(self, config: VAEConfig):
        super().__init__(config)
        self.k = config.sequence_len
        self.encoder_hidden_sizes = config.encoder_hidden_sizes
        self.decoder_hidden_sizes = config.decoder_hidden_sizes
        self.latent_size = config.latent_size
        self.activation = nn.ReLU
        self.kld_weight = config.kld_weight
        self.dropout_rate = config.dropout_rate
        self.num_eval_samples = config.num_eval_samples
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.lr = config.lr

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.data_dim = None

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def _build_model(self, dim):
        model = CVAE(
            x_dim=dim * self.k,
            c_dim=0,
            encoder_hidden_sizes=self.encoder_hidden_sizes,
            decoder_hidden_sizes=self.decoder_hidden_sizes,
            latent_size=self.latent_size,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )
        return model

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        self.model = self._build_model(train_data.shape[1]).to(self.device)
        self.data_dim = train_data.shape[1]

        loader = RollingWindowDataset(
            train_data,
            target_seq_index=None,
            shuffle=True,
            flatten=True,
            n_past=self.k,
            n_future=0,
            batch_size=self.batch_size,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        bar = ProgressBar(total=self.num_epochs)

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for i, (batch, _, _, _) in enumerate(loader):
                x = torch.tensor(batch, dtype=torch.float, device=self.device)
                recon_x, mu, log_var, _ = self.model(x, None)
                recon_loss = loss_func(x, recon_x)
                kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
                loss = recon_loss + kld_loss * self.kld_weight
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            if bar is not None:
                bar.print(epoch + 1, prefix="", suffix="Complete, Loss {:.4f}".format(total_loss / len(train_data)))
        return self._get_anomaly_score(train_data)

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        self.model.eval()
        ts = pd.concat((time_series_prev, time_series)) if time_series_prev is None else time_series
        loader = RollingWindowDataset(
            ts,
            target_seq_index=None,
            shuffle=False,
            flatten=True,
            n_past=self.k,
            n_future=0,
            batch_size=self.batch_size,
        )
        ys, rs = [], []
        for y, _, _, _ in loader:
            ys.append(y)
            y = torch.tensor(y, dtype=torch.float, device=self.device)
            r = np.zeros(y.shape)
            for _ in range(self.num_eval_samples):
                recon_y, _, _, _ = self.model(y, None)
                r += recon_y.cpu().data.numpy()
            r /= self.num_eval_samples
            rs.append(r)

        scores = np.zeros((ts.shape[0],), dtype=float)
        test_scores = np.sum(np.abs(np.concatenate(rs) - np.concatenate(ys)), axis=1)
        scores[self.k - 1 :] = test_scores
        scores[: self.k - 1] = test_scores[0]
        return pd.DataFrame(scores[-len(time_series) :], index=time_series.index)


class CVAE(nn.Module):
    """
    Conditional variational autoencoder.

    - paper: `Kihyuk Sohn and Honglak Lee and Xinchen Yan.
      Learning Structured Output Representation using Deep Conditional Generative Models. 2015.
      <https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html>`_

    :meta private:
    """

    def __init__(
        self,
        x_dim,
        c_dim,
        encoder_hidden_sizes,
        decoder_hidden_sizes,
        latent_size,
        dropout_rate=0.0,
        activation=nn.ReLU,
    ):
        """
        :param x_dim: The input variable dimension
        :param c_dim: The conditioned variable dimension
        :param encoder_hidden_sizes: The hidden layer sizes for the encoder
        :param decoder_hidden_sizes: The hidden layer sizes for the decoder
        :param latent_size: The latent size for both the encoder and decoder
        :param dropout_rate: The dropout rate
        :param activation: The activation functions for the hidden layers
        """
        super().__init__()
        self.encoder = Encoder(x_dim, c_dim, encoder_hidden_sizes, latent_size, dropout_rate, activation)
        self.decoder = Decoder(x_dim, c_dim, decoder_hidden_sizes, latent_size, dropout_rate, activation)

    def forward(self, x, c):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        return self.decoder(z, c)


class Encoder(nn.Module):
    """
    The encoder for the conditional VAE model

    :meta private:
    """

    def __init__(self, x_dim, c_dim, hidden_sizes, latent_size, dropout_rate, activation):
        """
        :param x_dim: The input variable dimension
        :param c_dim: The conditioned variable dimension
        :param hidden_sizes: The hidden layer sizes
        :param latent_size: The latent size
        :param dropout_rate: The dropout rate
        :param activation: The activation function for the hidden layers
        """
        super().__init__()
        assert len(hidden_sizes) > 0, "hidden sizes cannot be empty"
        self.mlp = build_hidden_layers(x_dim + c_dim, hidden_sizes, dropout_rate, activation)
        self.linear_means = nn.Linear(hidden_sizes[-1], latent_size)
        self.linear_vars = nn.Linear(hidden_sizes[-1], latent_size)
        self._init_log_var_weights()

    def _init_log_var_weights(self):
        torch.nn.init.uniform_(self.linear_vars.weight, -0.01, 0.01)
        torch.nn.init.constant_(self.linear_vars.bias, 0)

    def forward(self, x, c):
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        x = self.mlp(x)
        means = self.linear_means(x)
        log_vars = self.linear_vars(x)
        return means, log_vars


class Decoder(nn.Module):
    """
    The decoder for the conditional VAE model

    :meta private:
    """

    def __init__(self, x_dim, c_dim, hidden_sizes, latent_size, dropout_rate, activation):
        """
        :param x_dim: The input variable dimension
        :param c_dim: The conditioned variable dimension
        :param hidden_sizes: The hidden layer sizes
        :param latent_size: The latent size
        :param dropout_rate: The dropout rate
        :param activation: The activation function for the hidden layers
        """
        super().__init__()
        assert len(hidden_sizes) > 0, "hidden sizes cannot be empty"
        self.mlp = build_hidden_layers(latent_size + c_dim, hidden_sizes, dropout_rate, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], x_dim)

    def forward(self, z, c):
        if c is not None:
            z = torch.cat([z, c], dim=-1)
        return self.output_layer(self.mlp(z))


def build_hidden_layers(input_size, hidden_sizes, dropout_rate, activation):
    """
    :meta private:
    """
    hidden_layers = []
    for i in range(len(hidden_sizes)):
        s = input_size if i == 0 else hidden_sizes[i - 1]
        hidden_layers.append(nn.Linear(s, hidden_sizes[i]))
        hidden_layers.append(activation())
        hidden_layers.append(nn.Dropout(dropout_rate))
    return torch.nn.Sequential(*hidden_layers)
