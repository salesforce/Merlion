#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The autoencoder-based anomaly detector for multivariate time series
"""
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

from typing import Sequence

import numpy as np
import pandas as pd

from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.misc import ProgressBar, initializer
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset


class AutoEncoderConfig(DetectorConfig, NormalizingConfig):
    """
    Configuration class for AutoEncoder. The normalization is inherited from `NormalizingConfig`.
    The input data will be standardized automatically.
    """

    _default_threshold = AggregateAlarms(alm_threshold=2.5, abs_score=True)

    @initializer
    def __init__(
        self,
        hidden_size: int = 5,
        layer_sizes: Sequence[int] = (25, 10, 5),
        sequence_len: int = 1,
        lr: float = 1e-3,
        batch_size: int = 512,
        num_epochs: int = 50,
        **kwargs
    ):
        """
        :param hidden_size: The latent size
        :param layer_sizes: The hidden layer sizes for the MLP encoder and decoder,
            e.g., (25, 10, 5) for encoder and (5, 10, 25) for decoder
        :param sequence_len: The input series length, e.g., input = [x(t-sequence_len+1)...,x(t-1),x(t)]
        :param lr: The learning rate during training
        :param batch_size: The batch size during training
        :param num_epochs: The number of training epochs
        """
        super().__init__(**kwargs)


class AutoEncoder(DetectorBase):
    """
    The autoencoder-based multivariate time series anomaly detector.
    This detector utilizes an autoencoder to infer the correlations between
    different time series and estimate the joint distribution of the variables
    for anomaly detection.

    - paper: `Pierre Baldi. Autoencoders, Unsupervised Learning, and Deep Architectures. 2012.
      <https://proceedings.mlr.press/v27/baldi12a.html>`_
    """

    config_class = AutoEncoderConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: AutoEncoderConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.layer_sizes = config.layer_sizes
        self.k = config.sequence_len
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.data_dim = None

    def _build_model(self, dim):
        model = AEModule(input_size=dim * self.k, hidden_size=self.hidden_size, layer_sizes=self.layer_sizes)
        return model

    def _train(self, train_data: pd.DataFrame, train_config=None):
        self.model = self._build_model(train_data.shape[1]).to(self.device)
        self.data_dim = train_data.shape[1]

        loader = RollingWindowDataset(
            train_data,
            target_seq_index=None,
            shuffle=True,
            flatten=False,
            n_past=self.k,
            n_future=0,
            batch_size=self.batch_size,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bar = ProgressBar(total=self.num_epochs)

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for i, (batch, _, _, _) in enumerate(loader):
                batch = torch.tensor(batch, dtype=torch.float, device=self.device)
                loss = self.model.loss(batch)
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
            flatten=False,
            n_past=self.k,
            n_future=0,
            batch_size=self.batch_size,
        )
        scores = []
        for y, _, _, _ in loader:
            y = torch.tensor(y, dtype=torch.float, device=self.device)
            scores.append(self.model(y).cpu().data.numpy())
        scores = np.concatenate([np.ones(self.k - 1) * scores[0][0], *scores])
        return pd.DataFrame(scores[-len(time_series) :], index=time_series.index)


class AEModule(nn.Module):
    """
    The autoencoder module where the encoder and decoder are both MLPs.

    :meta private:
    """

    def __init__(self, input_size, hidden_size, layer_sizes, activation=nn.ReLU, dropout_prob=0.0):
        """
        :param input_size: The input dimension
        :param hidden_size: The latent size of the autoencoder
        :param layer_sizes: The hidden layer sizes for the encoder and decoder
        :param activation: The activation function for the hidden layers
        :param dropout_prob: The dropout rate
        """
        super().__init__()
        self.encoder = MLP(
            input_size=input_size,
            output_size=hidden_size,
            layer_sizes=layer_sizes,
            activation=activation,
            last_layer_activation=activation,
            dropout_prob=dropout_prob,
        )
        self.decoder = MLP(
            input_size=hidden_size,
            output_size=input_size,
            layer_sizes=layer_sizes[::-1],
            activation=activation,
            last_layer_activation=nn.Identity,
            dropout_prob=dropout_prob,
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        y = self.decoder(self.encoder(x))
        return torch.norm(x - y, dim=1)

    def loss(self, x):
        x = torch.flatten(x, start_dim=1)
        y = self.decoder(self.encoder(x))
        loss = self.loss_func(y, x)
        return loss


class MLP(nn.Module):
    """
    The MLP module used in the encoder and decoder

    :meta private:
    """

    def __init__(self, input_size, output_size, layer_sizes, activation, last_layer_activation, dropout_prob):
        """
        :param input_size: The input dimension
        :param output_size: The output dimension
        :param layer_sizes: The hidden layer sizes
        :param activation: The activation function for the hidden layers
        :param last_layer_activation: The activation function for the last layer
        :param dropout_prob: The dropout rate
        """
        super().__init__()
        layers, layer_sizes = [], [input_size] + list(layer_sizes)
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        layers.append(last_layer_activation())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
