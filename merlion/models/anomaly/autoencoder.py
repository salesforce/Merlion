#
# Copyright (c) 2022 salesforce.com, inc.
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

import numpy as np
from typing import Sequence
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.misc import ProgressBar, initializer
from merlion.models.anomaly.utils import InputData, batch_detect


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
      <http://proceedings.mlr.press/v27/baldi12a.html>`_
    """

    config_class = AutoEncoderConfig

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

    def _train(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        self.model = self._build_model(X.shape[1]).to(self.device)
        self.data_dim = X.shape[1]

        input_data = InputData(X, self.k)
        train_data = DataLoader(input_data, batch_size=self.batch_size, shuffle=True, collate_fn=InputData.collate_func)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bar = ProgressBar(total=self.num_epochs)

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for i, batch in enumerate(train_data):
                batch = batch.to(self.device)
                loss = self.model.loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            if bar is not None:
                bar.print(epoch + 1, prefix="", suffix="Complete, Loss {:.4f}".format(total_loss / len(train_data)))

    def _detect(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        self.model.eval()
        test_data = torch.FloatTensor([X[i + 1 - self.k : i + 1, :] for i in range(self.k - 1, X.shape[0])]).to(
            self.device
        )

        scores = np.zeros((X.shape[0],), dtype=float)
        test_scores = self.model(test_data).cpu().data.numpy()
        scores[self.k - 1 :] = test_scores
        scores[: self.k - 1] = test_scores[0]
        return scores

    def _get_sequence_len(self):
        return self.k

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        """
        Train a multivariate time series anomaly detector.

        :param train_data: A `TimeSeries` of metric values to train the model.
        :param anomaly_labels: A `TimeSeries` indicating which timestamps are
            anomalous. Optional.
        :param train_config: Additional training configs, if needed. Only
            required for some models.
        :param post_rule_train_config: The config to use for training the
            model's post-rule. The model's default post-rule train config is
            used if none is supplied here.

        :return: A `TimeSeries` of the model's anomaly scores on the training
            data.
        """
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)

        train_df = train_data.align().to_pd()
        self._train(train_df.values)
        scores = batch_detect(self, train_df.values)

        train_scores = TimeSeries({"anom_score": UnivariateTimeSeries(train_data.time_stamps, scores)})
        self.train_post_rule(
            anomaly_scores=train_scores, anomaly_labels=anomaly_labels, post_rule_train_config=post_rule_train_config
        )
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        """
        :param time_series: The `TimeSeries` we wish to predict anomaly scores for.
        :param time_series_prev: A `TimeSeries` immediately preceding ``time_series``.
        :return: A univariate `TimeSeries` of anomaly scores
        """
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
        ts = time_series_prev + time_series if time_series_prev is not None else time_series
        scores = batch_detect(self, ts.align().to_pd().values)
        timestamps = time_series.time_stamps
        return TimeSeries({"anom_score": UnivariateTimeSeries(timestamps, scores[-len(timestamps) :])})


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
