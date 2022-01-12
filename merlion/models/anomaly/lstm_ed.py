#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The LSTM-encoder-decoder-based anomaly detector for multivariate time series
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


class LSTMEDConfig(DetectorConfig, NormalizingConfig):
    """
    Configuration class for LSTM-encoder-decoder. The normalization is inherited from `NormalizingConfig`.
    The input data will be standardized automatically.
    """

    _default_threshold = AggregateAlarms(alm_threshold=2.5, abs_score=True)

    @initializer
    def __init__(
        self,
        hidden_size: int = 5,
        sequence_len: int = 20,
        n_layers: Sequence[int] = (1, 1),
        dropout: Sequence[int] = (0, 0),
        lr: float = 1e-3,
        batch_size: int = 256,
        num_epochs: int = 10,
        **kwargs
    ):
        """
        :param hidden_size: The hidden state size of the LSTM modules
        :param sequence_len: The input series length, e.g., input = [x(t-sequence_len+1)...,x(t-1),x(t)]
        :param n_layers: The number of layers for the LSTM encoder and decoder. ``n_layer`` has two values, i.e.,
            ``n_layer[0]`` is the number of encoder layers and ``n_layer[1]`` is the number of decoder layers.
        :param dropout: The dropout rate for the LSTM encoder and decoder. ``dropout`` has two values, i.e.,
            ``dropout[0]`` is the dropout rate for the encoder and ``dropout[1]`` is the dropout rate for the decoder.
        :param lr: The learning rate during training
        :param batch_size: The batch size during training
        :param num_epochs: The number of training epochs
        """
        super().__init__(**kwargs)


class LSTMED(DetectorBase):
    """
    The LSTM-encoder-decoder-based multivariate time series anomaly detector.
    The time series representation is modeled by an encoder-decoder network where
    both encoder and decoder are LSTMs. The distribution of the reconstruction error
    is estimated for anomaly detection.
    """

    config_class = LSTMEDConfig

    def __init__(self, config: LSTMEDConfig):
        super().__init__(config)
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.hidden_size = config.hidden_size
        self.sequence_length = config.sequence_len
        self.n_layers = config.n_layers
        self.dropout = config.dropout

        assert (
            len(self.n_layers) == 2
        ), "Param n_layers should contain two values: (num_layers for LSTM encoder, num_layers for LSTM decoder)"
        assert len(self.n_layers) == len(self.dropout), "Param dropout should contain two values"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lstmed = None
        self.data_dim = None

    def _build_model(self, dim):
        return LSTMEDModule(dim, self.hidden_size, self.n_layers, self.dropout, self.device)

    def _train(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        train_x = InputData(X, k=self.sequence_length)
        train_loader = DataLoader(
            dataset=train_x, batch_size=self.batch_size, shuffle=True, collate_fn=InputData.collate_func
        )
        self.data_dim = X.shape[1]
        self.lstmed = self._build_model(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss(reduction="sum")
        bar = ProgressBar(total=self.num_epochs)

        self.lstmed.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                output = self.lstmed(batch)
                loss = loss_func(output, batch.float())
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            if bar is not None:
                bar.print(epoch + 1, prefix="", suffix="Complete, Loss {:.4f}".format(total_loss / len(train_loader)))

    def _detect(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        data_loader = DataLoader(
            dataset=InputData(X, k=self.sequence_length), batch_size=self.batch_size, shuffle=False
        )
        self.lstmed.eval()
        scores, outputs = [], []
        for idx, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            output = self.lstmed(batch)
            error = nn.L1Loss(reduction="none")(output, batch.float())
            score = np.mean(error.view(-1, X.shape[1]).data.cpu().numpy(), axis=1)
            scores.append(score.reshape(batch.shape[0], self.sequence_length))

        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i : i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)
        return scores

    def _get_sequence_len(self):
        return self.sequence_length

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


class LSTMEDModule(nn.Module):
    """
    The LSTM-encoder-decoder module. Both the encoder and decoder are LSTMs.

    :meta private:
    """

    def __init__(self, n_features, hidden_size, n_layers, dropout, device):
        """
        :param n_features: The input feature dimension
        :param hidden_size: The LSTM hidden size
        :param n_layers: The number of LSTM layers
        :param dropout: The dropout rate
        :param device: CUDA or CPU
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[0],
            bias=True,
            dropout=self.dropout[0],
        )
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[1],
            bias=True,
            dropout=self.dropout[1],
        )
        self.output_layer = nn.Linear(self.hidden_size, self.n_features)

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x, return_latent=False):
        # Encoder
        enc_hidden = self.init_hidden_state(x.shape[0])
        _, enc_hidden = self.encoder(x.float(), enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        return (output, enc_hidden[1][-1]) if return_latent else output
