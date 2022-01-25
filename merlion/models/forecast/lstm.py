#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
A forecaster based on a LSTM neural net.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

import bisect
from copy import deepcopy
import datetime
import logging
import os
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.transform.normalize import MeanVarNormalize
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.resample import TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils.time_series import assert_equal_timedeltas, TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class LSTMConfig(ForecasterConfig):
    """
    Configuration class for `LSTM`.
    """

    _default_transform = TransformSequence(
        [
            TemporalResample(granularity=None, trainable_granularity=True),
            DifferenceTransform(),
            MeanVarNormalize(normalize_bias=True, normalize_scale=True),
        ]
    )

    def __init__(self, max_forecast_steps: int, nhid=1024, model_strides=(1,), **kwargs):
        """
        :param nhid: hidden dimension of LSTM
        :param model_strides: tuple indicating the stride(s) at which we would
            like to subsample the input data before giving it to the model.
        """
        self.model_strides = list(model_strides)
        self.nhid = nhid
        super().__init__(max_forecast_steps=max_forecast_steps, **kwargs)


class LSTMTrainConfig(object):
    """
    LSTM training configuration.
    """

    def __init__(
        self,
        lr=1e-5,
        batch_size=128,
        epochs=128,
        seq_len=256,
        data_stride=1,
        valid_split=0.2,
        checkpoint_file="checkpoint.pt",
    ):
        assert 0 < valid_split < 1
        self.lr = lr
        self.batch_size = batch_size  # 8
        self.epochs = epochs
        self.seq_len = seq_len
        self.data_stride = data_stride
        self.checkpoint_file = checkpoint_file
        self.valid_split = valid_split


class Corpus(Dataset):
    """
    Build a torch corpus from an input sequence

    :meta private:
    """

    def __init__(self, sequence, seq_len=32, stride=1):
        """
        :param sequence: a list of items
        :param seq_len: the sequence length used in the LSTM models
        :param stride: stride if you want to subsample the sequence up front
        """
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.sequence = sequence
        if len(self) == 0:
            raise RuntimeError(
                f"Zero length dataset! This typically occurs when "
                f"seq_len > len(sequence). Here seq_len={seq_len}, "
                f"len(sequence)={len(sequence)}."
            )
        logger.info(f"Dataset length: {len(self)}")

    def __len__(self):
        n = len(self.sequence) - (self.seq_len - 1) * self.stride
        return max(0, n)

    def __getitem__(self, idx):
        max_idx = idx + (self.seq_len - 1) * self.stride + 1
        return torch.FloatTensor(self.sequence[idx : max_idx : self.stride])


class _LSTMBase(nn.Module):
    """
    Two layer LSTM + a linear output layer. The model assumes equal time
    intervals across the whole input sequence, so time stamps are ignored.

    :meta private:
    """

    def __init__(self, nhid=51):
        """
        :param nhid: number of hidden neurons in each of the LSTM cells
        """
        super().__init__()
        self.nhid = nhid
        self.add_module("lstm1", nn.LSTMCell(1, self.nhid))
        self.add_module("lstm2", nn.LSTMCell(self.nhid, self.nhid))
        self.add_module("linear", nn.Linear(self.nhid, 1))
        self.h_t, self.c_t, self.h_t2, self.c_t2 = None, None, None, None

    def forward(self, input):
        outputs = []
        self.reset(bsz=input.size(0))

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            self.h_t, self.c_t = self.lstm1(input_t, (self.h_t, self.c_t))
            self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
            output = self.linear(self.h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def generate(self, input):
        self.h_t, self.c_t = self.lstm1(input, (self.h_t, self.c_t))
        self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
        output = self.linear(self.h_t2)
        return torch.stack([output], 1).squeeze(2)

    def reset(self, bsz):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.h_t = torch.zeros(bsz, self.nhid, dtype=torch.float, device=device)
        self.c_t = torch.zeros(bsz, self.nhid, dtype=torch.float, device=device)
        self.h_t2 = torch.zeros(bsz, self.nhid, dtype=torch.float, device=device)
        self.c_t2 = torch.zeros(bsz, self.nhid, dtype=torch.float, device=device)


class _LSTMMultiScale(nn.Module):
    """
    Multi-Scale LSTM Modeling
    the model decomposes the input sequence using different granularities specified in strides
    for each granularity it models the sequence using a two-layer LSTM
    then the output from all the models are summed up to produce the result

    :meta private:
    """

    def __init__(self, strides=(1, 16, 32), nhid=51):
        """
        :param strides: an iterable of strides
        :param nhid: number of hidden neurons used in the LSTM cell
        """
        super().__init__()
        self.strides = strides
        self.nhid = nhid
        self.rnns = nn.ModuleList([_LSTMBase(nhid=self.nhid) for _ in strides])

    def forward(self, input, future=0):
        """
        :param input: batch_size * sequence_length
        :param future: number of future steps for forecasting
        :return: the predicted values including both 1-step predictions and the future step predictions
        """
        outputs = [rnn(input[:, ::stride]) for stride, rnn in zip(self.strides, self.rnns)]
        batch_sz, dim = outputs[0].shape
        preds = [
            output.view(batch_sz, -1, 1).repeat(1, 1, stride).view(batch_sz, -1)[:, :dim]
            for output, stride in zip(outputs, self.strides)
        ]

        outputs = torch.stack(preds, dim=2).sum(dim=2)
        futures = []
        prev = outputs[:, -1].view(batch_sz, -1)

        preds = [x[:, -1].view(batch_sz, -1) for x in preds]

        for i in range(future):
            for j, (stride, rnn) in enumerate(zip(self.strides, self.rnns)):
                if (i + dim) % stride == 0:
                    preds[j] = rnn.generate(prev)

            prev = torch.stack(preds, dim=2).sum(dim=2)
            futures.append(prev)
        futures = torch.cat(futures, dim=1)
        return torch.cat([outputs, futures], dim=1)


def auto_stride(time_stamps, resolution=48):
    """
    automatically set the sequence stride
    experiments show LSTM does not work when the input sequence has super long period
    in this case we may need to subsample the sequence so that the period is not too long
    this function returns a stride suitable for LSTM modeling given the model period is daily.

    :param time_stamps: a list of UTC timestamps (in seconds)
    :param resolution: maximum number of points in each day. (default to 48 so that it is a 30 min prediction)
    :return: the selected stride

    :meta private:
    """
    day_delta = datetime.timedelta(days=1).total_seconds()
    start_day = bisect.bisect_left(time_stamps, time_stamps[-1] - day_delta)
    day_stamps = len(time_stamps) - start_day
    stride = day_stamps // resolution
    return stride


class LSTM(ForecasterBase):
    """
    LSTM forecaster: this assume the input time series has equal intervals across all its values
    so that we can use sequence modeling to make forecast.
    """

    config_class = LSTMConfig
    _default_train_config = LSTMTrainConfig()

    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        self.model = _LSTMMultiScale(strides=config.model_strides, nhid=config.nhid)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = None
        self.seq_len = None
        self._forecast = [0.0 for _ in range(self.max_forecast_steps)]

    def train(self, train_data: TimeSeries, train_config: LSTMTrainConfig = None) -> Tuple[TimeSeries, None]:
        if train_config is None:
            train_config = deepcopy(self._default_train_config)

        orig_train_data = train_data
        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)
        train_data = train_data.univariates[self.target_name]
        train_values = train_data.np_values

        valid_len = int(np.ceil(len(train_data) * train_config.valid_split))

        stride = train_config.data_stride
        self.seq_len = train_config.seq_len

        # Check to make sure the training data is well-formed
        assert_equal_timedeltas(train_data)
        i0 = (len(train_data) - 1) % stride
        self.last_train_time = train_data[-1][0]
        self.timedelta = (train_data[1][0] - train_data[0][0]) * stride

        #############
        train_scores = train_values[:-valid_len]
        _train_data = Corpus(sequence=train_scores, seq_len=self.seq_len, stride=stride)
        train_dataloader = DataLoader(_train_data, batch_size=train_config.batch_size, shuffle=True)

        ###############
        valid_scores = train_values[-valid_len:]
        _valid_data = Corpus(sequence=valid_scores, seq_len=self.seq_len, stride=stride)
        valid_dataloader = DataLoader(_valid_data, batch_size=train_config.batch_size, shuffle=False)

        ################
        no_progress_count = 0
        loss_best = 1e20
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=train_config.lr, momentum=0.9)

        for epoch in range(1, train_config.epochs + 1):
            self.model.train()
            total_loss = 0
            with tqdm(total=len(train_dataloader)) as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    self.optimizer.zero_grad()
                    out = self.model(batch[:, : -(self.max_forecast_steps + 1)], future=self.max_forecast_steps)
                    loss = F.l1_loss(out, batch[:, 1:])
                    loss.backward()

                    self.optimizer.step()
                    pbar.update(1)
                    total_loss += loss.item()
                    loss = total_loss / (batch_idx + 1)
                    pbar.set_description(f"Epoch {epoch}|mae={loss:.4f}")

            # Validate model (n-step prediction) after this epoch
            loss, count = 0, 0
            self.model.eval()
            with torch.no_grad():
                for batch in valid_dataloader:
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    feat = batch[:, : -(self.max_forecast_steps + 1)]
                    target = batch[:, -self.max_forecast_steps :]
                    out = self.model(feat, future=self.max_forecast_steps)
                    out = out[:, -self.max_forecast_steps :]
                    loss += F.l1_loss(out, target, reduction="sum").item()
                    count += target.shape[0] * target.shape[1]

            loss_eval = loss / count
            logger.info(f"val |mae={loss_eval:.4f}")

            if loss_eval < loss_best:
                logger.info(f"saving model |epoch={epoch} |mae={loss_eval:.4f}")
                dirname = os.path.dirname(train_config.checkpoint_file)
                if len(dirname) > 0:
                    os.makedirs(dirname, exist_ok=True)
                torch.save(self.model.state_dict(), train_config.checkpoint_file)
                loss_best = loss_eval
            else:
                no_progress_count += 1

            if no_progress_count > 64:
                logger.info("Dividing learning rate by 10")
                self.optimizer.param_groups[0]["lr"] /= 10.0
                no_progress_count = 0

        state_dict = torch.load(train_config.checkpoint_file, map_location=lambda storage, loc: storage)
        os.remove(train_config.checkpoint_file)
        self.model: _LSTMMultiScale
        self.model.load_state_dict(state_dict)
        for rnn in self.model.rnns:
            rnn.h_t_default = rnn.h_t
            rnn.c_t_default = rnn.c_t
            rnn.h_t2_default = rnn.h_t2
            rnn.c_t2.default = rnn.c_t2

        if not isinstance(self.transform, TransformSequence):
            self.transform = TransformSequence([self.transform])
        done = False
        for f in self.transform.transforms:
            if isinstance(f, TemporalResample):
                f.granularity = self.timedelta
                f.origin = train_data.np_time_stamps[i0]
                f.trainable_granularity = False
                done = True
        if not done:
            self.transform.append(
                TemporalResample(
                    granularity=self.timedelta, origin=train_data.np_time_stamps[i0], trainable_granularity=False
                )
            )

        # FORECASTING: forecast for next n steps using lstm model
        # since we've updated the transform's granularity, re-apply it on
        # the original train data before proceeding.
        ts = self.transform(orig_train_data)
        ts = ts.univariates[self.target_name]
        vals = torch.FloatTensor([ts.np_values])
        if torch.cuda.is_available():
            vals = vals.cuda()

        with torch.no_grad():
            n = self.max_forecast_steps
            preds = self.model(vals[:, :-n], future=n).squeeze().tolist()
            self._forecast = self.model(vals, future=n).squeeze().tolist()[-n:]

        return UnivariateTimeSeries(ts.index, preds, self.target_name).to_ts(), None

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr=False,
        return_prev=False,
    ) -> Tuple[TimeSeries, None]:
        assert not return_iqr, "LSTM does not support uncertainty estimates"

        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)
        n = len(time_stamps)

        if time_series_prev is None:
            yhat = self._forecast[:n]
            yhat = UnivariateTimeSeries(time_stamps, yhat, self.target_name)
            return yhat.to_ts().align(reference=orig_t), None

        # TODO: should we truncate time_series_prev to just the last
        #      (self.seq_len - self.max_forecast_steps) time steps?
        #      This would better match the training distribution
        time_series_prev = self.transform(time_series_prev)
        if time_series_prev.dim != 1:
            raise RuntimeError(
                f"Transform {self.transform} transforms data into a multi-"
                f"variate time series, but model {type(self).__name__} can "
                f"only handle uni-variate time series. Change the transform."
            )

        k = time_series_prev.names[self.target_seq_index]
        time_series_prev = time_series_prev.univariates[k]
        vals = torch.FloatTensor([time_series_prev.np_values])
        if torch.cuda.is_available():
            vals = vals.cuda()
            self.model.cuda()
        with torch.no_grad():
            yhat = self.model(vals, future=n).squeeze().tolist()

        if return_prev:
            t_prev = time_series_prev.time_stamps
            time_stamps = t_prev + time_stamps
            orig_t = None if orig_t is None else t_prev + orig_t
        else:
            yhat = yhat[-n:]

        yhat = UnivariateTimeSeries(time_stamps, yhat, self.target_name)
        return yhat.to_ts().align(reference=orig_t), None
