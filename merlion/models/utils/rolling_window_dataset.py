#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import numpy as np
from typing import Optional, Union
import pandas as pd
from merlion.utils.time_series import TimeSeries, to_pd_datetime
from merlion.utils.timefeatures import time_features

logger = logging.getLogger(__name__)


class RollingWindowDataset:
    def __init__(
        self,
        data: Union[TimeSeries, pd.DataFrame],
        target_seq_index: Optional[int],
        n_past: int,
        n_future: int,
        exog_data: Union[TimeSeries, pd.DataFrame] = None,
        shuffle: bool = False,
        ts_index: bool = False,
        batch_size: Optional[int] = 1,
        flatten: bool = True,
        ts_encoding: bool = False,
        ts_freq: str = "h",
        start_token_len: int = 0,
        seed: int = 0,
    ):
        """
        A rolling window dataset which returns ``(past, future)`` windows for the whole time series.
        If ``ts_index=True`` is used, a batch size of 1 is employed, and each window returned by the dataset is
        ``(past, future)``, where ``past`` and ``future`` are both `TimeSeries` objects.
        If ``ts_index=False`` is used (default option, more efficient), each window returned by the dataset is
        ``(past_np, past_time, future_np, future_time)``:

        - ``past_np`` is a numpy array with shape ``(batch_size, n_past * dim)`` if ``flatten`` is ``True``, otherwise
          ``(batch_size, n_past, dim)``.
        -  ``past_time`` is a numpy array of times with shape ``(batch_size, n_past)``
        - ``future_np`` is a numpy array with shape ``(batch_size, dim)`` if ``target_seq_index`` is ``None``
          (autoregressive prediction), or shape ``(batch_size, n_future)`` if ``target_seq_index`` is specified.
        -  ``future_time`` is a numpy array of times with shape ``(batch_size, n_future)``

        :param data: time series data in the format of TimeSeries or pandas DataFrame with DatetimeIndex
        :param target_seq_index: The index of the univariate (amongst all univariates in a general multivariate time
            series) whose value we would like to use for the future labeling. If ``target_seq_index = None``, it implies
            that all the sequences are required for the future labeling. In this case, we set ``n_future = 1`` and
            use the time series for 1-step autoregressive prediction.
        :param n_past: number of steps for past
        :param n_future: number of steps for future. If ``target_seq_index = None``, we manually set ``n_future = 1``.
        :param exog_data: exogenous data to as inputs for the model, but not as outputs to predict.
            We assume the future values of exogenous variables are known a priori at test time.
        :param shuffle: whether the windows of the time series should be shuffled.
        :param ts_index: keep original TimeSeries internally for all the slicing, and output TimeSeries.
            by default, Numpy array will handle the internal data workflow and Numpy array will be the output.
        :param batch_size: the number of windows to return in parallel. If ``None``, return the whole dataset.
        :param flatten: whether the output time series arrays should be flattened to 2 dimensions.
        :param ts_encoding: whether the timestamp should be encoded to a float vector, which can be used
            for training transformer based deep time series models
        :param ts_freq: Frequency for time features encoding options:[s:secondly, t:minutely, h:hourly,
            d:daily, b:business days, w:weekly, m:monthly]
        :param start_token_len: Length of start token for deep transformer encoder-decoder based models; start token,
            which you can view as the special token for nlp models (e.g., bos, sep, eos tokens). For non-transformer based models,
            we set the token_length = 0
        """
        assert isinstance(
            data, (TimeSeries, pd.DataFrame)
        ), "RollingWindowDataset expects to receive TimeSeries or pd.DataFrame data "
        if isinstance(data, TimeSeries):
            data = data.align()
            self.dim = data.dim
            if exog_data is not None:
                assert isinstance(exog_data, TimeSeries), "Expected exog_data to be TimeSeries if data is TimeSeries"
                exog_data = exog_data.align(reference=data.time_stamps)
        else:
            assert isinstance(data.index, pd.DatetimeIndex)
            if exog_data is not None:
                if isinstance(exog_data, TimeSeries):
                    exog_data = exog_data.align(reference=data.index).to_pd()
                assert isinstance(exog_data.index, pd.DatetimeIndex) and list(exog_data.index) == list(data.index)
            assert ts_index is False, "Only TimeSeries data support ts_index = True "
            self.dim = data.shape[1]

        if ts_index and batch_size != 1:
            logger.warning("Setting batch_size = 1 because ts_index=True.")
            batch_size = 1
        self.batch_size = batch_size
        self.n_past = n_past
        self.shuffle = shuffle
        self.flatten = flatten
        self.ts_encoding = ts_encoding
        self.ts_freq = ts_freq
        self.start_token_len = start_token_len

        self.target_seq_index = target_seq_index
        self.n_future = n_future

        self.ts_index = ts_index
        if ts_index:
            self.data = data.concat(exog_data, axis=1) if exog_data is not None else data
            self.target = (
                data if self.target_seq_index is None else data.univariates[data.names[target_seq_index]].to_ts()
            )
            self.timestamp = to_pd_datetime(data.np_time_stamps)
        else:
            df = data.to_pd() if isinstance(data, TimeSeries) else data
            exog_df = data.to_pd() if isinstance(exog_data, TimeSeries) else exog_data
            if exog_data is not None:
                if n_future > 0:
                    exog_vals = np.concatenate((exog_df.values[1:], np.full((1, exog_df.shape[1]), np.nan)))
                else:
                    exog_vals = exog_df.values
                self.data = np.concatenate((df.values, exog_vals), axis=1)
            self.data = np.concatenate((df.values, exog_df.values), axis=1) if exog_data is not None else df.values
            self.timestamp = df.index
            self.target = df.values if self.target_seq_index is None else df.values[:, target_seq_index]

        if self.ts_encoding:
            self._timestamp_encoding()

        self.seed = seed

    def _timestamp_encoding(self):
        self.timestamp = time_features(self.timestamp, freq=self.ts_freq)
        self.timestamp = self.timestamp.transpose(1, 0)

    @property
    def autoregressive(self):
        return (self.target_seq_index is None) and (self.n_future == 1)

    @property
    def n_points(self):
        return len(self.data) - self.n_past + 1 - self.n_future

    def __len__(self):
        return int(np.ceil(self.n_points / self.batch_size)) if self.batch_size is not None else 1

    def __iter__(self):
        batch = []
        if self.shuffle and self.batch_size is not None:
            order = np.random.RandomState(self.seed).permutation(self.n_points)
        else:
            order = range(self.n_points)
        for i in order:
            batch.append(self[i])
            if self.batch_size is not None and len(batch) >= self.batch_size:
                yield self.collate_batch(batch)
                batch = []
        if len(batch) > 0:
            yield self.collate_batch(batch)

    def collate_batch(self, batch):
        if self.ts_index:
            return batch[0]
        # TODO: allow output shape to be specified as class parameter
        past, past_ts, future, future_ts = zip(*batch)
        past = np.stack(past)
        past_ts = np.stack(past_ts)

        if self.flatten:
            past = past.reshape((len(batch), -1))
            past_ts = past_ts.reshape((len(batch), -1)) if self.ts_encoding else past_ts

        if future is not None:
            future = np.stack(future)
            future = future.reshape((len(batch), -1)) if self.flatten else future

            future_ts = np.stack(future_ts)
            if self.flatten and self.ts_encoding:
                future_ts = future_ts.reshape((len(batch), -1))
        else:
            future, future_ts = None, None
        return past, past_ts, future, future_ts

    def __getitem__(self, idx):
        assert 0 <= idx < self.n_points
        past_start = idx
        past_end = past_start + self.n_past
        future_start = past_end - self.start_token_len
        future_end = future_start + self.start_token_len + self.n_future

        past = self.data[past_start:past_end]
        past_timestamp = self.timestamp[past_start:past_end]
        future = self.target[future_start:future_end]
        future_timestamp = self.timestamp[future_start:future_end]

        return (past, future) if self.ts_index else (past, past_timestamp, future, future_timestamp)
