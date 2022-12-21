#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
A rolling window dataset
"""
import logging
import math
import numpy as np
from typing import Optional, Union
import pandas as pd
from merlion.utils.time_series import TimeSeries, to_pd_datetime
from merlion.models.utils.time_features import get_time_features

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
        ts_encoding: Union[None, str] = None,
        valid_fraction: float = 0.0,
        validation: bool = False,
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
            for training deep learning based time series models; if ``None``, the timestamp is not encoded.
            If not ``None``, it represents the frequency for time features encoding options:[s:secondly, t:minutely, h:hourly,
            d:daily, b:business days, w:weekly, m:monthly]
        :param valid_fraction: Fraction of validation set splitted from training data. if ``valid_fraction = 0``
            or ``valid_fraction = 1``, we iterate over the entire dataset
        :param validation: Whether the data is from the validation set or not.
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
            self.timestamp = get_time_features(self.timestamp, self.ts_encoding)

        self._seed = seed

        self._valid = validation
        self.valid_fraction = valid_fraction

        if valid_fraction <= 0.0 or valid_fraction >= 1.0:
            n_valid = n_train = self.n_windows
        else:
            n_valid = math.ceil(self.n_windows * self.valid_fraction)
            n_train = self.n_windows - n_valid

        data_indices = np.arange(self.n_windows)
        # use seed 0 to perturb the dataset
        if shuffle:
            data_indices = np.random.RandomState(seed).permutation(data_indices)

        self.train_indices = data_indices[:n_train]
        self.valid_indices = data_indices[-n_valid:]

    @property
    def validation(self):
        """
        If set ``False``, we only provide access to the training windows; if set ``True``,
        we only provide access to the validation windows.
        """
        return self._valid

    @validation.setter
    def validation(self, valid: bool):
        self._valid = valid

    @property
    def seed(self):
        """
        Set Random seed to perturb the training data
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """
        Set Random seed to perturb the training data
        """
        self._seed = seed

    @property
    def n_windows(self):
        """
        Number of total slides windows
        """
        return len(self.data) - self.n_past - self.n_future + 1

    @property
    def n_valid(self):
        """
        Number of slides windows in validation set
        """
        return len(self.valid_indices)

    @property
    def n_train(self):
        """
        Number of slides windows in training set
        """
        return len(self.train_indices)

    @property
    def n_points(self):
        return self.n_valid if self.validation else self.n_train

    def __len__(self):
        return int(np.ceil(self.n_points / self.batch_size)) if self.batch_size is not None else 1

    def __iter__(self):
        batch = []

        if self.validation:
            order = self.valid_indices
        elif self.shuffle and self.batch_size is not None:
            order = np.random.RandomState(self.seed).permutation(self.train_indices)
        else:
            order = self.train_indices

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
        if self.validation:
            assert idx in self.valid_indices
        else:
            assert idx in self.train_indices

        past_start = idx
        past_end = past_start + self.n_past
        future_start = past_end
        future_end = future_start + self.n_future

        past = self.data[past_start:past_end]
        past_timestamp = self.timestamp[past_start:past_end]
        future = self.target[future_start:future_end]
        future_timestamp = self.timestamp[future_start:future_end]

        return (past, future) if self.ts_index else (past, past_timestamp, future, future_timestamp)
