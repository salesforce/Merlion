import logging
import numpy as np
from typing import Optional, Union
import pandas as pd
from merlion.utils.time_series import TimeSeries

logger = logging.getLogger(__name__)


class RollingWindowDataset:

    def __init__(
            self,
            data: Union[TimeSeries, pd.DataFrame],
            target_seq_index: Optional[int],
            n_past: int,
            n_future: int,
            shuffle: bool = False,
            ts_index: bool = False,
            batch_size: Optional[int] = 1,
    ):
        """
        A rolling window dataset iterable to yield (past, future).

        :param data: time series data in the format of TimeSeries or DatetimeIndex indexed pandas DataFrame
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to use for the future labeling. If target_seq_index = None,
            it implies that all the sequences are required for the future labeling,
            in this case, future will be rolled along the sequence dimension.
        :param n_past: number of steps for past
        :param n_future: number of steps for future. If target_seq_index = None, n_future = 1 by enforcement.
        :param shuffle: True or False to randomly shuffle the returning window, please note that the
            time series data itself will never be shuffled
        :param ts_index: keep original TimeSeries internally for all the slicing, and output TimeSeries.
            by default, Numpy array will handle the internal data workflow and Numpy array will be the output.
        :param batch_size: if >= 1: will yield a list of (past, future) with batch_size
            if None: will give a one-shot dataset for all the rolling windows.
        """
        assert isinstance(data, (TimeSeries, pd.DataFrame)), \
            "RollingWindowDataset expects to receive TimeSeries or pd.DataFrame data "
        if isinstance(data, TimeSeries):
            data = data.align()
            self.dim = data.dim
        else:
            assert isinstance(data.index, pd.DatetimeIndex)
            assert ts_index is False, "Only TimeSeries data support ts_index = True "
            self.dim = data.shape[1]

        self.batch_size = batch_size
        self.ts_index = ts_index

        self.target_seq_index = target_seq_index
        if self.target_seq_index is None:
            logger.info(
                f"target_seq_index is None, therefore, the future data will be rolling along "
                f"the entire sequence dimension with timestamp increment by 1, i.e, self.n_future = 1."
                f"This is the rolling strategy for autoregression algorithm where all the future "
                f"sequences need to be provided as the training prior. "
                f"If you are not expecting this behavior, please properly set up the target_seq_index "
                f"as a valid integer. "
            )
            # label rolling along the sequence dimension
            self._label_axis = 1
            self.n_future = 1
        else:
            # label rolling along the time dimension
            self._label_axis = 0
            self.n_future = n_future

        self.n_past = n_past

        if ts_index:
            self.data_ts = data
            self.target_ts = data.univariates[data.names[target_seq_index]] if self._label_axis == 0 else None
        else:
            data_df = data.to_pd() if isinstance(data, TimeSeries) else data
            self.data = data_df.values
            self.timestamp = data_df.index
            self.target = data_df.iloc[:, target_seq_index].values if self._label_axis == 0 else None
            self.target_timestamp = data_df.index

        self.shuffle = shuffle
        self._data_len = len(data)

    def __len__(self):
        return self._data_len

    def __iter__(self):

        if self.batch_size is None or self.batch_size < 1:
            if self._label_axis == 0:
                yield self._get_entire_train_data_along_time()
            elif self._label_axis == 1:
                yield self._get_entire_train_data_along_sequence()
            return

        if self.batch_size == 1:
            yield from self._get_iterator()
        elif self.batch_size > 1:
            batch = list()
            for i in self._get_iterator():
                batch.append(i)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = list()

    def _get_iterator(self):
        """
        For example, if the input is like the following

            for self._label_axis = 0, we roll the window along the time axis
            .. code-block:: python

                data = (t0, (1,2,3)),
                       (t1, (4,5,6)),
                       (t2, (7,8,9)),
                       (t3, (10,11,12))
                target_seq_index = 0
                n_past = 2
                n_future = 2

        Then we may yield the following for the first time

            .. code-block:: python
                (
                    (t0, (1,2,3)),
                    (t1, (4,5,6))
                ),
                (
                    (t2, (7))
                    (t3, (10))
                )

        """
        if self._label_axis == 0:
            _valid_rolling_steps = len(self) - self.n_past - self.n_future
        elif self._label_axis == 1:
            _valid_rolling_steps = len(self) - self.n_past
        order = np.random.permutation(_valid_rolling_steps + 1) if self.shuffle \
            else range(_valid_rolling_steps + 1)
        for i in order:
            j = i + self.n_past
            if self._label_axis == 0:
                if self.ts_index:
                    past_ts = self.data_ts[i: j]
                    future_ts = self.target_ts[j: j + self.n_future]
                    yield past_ts, future_ts
                else:
                    past = self.data[i: j]
                    future = self.target[j: j + self.n_future]
                    future_timestamp = self.target_timestamp[j: j + self.n_future]
                    yield past, future, future_timestamp
            elif self._label_axis == 1:
                if self.ts_index:
                    past_ts = self.data_ts[i: j]
                    future_ts = self.data_ts[j]
                    yield past_ts, future_ts
                else:
                    past = self.data[i: j]
                    future = self.data[j]
                    future_timestamp = np.atleast_1d(self.target_timestamp[j])
                    yield past, future, future_timestamp

    def __getitem__(self, idx):

        if self._label_axis == 0:
            _valid_rolling_steps = len(self) - self.n_past - self.n_future
        elif self._label_axis == 1:
            _valid_rolling_steps = len(self) - self.n_past

        assert 0 <= idx <= _valid_rolling_steps
        idx_end = idx + self.n_past
        if self._label_axis == 0:
            if self.ts_index:
                past_ts = self.data_ts[idx: idx_end]
                future_ts = self.target_ts[idx_end: idx_end + self.n_future]
                return past_ts, future_ts
            else:
                past = self.data[idx: idx_end]
                future = self.target[idx_end: idx_end + self.n_future]
                future_timestamp = self.target_timestamp[idx_end: idx_end + self.n_future]
                return past, future, future_timestamp
        elif self._label_axis == 1:
            if self.ts_index:
                past_ts = self.data_ts[idx: idx_end]
                future_ts = self.data_ts[idx_end]
                return past_ts, future_ts
            else:
                past = self.data[idx: idx_end]
                future = self.data[idx_end]
                future_timestamp = np.atleast_1d(self.target_timestamp[idx_end])
                return past, future, future_timestamp

    def _get_entire_train_data_along_time(self):
        """
        default rolling window processor for the model to consume data as the (inputs, labels), so it gives out
        train and label on a rolling window basis, in the format of numpy array
        return shape:
                inputs.shape = [n_samples, n_seq * n_past]
                labels.shape = [n_samples, n_future]
        """
        _valid_rolling_steps = len(self) - self.n_past - self.n_future
        inputs = np.zeros((_valid_rolling_steps + 1, self.n_past * self.dim))
        for i in range(self.n_past, len(self.data) - self.n_future + 1):
            inputs[i - self.n_past] = self.data[i - self.n_past: i].reshape(-1, order="F")

        labels = np.zeros((_valid_rolling_steps + 1, self.n_future))
        for i in range(self.n_past, len(self.data) - self.n_future + 1):
            labels[i - self.n_past] = self.target[i: i + self.n_future]

        labels_timestamp = self.target_timestamp[self.n_past: len(self.data) - self.n_future + 1]

        return inputs, labels, labels_timestamp

    def _get_entire_train_data_along_sequence(self):
        """
        regressive window processor for the auto-regression model to consume data as the (inputs, labels),
        so it gives out train and label on a rolling window basis auto-regressively, in the format of numpy array
        return shape:
                inputs.shape = [n_samples, n_seq * n_past]
                labels.shape = [n_samples, n_seq]
        """

        inputs = np.zeros((len(self) - self.n_past, self.n_past * self.dim))
        labels = np.zeros((len(self) - self.n_past, self.dim))

        for i in range(self.n_past, len(self.data)):
            inputs[i - self.n_past] = self.data[i - self.n_past: i].reshape(-1, order="F")
            labels[i - self.n_past] = self.data[i]

        labels_timestamp = self.target_timestamp[self.n_past: len(self.data)]

        return inputs, labels, labels_timestamp


def max_feasible_forecast_steps(data: Union[TimeSeries, pd.DataFrame], maxlags: int):
    return len(data) - maxlags