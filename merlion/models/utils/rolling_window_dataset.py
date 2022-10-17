import numpy as np
from typing import Optional
from merlion.utils.time_series import TimeSeries


class RollingWindowDataset:

    def __init__(
            self,
            data: TimeSeries,
            target_seq_index: int,
            maxlags: int,
            forecast_steps: int,
            shuffle: bool = False,
            ts_index: bool = False,
            batch_size: Optional[int] = 1,
            label_axis: int = 0,
    ):
        assert isinstance(data, TimeSeries), \
            "RollingWindowDataset expects to receive TimeSeries data"
        data = data.align()
        self.dim = data.dim
        self.batch_size = batch_size
        self.ts_index = ts_index

        if ts_index:
            self.data_ts = data
            self.target_ts = data.univariates[data.names[target_seq_index]]
        else:
            data_df = data.to_pd()
            self.data = data_df.values
            self.timestamp = data_df.index
            self.target = data.univariates[data.names[target_seq_index]].values
            self.target_timestamp = data.univariates[data.names[target_seq_index]].index

        self.target_seq_index = target_seq_index
        self.maxlags = maxlags
        self.forecast_steps = forecast_steps

        self.shuffle = shuffle
        self.label_axis = label_axis

        self._valid_rolling_steps = len(data) - self.maxlags - self.forecast_steps
        self._data_len = len(data)

    @property
    def valid_rolling_steps(self):
        return self._valid_rolling_steps

    def __len__(self):
        return self._data_len

    def __iter__(self):

        if self.batch_size is None or self.batch_size < 1:
            if self.label_axis == 0:
                yield self._get_entire_train_data_along_time()
            elif self.label_axis == 1:
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

            for self.label_axis = 0, we roll the window along the time axis
            .. code-block:: python

                data = (t0, (1,2,3)),
                       (t1, (4,5,6)),
                       (t2, (7,8,9)),
                       (t3, (10,11,12))
                target_seq_index = 0
                maxlags = 2
                forecast_steps = 2

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
        order = np.random.permutation(self.valid_rolling_steps + 1) if self.shuffle \
            else range(self.valid_rolling_steps + 1)
        for i in order:
            j = i + self.maxlags
            if self.label_axis == 0:
                if self.ts_index:
                    past_ts = self.data_ts[i: j]
                    future_ts = self.target_ts[j: j + self.forecast_steps]
                    yield past_ts, future_ts
                else:
                    past = self.data[i: j]
                    future = self.target[j: j + self.forecast_steps]
                    future_timestamp = self.target_timestamp[j: j + self.forecast_steps]
                    yield past, future, future_timestamp
            elif self.label_axis == 1:
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

        assert 0 <= idx <= self.valid_rolling_steps
        idx_end = idx + self.maxlags
        if self.label_axis == 0:
            if self.ts_index:
                past_ts = self.data_ts[idx: idx_end]
                future_ts = self.target_ts[idx_end: idx_end + self.forecast_steps]
                return past_ts, future_ts
            else:
                past = self.data[idx: idx_end]
                future = self.target[idx_end: idx_end + self.forecast_steps]
                future_timestamp = self.target_timestamp[idx_end: idx_end + self.forecast_steps]
                return past, future, future_timestamp
        elif self.label_axis == 1:
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
                inputs.shape = [n_samples, n_seq * maxlags]
                labels.shape = [n_samples, forecast_steps]
        """
        inputs = np.zeros((self.valid_rolling_steps + 1, self.maxlags * self.dim))
        for i in range(self.maxlags, len(self.data) - self.forecast_steps + 1):
            inputs[i - self.maxlags] = self.data[i - self.maxlags: i].reshape(-1, order="F")

        labels = np.zeros((self.valid_rolling_steps + 1, self.forecast_steps))
        for i in range(self.maxlags, len(self.data) - self.forecast_steps + 1):
            labels[i - self.maxlags] = self.target[i: i + self.forecast_steps]

        labels_timestamp = self.target_timestamp[self.maxlags: len(self.data) - self.forecast_steps + 1]

        return inputs, labels, labels_timestamp

    def _get_entire_train_data_along_sequence(self):
        """
        regressive window processor for the auto-regression model to consume data as the (inputs, labels),
        so it gives out train and label on a rolling window basis auto-regressively, in the format of numpy array
        return shape:
                inputs.shape = [n_samples, n_seq * maxlags]
                labels.shape = [n_samples, n_seq]
        """

        inputs = np.zeros((len(self.data) - self.maxlags, self.maxlags * self.dim))
        labels = np.zeros((len(self.data) - self.maxlags, self.dim))

        for i in range(self.maxlags, len(self.data)):
            inputs[i - self.maxlags] = self.data[i - self.maxlags: i].reshape(-1, order="F")
            labels[i - self.maxlags] = self.data[i]

        labels_timestamp = self.target_timestamp[self.maxlags: len(self.data)]

        return inputs, labels, labels_timestamp


def max_feasible_forecast_steps(data: TimeSeries, maxlags: int):
    return len(data) - maxlags