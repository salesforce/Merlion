import numpy as np
from merlion.utils.time_series import TimeSeries


class RollingWindowDataset:

    def __init__(self, data: TimeSeries, target_seq_index: int, maxlags: int, forecast_steps: int,):
        assert isinstance(data, TimeSeries), \
            "RollingWindowDataset expects to receive TimeSeries data"
        self.data = data.align()
        self.target_seq_index = target_seq_index
        self.maxlags = maxlags
        self.forecast_steps = forecast_steps

        self.data_uni = self.data.univariates[self.data.names[target_seq_index]]

        self.iter = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        """
        For example, if the input is like the following

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
        if self.iter <= self.valid_rolling_steps:
            past = self.data[self.iter : self.iter + self.maxlags]
            future = self.data_uni[self.iter + self.maxlags : self.iter + self.maxlags + self.forecast_steps]
            self.iter += 1
            return past, future
        else:
            raise StopIteration

    def __getitem__(self, idx):

        assert 0 <= idx <= self.valid_rolling_steps

        past = self.data[idx: idx + self.maxlags]
        future = self.data_uni[idx + self.maxlags: idx + self.maxlags + self.forecast_steps]
        return past, future

    def process_one_step_prior(self):
        """
        rolling window processor for the model to consume data, so it gives out
        data in a rolling window basis for forecasting, in the format of numpy array
        """
        data = self.data[-self.maxlags:]
        inputs = []
        for uni in data.univariates:
            inputs.append(uni.values)
        return np.concatenate(inputs, axis=0)

    def process_rolling_train_data(self):
        """
        default rolling window processor for the model to consume data, so it gives out
        train and label on a rolling window basis, in the format of numpy array
        return shape:
                inputs.shape = [n_samples, n_seq * maxlags]
                labels.shape = [n_samples, forecast_steps]
        """
        inputs = np.zeros((self.valid_rolling_steps + 1, self.maxlags * self.data.dim))
        for seq_ind, uni in enumerate(self.data.univariates):
            uni_data = uni.values
            for i in range(self.maxlags, len(self.data) - self.forecast_steps + 1):
                inputs[i - self.maxlags, seq_ind * self.maxlags: (seq_ind + 1) * self.maxlags] = \
                    uni_data[i - self.maxlags: i]

        labels = np.zeros((self.valid_rolling_steps + 1, self.forecast_steps))
        target_name = self.data.names[self.target_seq_index]
        target_data = self.data.univariates[target_name].values
        target_timestamp = self.data.univariates[target_name].index
        for i in range(self.maxlags, len(self.data) - self.forecast_steps + 1):
            labels[i - self.maxlags] = target_data[i: i + self.forecast_steps]

        labels_timestamp = target_timestamp[self.maxlags: len(self.data) - self.forecast_steps + 1]

        return inputs, labels, labels_timestamp

    def process_regressive_train_data(self):
        """
        regressive window processor for the auto-regression seq2seq model to consume data, so it gives out
        train and label on a rolling window basis auto-regressively, in the format of numpy array
        return shape:
                inputs.shape = [n_samples, n_seq * maxlags]
                labels.shape = [n_samples, n_seq]
        """

        inputs = np.zeros((len(self.data) - self.maxlags, self.maxlags * self.data.dim))
        labels = np.zeros((len(self.data) - self.maxlags, self.data.dim))

        for seq_ind, uni in enumerate(self.data.univariates):
            uni_data = uni.values
            for i in range(self.maxlags, len(self.data)):
                inputs[i - self.maxlags, seq_ind * self.maxlags: (seq_ind + 1) * self.maxlags] = \
                    uni_data[i - self.maxlags: i]
                labels[i - self.maxlags, seq_ind] = uni_data[i]

        target_timestamp = self.data.univariates[self.data.names[0]].index
        labels_timestamp = target_timestamp[self.maxlags: len(self.data)]

        return inputs, labels, labels_timestamp

    @property
    def valid_rolling_steps(self):
        return len(self.data) - self.maxlags - self.forecast_steps


def max_feasible_forecast_steps(data: TimeSeries, maxlags: int):
    return len(data) - maxlags