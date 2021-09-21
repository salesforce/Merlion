#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from merlion.utils.time_series import TimeSeries
from merlion.transform.base import TransformBase


def process_one_step_prior(data: TimeSeries, maxlags: int, sampling_mode="normal"):
    """
    rolling window processor for the seq2seq model to consume data, so it gives out
    train in a rolling window basis

    :param data: multivariate timeseries data
    :param maxlags: max number of lags
    :param sampling_mode: if "normal" then concatenate all sequences over the window;
                          if "stats" then give statistics measures over the window
    :return: inputs
    """
    data = data.align()
    data = data[-maxlags:]
    if sampling_mode == "normal":
        inputs = []
        for uni in data.univariates:
            inputs.append(uni.values)
        return np.concatenate(inputs, axis=0)
    elif sampling_mode == "stats":
        inputs = []
        for uni in data.univariates:
            uni_values = uni.values
            inputs += [np.max(uni_values), np.min(uni_values), np.std(uni_values, ddof=1), np.median(uni_values)]
        return np.array(inputs)


def process_one_step_prior_for_autoregression(data: TimeSeries, maxlags: int, sampling_mode="normal"):
    """
    regressive window processor for the seq2seq model to consume data, so it
    gives out train in an autoregressive window basis

    :param data: multivariate timeseries data
    :param maxlags: max number of lags
    :param sampling_mode: if "normal" then concatenate all sequences over the window;
                          if "stats" then give statistics measures over the window
    :return: inputs, stats
    """
    data = data.align()
    data = data[-maxlags:]
    inputs = []
    for uni in data.univariates:
        inputs.append(uni.values)
    inputs = np.concatenate(inputs, axis=0)

    if sampling_mode == "normal":
        return inputs, None

    elif sampling_mode == "stats":
        stats = []
        for uni in data.univariates:
            uni_values = uni.values
            stats += [np.max(uni_values), np.min(uni_values), np.std(uni_values, ddof=1), np.median(uni_values)]
        stats = np.array(stats)
        return inputs, stats


def max_feasible_forecast_steps(data: TimeSeries, maxlags: int):
    return len(data) - maxlags


def process_rolling_train_data(
    data: TimeSeries, target_seq_index: int, maxlags: int, forecast_steps: int, sampling_mode="normal"
):
    """
    rolling window processor for the sequence2sequence seq2seq model to consume data, so it gives out
    train and label on a rolling window basis
    :param data: multivariate timeseries data
    :param target_seq_index: the target sequence index in TimeSeries data
    :param maxlags: Max # of lags
    :param sampling_mode: if "normal" then concatenate all sequences over the window;
                          if "stats" then give statistics measures over the window
    :return: inputs, labels, labels_timestamp
    """
    data = data.align()
    if sampling_mode == "normal":
        inputs = np.zeros((len(data) - maxlags - forecast_steps + 1, maxlags * data.dim))
        for seq_ind, uni in enumerate(data.univariates):
            uni_data = uni.values
            for i in range(maxlags, len(data) - forecast_steps + 1):
                inputs[i - maxlags, seq_ind * maxlags : (seq_ind + 1) * maxlags] = uni_data[i - maxlags : i]
    elif sampling_mode == "stats":
        data_df = data.to_pd()
        inputs = data_df.rolling(window=maxlags, center=False)
        inputs = inputs.aggregate(["max", "min", "std", "median"])
        inputs = inputs.iloc[maxlags - 1 : -forecast_steps].values
        assert inputs.shape[0] == len(data) - maxlags - forecast_steps + 1
    else:
        raise Exception("unknown sampling for the tree model ")

    labels = np.zeros((len(data) - maxlags - forecast_steps + 1, forecast_steps))
    target_name = data.names[target_seq_index]
    target_data = data.univariates[target_name].values
    target_timestamp = data.univariates[target_name].time_stamps
    for i in range(maxlags, len(data) - forecast_steps + 1):
        labels[i - maxlags] = target_data[i : i + forecast_steps]

    labels_timestamp = target_timestamp[maxlags : len(data) - forecast_steps + 1]

    return inputs, labels, labels_timestamp


def process_regressive_train_data(data: TimeSeries, maxlags: int, sampling_mode="normal"):
    """
    regressive window processor for the auto-regression seq2seq model to consume data, so it gives out
    train and label on a regressive basis
    :param data: multivariate timeseries data
    :param maxlags: Max # of lags
    :param sampling_mode: if "normal" then concatenate all sequences over the window;
                          if "stats" then give statistics measures over the window
    :return: inputs, labels, labels_timestamp, stats
    return shape:
            inputs.shape = [n_samples, n_seq * maxlags]
            labels.shape = [n_samples, n_seq]
            labels_timestamp.shape = [n_samples, 1]
            stats.shape = [n_samples, n_seq * 4], 4 are values from (max, min, std, median)
    """
    data = data.align()
    inputs = np.zeros((len(data) - maxlags, maxlags * data.dim))
    labels = np.zeros((len(data) - maxlags, data.dim))

    for seq_ind, uni in enumerate(data.univariates):
        uni_data = uni.values
        for i in range(maxlags, len(data)):
            inputs[i - maxlags, seq_ind * maxlags : (seq_ind + 1) * maxlags] = uni_data[i - maxlags : i]
            labels[i - maxlags, seq_ind] = uni_data[i]
    if sampling_mode == "normal":
        stats = None

    elif sampling_mode == "stats":
        data_df = data.to_pd()
        stats = data_df.rolling(window=maxlags, center=False)
        stats = stats.aggregate(["max", "min", "std", "median"])
        stats = stats.iloc[maxlags - 1 : -1].values
        assert stats.shape[0] == len(data) - maxlags

    else:
        raise Exception("unknown sampling for the tree model ")

    target_timestamp = data.univariates[data.names[0]].time_stamps
    labels_timestamp = target_timestamp[maxlags : len(data)]

    return inputs, labels, labels_timestamp, stats


def update_prior_nd(prior: np.ndarray, next_forecast: np.ndarray, num_seq: int, maxlags: int, sampling_mode="normal"):
    """
    regressively update the prior by concatenate prior with next_forecast on the sequence dimension,
    if sampling_mode="stats", also update the stats
    :param prior: the prior [n_samples, n_seq * maxlags]
    :param next_forecast: the next forecasting result [n_samples, n_seq]
    :param num_seq: number of univariate sequences
    :return: updated prior, updated stats (if sampling_mode="stats")
    """
    assert isinstance(prior, np.ndarray) and len(prior.shape) == 2
    assert isinstance(next_forecast, np.ndarray) and len(next_forecast.shape) == 2

    # unsqueeze the sequence dimension so prior and next_forecast can be concatenated along sequence dimension
    # for example,
    # prior = [[1,2,3,4,5,6,7,8,9], [10,20,30,40,50,60,70,80,90]], after the sequence dimension is expanded
    # prior = [[[1,2,3], [4,5,6], [7,8,9]],
    #          [[10,20,30],[40,50,60],[70,80,90]]
    #         ]
    # next_forcast = [[0.1,0.2,0.3],[0.4,0.5,0.6]], after the sequence dimension is expanded
    # next_forecast = [[[0.1],[0.2],[0.3]],
    #                  [[0.4],[0.5],[0.6]]
    #                 ]
    prior = prior.reshape(len(prior), num_seq, -1)
    next_forecast = np.expand_dims(next_forecast, axis=2)
    prior = np.concatenate([prior, next_forecast], axis=2)[:, :, -maxlags:]
    if sampling_mode != "stats":
        return prior.reshape(len(prior), -1), None
    else:
        stats = np.concatenate(
            [
                np.max(prior, axis=2, keepdims=True),
                np.min(prior, axis=2, keepdims=True),
                np.std(prior, axis=2, keepdims=True, ddof=1),
                np.median(prior, axis=2, keepdims=True),
            ],
            axis=2,
        )
        return prior.reshape(len(prior), -1), stats.reshape(len(stats), -1)


def update_prior_1d(prior: np.ndarray, next_forecast: np.ndarray, maxlags: int):
    """
    regressively update the prior by concatenate prior with next_forecast for the univariate,
    :param prior: the prior [n_samples, maxlags]
    :param next_forecast: the next forecasting result [n_samples, n_prediction_steps],
        if n_prediciton_steps ==1, maybe [n_samples,]
    :return: updated prior
    """
    assert isinstance(prior, np.ndarray) and len(prior.shape) == 2
    assert isinstance(next_forecast, np.ndarray)

    if len(next_forecast.shape) == 1:
        next_forecast = np.expand_dims(next_forecast, axis=1)
    prior = np.concatenate([prior, next_forecast], axis=1)[:, -maxlags:]
    return prior.reshape(len(prior), -1)


def gen_next_seq_label_pairs(data: TimeSeries, target_seq_index: int, maxlags: int, forecast_steps: int):
    """
    Customized helper function to yield
    ``(multivariates_sequences, multisteps_labels)`` tuple pair

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

    :param data: multivariate time series data in the format of TimeSeries
    :param target_seq_index: indicate which univariate is the target time
        series sequence
    :param maxlags: maximum number of lags to include
    :param forecast_steps: number of forecasting steps for the target_seq_index
        univariate
    """
    sz = len(data)
    data_uni = data.univariates[data.names[target_seq_index]]
    for i in range(maxlags, sz - forecast_steps + 1):
        yield data[i - maxlags : i], data_uni[i : i + forecast_steps]


def hybrid_forecast(model, inputs, steps, prediction_stride, maxlags):
    """
    n-step autoregression method for univairate data, each regression step updates n_prediction_steps data points
    :param model: model object to use when generating the forecast. model must have a ``predict`` method.
    :param inputs: regression inputs [n_samples, maxlags]
    :param steps: forecasting steps
    :param prediction_stride: the prediction step for training and forecasting
    :param maxlags
    :return: pred of target_seq_index for steps [n_samples, steps]
    """
    inputs = np.atleast_2d(inputs)

    pred = np.empty((len(inputs), (int((steps - 1) / prediction_stride) + 1) * prediction_stride))
    start = 0
    while True:
        next_forecast = model.predict(inputs)
        if len(next_forecast.shape) == 1:
            next_forecast = np.expand_dims(next_forecast, axis=1)
        pred[:, start : start + prediction_stride] = next_forecast
        start += prediction_stride
        if start >= steps:
            break
        inputs = update_prior_1d(inputs, next_forecast, maxlags)
    return pred[:, :steps]
