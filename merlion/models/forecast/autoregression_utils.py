#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import numpy as np
from merlion.utils.time_series import TimeSeries
from merlion.models.forecast import seq_ar_common

logger = logging.getLogger(__name__)


class MultiVariateAutoRegressionMixin:
    """
    Mixin class working together with ForecasterBase to provide the flexible number of forecasting steps
    for the multivariate forecasting models that have the fixed sequence-to-sequence training basis. Here,
    the algorithm is based on the autoregression algorithm similar to the inference algorithm utilized in VectorAR

    Prerequisite:
    self.max_forecast_steps, self.dim and self.target_seq_index, defined in ForecasterBase
    self.model defined in the derived forecasting model

    During the training,
    (1) the original TimeSeries data autoregressively produces data such that
    sequence input/target pair will be in the shape of
            inputs.shape = [n_samples, n_seq * maxlags]
            labels.shape = [n_samples, n_seq]
    where n_seq = data.dim
    (2) Mixin class will call self.model to fit. The model shall be a sequence to sequence basis in general,
    like tree-ensemble models, deep learning autoencoders, e.g.

    During the forecasting,
    (1) Mixin class will call self.model to infer one horizon step for all sequences, and append the predicted value of
    the target_seq_index to the final `pred`
    (2) Mixin class will concatenate all the inference for all sequences from step (1) to the `prior`
    (3) Mixin class will maintain the last maxlags steps in `prior` and go back to step (1)


    """

    def autoregression_train(self, data: TimeSeries, maxlags: int, sampling_mode: str = "normal"):
        """
        :param data: input data
        :param maxlags: Max # of lags for forecasting
        :param sampling_mode: how to process time series data for the tree model
            If "normal" then concatenate all sequences over the window
            If "stats" then give statistics measures over the window (usually only for the tree-ensemble)
        :return: serve a typical model.train() return as UnivariateTimeSeries(labels_train_ts, pred, self.target_name).to_ts()
                 prior_forecast: the forecast result from the last training series, and usually it is used to define
                                 self._forecast = prior_forecast
                 labels_train_ts: timestamp for the training data
                 pred: forecasting from the training data
        """
        assert (
            hasattr(self, "model") and self.model is not None
        ), "MultiVariateAutoRegressionMixin class is not in charge of defining the model"
        assert (
            hasattr(self, "dim") and self.dim is not None
        ), "MultiVariateAutoRegressionMixin requires the definition of dimension self.dim"
        assert self.dim == data.dim
        assert self.dim > 1

        (inputs_train, labels_train, labels_train_ts, stats_train) = seq_ar_common.process_regressive_train_data(
            data, maxlags, sampling_mode
        )
        if sampling_mode == "stats":
            self.model.fit(stats_train, labels_train)
            prior_stats = stats_train[-1]
        else:
            self.model.fit(inputs_train, labels_train)
            prior_stats = None
        prior = np.atleast_2d(inputs_train[-1])
        prior_forecast = self._autoregressive_forecast(
            prior, prior_stats, maxlags=maxlags, steps=None, sampling_mode=sampling_mode
        ).reshape(-1)
        inputs_train = np.atleast_2d(inputs_train)
        if sampling_mode == "stats":
            stats_train = np.atleast_2d(stats_train)
        pred = self._autoregressive_forecast(
            inputs_train, stats_train, maxlags=maxlags, steps=None, sampling_mode=sampling_mode
        )
        pred = pred[:, 0].reshape(-1)
        return prior_forecast, labels_train_ts, pred

    def autoregression_forecast(
        self, time_series_prev: TimeSeries, maxlags: int, forecast_steps: int, sampling_mode: str = "normal"
    ):
        """
        :param data: input data
        :param maxlags: Max # of lags for forecasting
        :param forecast_steps:
        :param sampling_mode: how to process time series data for the tree model
            If "normal" then concatenate all sequences over the window
            If "stats" then give statistics measures over the window (usually only for the tree-ensemble)
        :return: yhat: values of forecasting given time_series_prev, and yhat serves as
                 a typical ``model.forecast()`` return as
                 ``UnivariateTimeSeries(name=self.target_name, time_stamps=t, values=yhat)``
        """

        assert (
            hasattr(self, "model") and self.model is not None
        ), "MultiVariateAutoRegressionMixin class is not in charge of defining the model"
        assert (
            hasattr(self, "dim") and self.dim is not None
        ), "MultiVariateAutoRegressionMixin requires the definition of dimension self.dim"
        assert self.dim == time_series_prev.dim
        assert self.dim > 1

        (time_series_prev_no_ts, stats_prev_no_ts) = seq_ar_common.process_one_step_prior_for_autoregression(
            time_series_prev, maxlags, sampling_mode
        )
        yhat = self._autoregressive_forecast(
            time_series_prev_no_ts, stats_prev_no_ts, maxlags=maxlags, steps=forecast_steps, sampling_mode=sampling_mode
        ).reshape(-1)
        return yhat

    def _autoregressive_forecast(self, inputs, stats, maxlags: int, steps: [int, None], sampling_mode: str = "normal"):
        """
         1-step auto-regression method for multivariate data, each regression step updates one data point for each sequence

        :param inputs: regression inputs [n_samples, self.dim * maxlags]
        :param stats: regression stats derived from inputs [n_samples, self.dim * 4]
        :param steps: forecasting steps
        :return: pred of target_seq_index for steps [n_samples, steps]
        """

        if steps is None:
            steps = self.max_forecast_steps

        inputs = np.atleast_2d(inputs)
        if sampling_mode == "stats":
            assert stats is not None
            stats = np.atleast_2d(stats)

        pred = np.empty((len(inputs), steps))

        for i in range(steps):
            # next forecast shape: [n_samples, self.dim]
            if sampling_mode == "stats":
                next_forecast = self.model.predict(stats)
            else:
                next_forecast = self.model.predict(inputs)
            pred[:, i] = next_forecast[:, self.target_seq_index]
            if i == steps - 1:
                break
            inputs, stats = seq_ar_common.update_prior_nd(inputs, next_forecast, self.dim, maxlags, sampling_mode)
        return pred
