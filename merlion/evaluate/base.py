#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class for an automated model evaluation framework.
"""

from abc import abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from merlion.models.base import ModelBase
from merlion.models.forecast.base import ForecasterBase
from merlion.utils.misc import AutodocABCMeta
from merlion.utils.resample import granularity_str_to_seconds, to_offset
from merlion.utils.time_series import TimeSeries


class EvaluatorConfig(metaclass=AutodocABCMeta):
    """
    Abstract class which defines an evaluator config.
    """

    _timedelta_keys = ["train_window", "retrain_freq", "cadence"]

    def __init__(self, train_window: float = None, retrain_freq: float = None, cadence: float = None):
        """
        :param train_window: the maximum duration of data we would like to train the model on. ``None`` means no limit.
        :param retrain_freq: the frequency at which we want to re-train the model. ``None`` means we only train the
            model once on the initial training data.
        :param cadence: the frequency at which we want to obtain predictions from the model.
            ``None`` means that we obtain a new prediction at the same frequency as the model's predictive horizon.
            ``0`` means that we obtain a new prediction at every timestamp.
        """
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.cadence = cadence

    @property
    def train_window(self) -> Union[pd.Timedelta, pd.DateOffset, None]:
        """
        :return: the maximum duration of data we would like to train the model on. ``None`` means no limit.
        """
        return self._train_window

    @train_window.setter
    def train_window(self, train_window):
        self._train_window = to_offset(train_window)

    @property
    def retrain_freq(self) -> Union[pd.Timedelta, pd.DateOffset, None]:
        """
        :return: the frequency at which we want to re-train the model. ``None`` means we only train the model on the
            initial training data.
        """
        return self._retrain_freq

    @retrain_freq.setter
    def retrain_freq(self, retrain_freq):
        self._retrain_freq = to_offset(retrain_freq)

    @property
    def cadence(self) -> Union[pd.Timedelta, pd.DateOffset]:
        """
        :return: the cadence at which we are having our model produce new predictions. Defaults to the retraining
            frequency if not explicitly provided.
        """
        if self._cadence is None:
            return self.retrain_freq
        return self._cadence

    @cadence.setter
    def cadence(self, cadence):
        self._cadence = to_offset(cadence)

    @property
    def horizon(self) -> pd.DateOffset:
        """
        :return: the horizon our model is predicting into the future. Equal to the prediction cadence by default.
        """
        return self.cadence

    def to_dict(self):
        config_dict = {}
        for key, value in self.__dict__.items():
            k_strip = key.lstrip("_")
            if k_strip in self._timedelta_keys and value is not None:
                config_dict[k_strip] = value.microseconds / 1e6 if isinstance(value, pd.Timedelta) else value.freqstr
            else:
                config_dict[k_strip] = value
        return config_dict


class EvaluatorBase(metaclass=AutodocABCMeta):
    """
    An evaluator simulates the live deployment of a model on historical data.
    It trains a model on an initial time series, and then re-trains that model
    at a specified frequency.

    The `EvaluatorBase.get_predict` method returns the train & test predictions
    of a model, as if it were being trained incrementally on the test data in
    the manner described above.

    Subclasses define slightly different protocols for different tasks, e.g.
    anomaly detection vs. forecasting.
    """

    config_class = EvaluatorConfig

    def __init__(self, model: ModelBase, config: EvaluatorConfig):
        """
        :param model: the model to evaluate.
        :param config: the evaluation configuration.
        """
        assert isinstance(model, ModelBase)
        assert isinstance(config, self.config_class)
        self.model = model
        self.config = config

    @property
    def train_window(self):
        return self.config.train_window

    @property
    def retrain_freq(self):
        return self.config.retrain_freq

    @property
    def cadence(self):
        return self.config.cadence

    @property
    def horizon(self):
        return self.config.horizon

    @abstractmethod
    def _call_model(
        self, time_series: TimeSeries, time_series_prev: TimeSeries, exog_data: TimeSeries = None
    ) -> TimeSeries:
        raise NotImplementedError

    def _train_model(self, train_vals: TimeSeries, **train_kwargs) -> TimeSeries:
        return self.model.train(train_vals, **train_kwargs)

    def default_train_kwargs(self) -> dict:
        return {}

    def default_retrain_kwargs(self) -> dict:
        return {}

    @property
    def _concat_result(self):
        """
        In general, concatenate the result of ``get_predict()`` into a single
        `TimeSeries` if the prediction cadence is the same as the predictive
        horizon.
        """
        return self.cadence == self.horizon

    def get_predict(
        self,
        train_vals: TimeSeries,
        test_vals: TimeSeries,
        exog_data: TimeSeries = None,
        train_kwargs: dict = None,
        retrain_kwargs: dict = None,
    ) -> Tuple[Any, Union[TimeSeries, List[TimeSeries]]]:
        """
        Initialize the model by training it on an initial set of train data.
        Get the model's predictions on the test data, retraining the model as
        appropriate.

        :param train_vals: initial training data
        :param test_vals: all data where we want to get the model's predictions and compare it to the ground truth
        :param exog_data: any exogenous data (only used for some models)
        :param train_kwargs: dict of keyword arguments we want to use for the initial training process
        :param retrain_kwargs: dict of keyword arguments we want to use for all subsequent retrainings

        :return: ``(train_result, result)``. ``train_result`` is the output of training the model on ``train_vals``
            (``None`` if ``pretrained`` is ``True``). ``result`` is the model's predictions on ``test_vals``, and is
            specific to each evaluation task.
        """
        # Determine the appropriate training/retraining kwargs
        train_kwargs = {} if train_kwargs is None else train_kwargs
        full_train_kwargs = self.default_train_kwargs()
        full_train_kwargs.update(train_kwargs)
        retrain_kwargs = {} if retrain_kwargs is None else retrain_kwargs
        full_retrain_kwargs = self.default_retrain_kwargs()
        full_retrain_kwargs.update(retrain_kwargs)
        if isinstance(self.model, ForecasterBase):
            full_train_kwargs.update(exog_data=exog_data)
            full_retrain_kwargs.update(exog_data=exog_data)

        # Train the initial model (if not pretrained)
        self.model.reset()
        train_result = self._train_model(train_vals, **full_train_kwargs)
        if test_vals is None:
            return train_result, None

        # We will incrementally build up the final result window-by-window, where each window is a time series.
        # t_next is the next time we will re-train the model.
        all_t = test_vals.index
        t, tf = all_t[0], all_t[-1] + pd.Timedelta(milliseconds=1)
        t_next = tf if self.retrain_freq is None else t + self.retrain_freq

        result = []
        pbar = tqdm(total=int(granularity_str_to_seconds(tf - t)), desc=type(self).__name__)
        t_prev = t
        while t < tf:
            pbar.update(int(granularity_str_to_seconds(t - t_prev)))
            # Get the train & test data for the current window
            cur_train, cur_test = test_vals.bisect(t, t_in_left=False)
            cur_train = cur_train if train_vals is None else train_vals + cur_train
            if self.train_window is not None:
                cur_train = cur_train.window(t - self.train_window, t)
            if self.horizon is not None:
                i = np.searchsorted(all_t, t)
                tf_pred = cur_train.index[-1] + self.horizon
                if self.horizon is not None and i + 1 < len(all_t):
                    tf_pred = max(tf_pred, all_t[i + 1])
                cur_test = cur_test.window(t, tf_pred, include_tf=True)

            # Fully re-train the model when it is time to do so
            if t >= t_next and not cur_train.is_empty() and not cur_test.is_empty():
                self.model.reset()
                self._train_model(cur_train, **full_retrain_kwargs)
                i = np.searchsorted(all_t, t_next)
                if i + 1 < len(all_t):
                    t_next = max(t_next + self.retrain_freq, all_t[i + 1])
                else:
                    t_next = t_next + self.retrain_freq

            # Add this result if there is any result to add
            if not cur_test.is_empty():
                cur_result = self._call_model(time_series=cur_test, time_series_prev=cur_train, exog_data=exog_data)
                result.append(cur_result)

            # Move to the next eval window based on the cadence.
            i = np.searchsorted(all_t, t)
            t_prev = t
            if self.cadence is not None and i + 1 < len(all_t):
                t = max(t + self.cadence, all_t[i + 1])
            else:
                t = tf
        pbar.update(int(granularity_str_to_seconds(tf - t_prev)))

        # Concatenate everything together into a single time series if desired
        pbar.close()
        if self._concat_result:
            result = sum(result[1:], result[0])

        return train_result, result

    @abstractmethod
    def evaluate(self, ground_truth, predict, metric):
        """
        Given the ground truth time series & the model's prediction (as produced
        by `EvaluatorBase.get_predict`), compute the specified evaluation
        metric. If no metric is specified, return the appropriate score
        accumulator for the task. Implementation is task-specific.
        """
        raise NotImplementedError
