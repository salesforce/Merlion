#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Bagging Tree-based models for multivariate time series forecasting.
Random Forest
ExtraTreesRegressor
"""
import logging
from typing import List, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.utils.time_series import assert_equal_timedeltas, TimeSeries, UnivariateTimeSeries
from merlion.models.forecast.autoregression_utils import MultiVariateAutoRegressionMixin
from merlion.models.forecast import seq_ar_common

logger = logging.getLogger(__name__)


class BaggingTreeForecasterConfig(ForecasterConfig):
    """
    Configuration class for bagging Tree-based forecaster model.
    """

    def __init__(
        self,
        max_forecast_steps: int,
        maxlags: int,
        target_seq_index: int = None,
        sampling_mode: str = "normal",
        prediction_stride: int = 1,
        n_estimators: int = 100,
        random_state=None,
        max_depth=None,
        min_samples_split=2,
        **kwargs,
    ):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param maxlags: Max # of lags for forecasting
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param sampling_mode: how to process time series data for the tree model. If "normal",
            then concatenate all sequences over the window. If "stats", then give statistics
            measures over the window. Note: "stats" mode is statistical summary for a multivariate dataset,
            mainly to reduce the computation cost for high-dimensional time series. For univariate data, it is
            not necessary to use "stats" instead of the sequence itself as the input. Therefore, for univariate,
            the model will automatically adopt "normal" mode.
        :param prediction_stride: the prediction step for training and forecasting

            - If univariate: the sequence target of the length of prediction_stride will be utilized, forecasting will be
              done by means of autoregression with the stride unit of prediction_stride
            - If multivariate:

                - if = 1: the autoregression with the stride unit of 1
                - if > 1: only support sequence mode, and the model will set prediction_stride = max_forecast_steps
        :param n_estimators: number of base estimators for the tree ensemble
        :param random_state: random seed for bagging
        :param max_depth: max depth of base estimators
        :param min_samples_split: min split for tree leaves
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.maxlags = maxlags
        self.sampling_mode = sampling_mode
        self.prediction_stride = prediction_stride
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state


class BaggingTreeForecaster(ForecasterBase, MultiVariateAutoRegressionMixin):
    """
    Tree model for multivariate time series forecasting.
    """

    config_class = BaggingTreeForecasterConfig

    model = None

    def __init__(self, config: BaggingTreeForecasterConfig):
        super().__init__(config)
        self._forecast = np.zeros(self.max_forecast_steps)
        self._input_already_transformed = False

    @property
    def maxlags(self) -> int:
        return self.config.maxlags

    @property
    def sampling_mode(self) -> str:
        return self.config.sampling_mode

    @property
    def prediction_stride(self) -> int:
        return self.config.prediction_stride

    def train(self, train_data: TimeSeries, train_config=None):
        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)

        # univariate case, hybrid of sequence + autoregression
        if self.dim == 1:
            logger.info(
                f"Model is working on a univariate dataset, "
                f"hybrid of sequence and autoregression training strategy will be adopted"
                f"with prediction_stride = {self.prediction_stride} "
            )
            if self.sampling_mode != "normal":
                logger.warning('For univariate dataset, only supports "normal" sampling mode')
                self.config.sampling_mode = "normal"
            # process train data
            (inputs_train, labels_train, labels_train_ts) = seq_ar_common.process_rolling_train_data(
                train_data, self.target_seq_index, self.maxlags, self.prediction_stride, self.sampling_mode
            )
            self.model.fit(inputs_train, labels_train)
            prior = np.atleast_2d(inputs_train[-1])
            forecast_result = self._hybrid_forecast(prior)
            self._forecast = forecast_result.reshape(-1)
            inputs_train = np.atleast_2d(inputs_train)
            pred = self._hybrid_forecast(inputs_train)
            # since the model may predict multiple steps, we concatenate all the first steps together
            pred = pred[:, 0].reshape(-1)

        # multivariate case, sequence or 1-step autoregression
        else:
            # sequence mode, set prediction_stride = max_forecast_steps
            if self.prediction_stride > 1:
                max_forecast_steps = seq_ar_common.max_feasible_forecast_steps(train_data, self.maxlags)
                if self.max_forecast_steps > max_forecast_steps:
                    logger.warning(
                        f"With train data of length {len(train_data)} and "
                        f"maxlags={self.maxlags}, the maximum supported forecast "
                        f"steps is {max_forecast_steps}, but got "
                        f"max_forecast_steps={self.max_forecast_steps}. Reducing "
                        f"to the maximum permissible value or switch to "
                        f"'training_mode = autogression'."
                    )
                    self.config.max_forecast_steps = max_forecast_steps
                if self.prediction_stride != self.max_forecast_steps:
                    logger.warning(
                        f"For multivariate dataset, reset prediction_stride = max_forecast_steps = {self.max_forecast_steps} "
                    )
                    self.config.prediction_stride = self.max_forecast_steps
                # process train data
                (inputs_train, labels_train, labels_train_ts) = seq_ar_common.process_rolling_train_data(
                    train_data, self.target_seq_index, self.maxlags, self.prediction_stride, self.sampling_mode
                )
                self.model.fit(inputs_train, labels_train)
                prior = np.atleast_2d(inputs_train[-1])
                forecast_result = self.model.predict(prior)
                self._forecast = forecast_result.reshape(-1)
                # since the model may predict multiple steps, we concatenate all the first steps together
                pred = self.model.predict(np.atleast_2d(inputs_train))[:, 0].reshape(-1)
            else:
                # autoregression mode
                prior_forecast, labels_train_ts, pred = self.autoregression_train(
                    data=train_data, maxlags=self.maxlags, sampling_mode=self.sampling_mode
                )
                self._forecast = prior_forecast

        return (UnivariateTimeSeries(labels_train_ts, pred, self.target_name).to_ts(), None)

    def forecast(
        self, time_stamps: List[int], time_series_prev: TimeSeries = None, return_iqr=False, return_prev=False
    ):

        t = self.resample_time_stamps(time_stamps, time_series_prev)

        if time_series_prev is not None:
            if not self._input_already_transformed:
                time_series_prev = self.transform(time_series_prev)
            # make sure training data agree with prediction data in the shape
            assert time_series_prev.dim == self.dim, (
                f"time_series_prev has multivariate dimension of "
                f"{time_series_prev.dim} that is different from "
                f"training data dimension of {self.dim} for the model"
            )
            assert len(time_series_prev) >= self.maxlags, (
                f"time_series_prev has a data length of "
                f"{len(time_series_prev)} that is shorter than the maxlags "
                f"for the model"
            )
            assert not return_prev, f"{type(self).__name__}.forecast() does not support return_prev=True"
            assert not return_iqr, f"{type(self).__name__} does not support return_iqr=True"

        if time_series_prev is None:
            yhat = self._forecast[: len(t)]
        else:
            if self.dim == 1:
                time_series_prev_no_ts = seq_ar_common.process_one_step_prior(
                    time_series_prev, self.maxlags, self.sampling_mode
                )
                yhat = self._hybrid_forecast(np.atleast_2d(time_series_prev_no_ts), len(t)).reshape(-1)
            else:
                if self.prediction_stride > 1:
                    time_series_prev_no_ts = seq_ar_common.process_one_step_prior(
                        time_series_prev, self.maxlags, self.sampling_mode
                    )
                    yhat = self.model.predict(np.atleast_2d(time_series_prev_no_ts)).reshape(-1)
                    yhat = yhat[: len(t)]
                else:
                    yhat = self.autoregression_forecast(
                        time_series_prev, maxlags=self.maxlags, forecast_steps=len(t), sampling_mode=self.sampling_mode
                    )

        forecast = (
            UnivariateTimeSeries(name=self.target_name, time_stamps=t, values=yhat).to_ts().align(reference=time_stamps)
        )
        return forecast, None

    def _hybrid_forecast(self, inputs, steps=None):
        """
        n-step autoregression method for univairate data, each regression step updates n_prediction_steps data points
        :param inputs: regression inputs [n_samples, maxlags]
        :return: pred of target_seq_index for steps [n_samples, steps]
        """

        if steps is None:
            steps = self.max_forecast_steps

        pred = seq_ar_common.hybrid_forecast(
            model=self.model, inputs=inputs, steps=steps, prediction_stride=self.prediction_stride, maxlags=self.maxlags
        )
        return pred

    def set_data_already_transformed(self):
        self._input_already_transformed = True

    def reset_data_already_transformed(self):
        self._input_already_transformed = False


class RandomForestForecasterConfig(BaggingTreeForecasterConfig):
    """
    Config class for `RandomForestForecaster`.
    """

    pass


class RandomForestForecaster(BaggingTreeForecaster):
    """
    Random Forest Regressor for time series forecasting

    Random Forest is a meta estimator that fits a number of classifying decision
    trees on various sub-samples of the dataset, and uses averaging to improve
    the predictive accuracy and control over-fitting.
    """

    config_class = RandomForestForecasterConfig

    def __init__(self, config: RandomForestForecasterConfig):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=self.config.random_state,
        )


class ExtraTreesForecasterConfig(BaggingTreeForecasterConfig):
    """
    Config cass for `ExtraTreesForecaster`.
    """

    pass


class ExtraTreesForecaster(BaggingTreeForecaster):
    """
    Extra Trees Regressor for time series forecasting

    Extra Trees Regressor implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples of
    the dataset and uses averaging to improve the predictive accuracy and
    control over-fitting.
    """

    config_class = ExtraTreesForecasterConfig

    def __init__(self, config: ExtraTreesForecasterConfig):
        super().__init__(config)
        self.model = ExtraTreesRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=self.config.random_state,
        )
