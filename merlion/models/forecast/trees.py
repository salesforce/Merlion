#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tree-based models for multivariate time series forecasting.
"""
import logging
from typing import List, Tuple

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

from merlion.models.forecast.autoregressivebase import AutoRegressiveForecaster, AutoRegressiveForecasterConfig
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset, max_feasible_forecast_steps
from merlion.utils.time_series import to_pd_datetime, TimeSeries

logger = logging.getLogger(__name__)


class TreeEnsembleForecasterConfig(AutoRegressiveForecasterConfig):
    """
    Configuration class for bagging tree-based forecaster model.
    """

    def __init__(
        self,
        maxlags: int,
        max_forecast_steps: int = None,
        target_seq_index: int = None,
        prediction_stride: int = 1,
        n_estimators: int = 100,
        random_state=None,
        max_depth=None,
        **kwargs,
    ):
        """
        :param maxlags: Max # of lags for forecasting
        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param prediction_stride: the number of steps being forecasted in a single call to underlying the model

            - If univariate: the sequence target of the length of prediction_stride will be utilized, forecasting will
              be done autoregressively, with the stride unit of prediction_stride
            - If multivariate:

                - if = 1: autoregressively forecast all variables in the time series, one step at a time
                - if > 1: only support directly forecasting the next prediction_stride steps in the future.
                Autoregression not supported. Note that the model will set prediction_stride = max_forecast_steps
        :param n_estimators: number of base estimators for the tree ensemble
        :param random_state: random seed for bagging
        :param max_depth: max depth of base estimators
        """
        super().__init__(
            maxlags=maxlags,
            max_forecast_steps=max_forecast_steps,
            target_seq_index=target_seq_index,
            prediction_stride=prediction_stride,
            **kwargs)
        self.maxlags = maxlags
        self.prediction_stride = prediction_stride
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth


class TreeEnsembleForecaster(AutoRegressiveForecaster):
    """
    Tree model for multivariate time series forecasting.
    """

    config_class = TreeEnsembleForecasterConfig
    model = None

    def __init__(self, config: TreeEnsembleForecasterConfig):
        super().__init__(config)

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def require_univariate(self) -> bool:
        return False

    @property
    def _default_train_config(self):
        return dict()

    def _train(self, train_data: pd.DataFrame, train_config=None):
        # if univariate, or prediction_stride is 1 for multivariate, call autoregressive model
        if self.dim == 1 or self.prediction_stride == 1:
            return super()._train(train_data, train_config)

        logger.info(
            f"Model is working on a multivariate dataset with prediction_stride > 1, "
            f"default multi-output regressor training strategy will be adopted "
            f"with prediction_stride = {self.prediction_stride} "
        )

        fit = train_config.get("fit", True)

        if isinstance(train_data, TimeSeries):
            assert self.dim == train_data.dim
        else:
            assert self.dim == train_data.shape[1]

        # multivariate case, fixed prediction horizon using the default multioutput tree regressor
        max_forecast_steps = max_feasible_forecast_steps(train_data, self.maxlags)
        if self.max_forecast_steps is not None and self.max_forecast_steps > max_forecast_steps:
            logger.warning(
                f"With train data of length {len(train_data)} and "
                f"maxlags={self.maxlags}, the maximum supported forecast "
                f"steps is {max_forecast_steps}, but got "
                f"max_forecast_steps={self.max_forecast_steps}. Reducing "
                f"to the maximum permissible value or switch to "
                f"'training_mode = autogression'."
            )
            self.config.max_forecast_steps = max_forecast_steps
        if self.max_forecast_steps is not None and self.prediction_stride != self.max_forecast_steps:
            logger.warning(
                f"For multivariate dataset, reset prediction_stride = max_forecast_steps = {self.max_forecast_steps} "
            )
            self.config.prediction_stride = self.max_forecast_steps

        # process train data to the rolling window dataset
        dataset = RollingWindowDataset(data=train_data,
                                       target_seq_index=self.target_seq_index,
                                       n_past=self.maxlags,
                                       n_future=self.prediction_stride,
                                       batch_size=None,
                                       )
        inputs_train, labels_train, labels_train_ts = next(iter(dataset))

        if fit:
            self.model.fit(inputs_train, labels_train)
        # since the model may predict multiple steps, we concatenate all the first steps together
        pred = self.model.predict(np.atleast_2d(inputs_train))[:, 0].reshape(-1)

        return pd.DataFrame(pred, index=labels_train_ts, columns=[self.target_name]), None

    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, None]:

        # if univariate, or prediction_stride is 1 for multivariate, call autoregressive model
        if self.dim == 1 or self.prediction_stride == 1:
            return super()._forecast(time_stamps, time_series_prev, return_prev)

        if time_series_prev is not None:
            assert len(time_series_prev) >= self.maxlags, (
                f"time_series_prev has a data length of "
                f"{len(time_series_prev)} that is shorter than the maxlags "
                f"for the model"
            )

        n = len(time_stamps)
        prev_pred, prev_err = None, None
        if time_series_prev is None:
            time_series_prev = self.transform(self.train_data)
        elif time_series_prev is not None and return_prev:
            prev_pred, prev_err = self._train(time_series_prev, train_config=dict(fit=False))

        time_series_prev_no_ts = self._get_immedidate_forecasting_prior(time_series_prev)

        yhat = self.model.predict(np.atleast_2d(time_series_prev_no_ts)).reshape(-1)
        yhat = yhat[:n]

        forecast = pd.DataFrame(yhat, index=to_pd_datetime(time_stamps), columns=[self.target_name])
        if prev_pred is not None:
            forecast = pd.concat((prev_pred, forecast))
        return forecast, None


class RandomForestForecasterConfig(TreeEnsembleForecasterConfig):
    """
    Config class for `RandomForestForecaster`.
    """

    def __init__(self, max_forecast_steps: int, maxlags: int, min_samples_split=2, **kwargs):
        """
        :param min_samples_split: min split for tree leaves
        """
        super().__init__(max_forecast_steps=max_forecast_steps, maxlags=maxlags, **kwargs)
        self.min_samples_split = min_samples_split


class RandomForestForecaster(TreeEnsembleForecaster):
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


class ExtraTreesForecasterConfig(TreeEnsembleForecasterConfig):
    """
    Config class for `ExtraTreesForecaster`.
    """

    def __init__(self, maxlags: int, min_samples_split=2, **kwargs):
        """
        :param min_samples_split: min split for tree leaves
        """
        super().__init__(maxlags=maxlags, **kwargs)
        self.min_samples_split = min_samples_split


class ExtraTreesForecaster(TreeEnsembleForecaster):
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


class LGBMForecasterConfig(TreeEnsembleForecasterConfig):
    """
    Config class for `LGBMForecaster`.
    """

    def __init__(self, maxlags: int, learning_rate=0.1, n_jobs=-1, **kwargs):
        """
        :param learning_rate: learning rate for boosting
        :param n_jobs: num of threading, -1 or 0 indicates device default, positive int indicates num of threads
        """
        super().__init__(maxlags=maxlags, **kwargs)
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs


class LGBMForecaster(TreeEnsembleForecaster):
    """
    Light gradient boosting (LGBM) regressor for time series forecasting

    LightGBM is a light weight and fast gradient boosting framework that uses tree based learning algorithms, for more
    details, please refer to the document https://lightgbm.readthedocs.io/en/latest/Features.html
    """

    config_class = LGBMForecasterConfig

    def __init__(self, config: LGBMForecasterConfig):
        super().__init__(config)
        self.model = MultiOutputRegressor(
            LGBMRegressor(
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            ),
            n_jobs=1,
        )
