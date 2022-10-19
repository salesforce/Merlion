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

from merlion.models.forecast.sklearn_base import SKLearnForecaster, SKLearnForecasterConfig
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.utils.time_series import to_pd_datetime, TimeSeries

logger = logging.getLogger(__name__)


class TreeEnsembleForecasterConfig(SKLearnForecasterConfig):
    """
    Configuration class for bagging tree-based forecaster model.
    """

    def __init__(self, maxlags: int, n_estimators: int = 100, random_state=None, max_depth=None, **kwargs):
        """
        :param maxlags: Max # of lags for forecasting
        :param n_estimators: number of base estimators for the tree ensemble
        :param random_state: random seed for bagging
        :param max_depth: max depth of base estimators
        """
        super().__init__(maxlags=maxlags, **kwargs)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth


class RandomForestForecasterConfig(TreeEnsembleForecasterConfig):
    """
    Config class for `RandomForestForecaster`.
    """

    def __init__(self, maxlags: int, min_samples_split=2, **kwargs):
        """
        :param min_samples_split: min split for tree leaves
        """
        super().__init__(maxlags=maxlags, **kwargs)
        self.min_samples_split = min_samples_split


class RandomForestForecaster(SKLearnForecaster):
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


class ExtraTreesForecaster(SKLearnForecaster):
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


class LGBMForecaster(SKLearnForecaster):
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
