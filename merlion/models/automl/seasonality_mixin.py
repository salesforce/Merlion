#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from abc import ABC
from typing import Iterator, Tuple, Optional, Any

from merlion.models.automl.forecasting_layer_base import ForecasterAutoMLBase
from merlion.models.forecast.base import ForecasterBase, logger
from merlion.utils import TimeSeries, autosarima_utils


class SeasonalityModel(ABC):
    """
    Class provides simple implementation to set the seasonality in a model. Extend this class to implement custom
    behavior for seasonality processing.
    """

    def set_seasonality(self, theta, train_data):
        """
        Implement this method to do any model-specific adjustments on the seasonality that was provided by
        `SeasonalityLayer`.

        :param theta: Seasonality processed by `SeasonalityLayer`.
        :param train_data: Training data (or numpy array representing the target univariate)
            for any model-specific adjustments you might want to make.
        """
        self.seasonality = theta


class SeasonalityLayer(ForecasterAutoMLBase, ABC):
    """
    Seasonality Layer that uses AutoSARIMA-like methods to determine seasonality of your data. Can be used directly on
    any model that implements `SeasonalityModel` class.
    """

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        model.set_seasonality(theta, train_data.univariates[self.target_name])

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None
    ) -> Tuple[Any, Optional[ForecasterBase], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:
        # assume only one seasonality is returned in this case
        return next(thetas), None, None

    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        y = train_data.univariates[self.target_name]

        periodicity_strategy = self.periodicity_strategy

        periods = autosarima_utils.multiperiodicity_detection(y)
        if len(periods) > 0:
            if periodicity_strategy == "min":
                m = periods[0]
            else:
                m = periods[-1]
        else:
            m = 1
        logger.info(f"Automatically detect the periodicity is {str(m)}")
        return iter([m])
