#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic seasonality detection.
"""
from abc import ABC, abstractmethod
import logging
from typing import Iterator, Tuple, Optional, Any

from merlion.models.automl.base import AutoMLMixIn
from merlion.models.base import LayeredModelConfig, ModelBase
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries, autosarima_utils

logger = logging.getLogger(__name__)


class SeasonalityModel(ABC):
    """
    Class provides simple implementation to set the seasonality in a model. Extend this class to implement custom
    behavior for seasonality processing.
    """

    @abstractmethod
    def set_seasonality(self, theta, train_data):
        """
        Implement this method to do any model-specific adjustments on the seasonality that was provided by
        `SeasonalityLayer`.

        :param theta: Seasonality processed by `SeasonalityLayer`.
        :param train_data: Training data (or numpy array representing the target univariate)
            for any model-specific adjustments you might want to make.
        """
        raise NotImplementedError


class SeasonalityConfig(LayeredModelConfig):

    _default_transform = TemporalResample()

    def __init__(self, model, periodicity_strategy="max", **kwargs):
        """
        :param periodicity_strategy: selection strategy when detecting multiple
            periods. 'min' signifies to select the smallest period, while 'max' signifies to select
            the largest period
        """
        self.periodicity_strategy = periodicity_strategy
        super().__init__(model=model, **kwargs)


class SeasonalityLayer(AutoMLMixIn, ABC):
    """
    Seasonality Layer that uses AutoSARIMA-like methods to determine seasonality of your data. Can be used directly on
    any model that implements `SeasonalityModel` class.
    """

    config_class = SeasonalityConfig
    require_univariate = True
    require_even_sampling = True

    @property
    def periodicity_strategy(self):
        return self.config.periodicity_strategy

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        model.set_seasonality(theta, train_data.univariates[self.target_name])

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None
    ) -> Tuple[Any, Optional[ModelBase], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:
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
