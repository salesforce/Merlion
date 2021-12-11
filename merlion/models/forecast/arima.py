#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The classic statistical forecasting model ARIMA (AutoRegressive Integrated
Moving Average).
"""
import logging
from typing import Tuple

from merlion.models.forecast.sarima import SarimaConfig, Sarima
from merlion.transform.resample import TemporalResample

logger = logging.getLogger(__name__)


class ArimaConfig(SarimaConfig):
    """
    Configuration class for `Arima`. Just a `Sarima` model with seasonal order ``(0, 0, 0, 0)``.
    """

    _default_transform = TemporalResample(granularity=None, trainable_granularity=True)

    def __init__(self, order=(4, 1, 2), seasonal_order=(0, 0, 0, 0), **kwargs):
        """
        :param seasonal_order: (0, 0, 0, 0) because ARIMA has no seasonal order.
        """
        super().__init__(order=order, seasonal_order=seasonal_order, **kwargs)

    @property
    def seasonal_order(self) -> Tuple[int, int, int, int]:
        """
        :return: (0, 0, 0, 0) because ARIMA has no seasonal order.
        """
        return 0, 0, 0, 0

    @seasonal_order.setter
    def seasonal_order(self, seasonal_order: Tuple[int, int, int, int]):
        assert tuple(seasonal_order) == (0, 0, 0, 0), "Seasonal order must be (0, 0, 0, 0) for ARIMA"


class Arima(Sarima):
    """
    Implementation of the classic statistical model ARIMA (AutoRegressive
    Integrated Moving Average) for forecasting.
    """

    config_class = ArimaConfig
