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

    def __init__(self, max_forecast_steps=None, target_seq_index=None, order=(4, 1, 2), **kwargs):
        if "seasonal_order" in kwargs:
            raise ValueError("cannot specify seasonal_order for ARIMA")
        super().__init__(
            max_forecast_steps=max_forecast_steps,
            target_seq_index=target_seq_index,
            order=order,
            seasonal_order=(0, 0, 0, 0),
            **kwargs
        )

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
