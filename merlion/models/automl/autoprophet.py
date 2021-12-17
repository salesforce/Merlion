#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic (multi)-seasonality detection for Facebook's Prophet.
"""
from typing import Union

from merlion.models.automl.seasonality import PeriodicityStrategy, SeasonalityConfig, SeasonalityLayer
from merlion.models.forecast.prophet import Prophet


class AutoProphetConfig(SeasonalityConfig):
    """
    Config class for Prophet with automatic seasonality detection.
    """

    def __init__(
        self,
        model: Union[Prophet, dict] = None,
        periodicity_strategy: PeriodicityStrategy = PeriodicityStrategy.All,
        **kwargs,
    ):
        model = dict(name="Prophet") if model is None else model
        super().__init__(model=model, periodicity_strategy=periodicity_strategy, **kwargs)

    @property
    def multi_seasonality(self):
        """
        :return: ``True`` because Prophet supports multiple seasonality.
        """
        return True


class AutoProphet(SeasonalityLayer):
    """
    Prophet with automatic seasonality detection. Automatically detects and adds
    additional seasonalities that the existing Prophet may not detect (e.g. hourly).
    """

    config_class = AutoProphetConfig
