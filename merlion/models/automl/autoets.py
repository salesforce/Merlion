#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic seasonality detection for ETS.
"""

from typing import Union

from merlion.models.forecast.ets import ETS
from merlion.models.automl.seasonality import SeasonalityConfig, SeasonalityLayer


class AutoETSConfig(SeasonalityConfig):
    """
    Config class for ETS with automatic seasonality detection.
    """

    def __init__(self, model: Union[ETS, dict] = None, **kwargs):
        model = dict(name="ETS") if model is None else model
        super().__init__(model=model, **kwargs)


class AutoETS(SeasonalityLayer):
    """
    ETS with automatic seasonality detection.
    """

    config_class = AutoETSConfig
