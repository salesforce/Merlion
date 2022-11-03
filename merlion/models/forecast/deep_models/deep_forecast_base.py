#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Base class for Deep Learning Forecasting Models
"""

import copy
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from typing import List, Optional, Tuple, Union
from abc import abstractmethod


from merlion.models.deep_base import DeepConfig, DeepModelBase
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig

from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.utils.misc import initializer
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy

logger = logging.getLogger(__name__)


class DeepForecastConfig(DeepConfig, ForecasterConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 256,
        num_epochs: int = 10,
        max_forecast_steps: int = None,
        target_seq_index: int = None,
        invert_transform=None,
        **kwargs
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            max_forecast_steps=max_forecast_steps,
            target_seq_index=target_seq_index,
            invert_transform=invert_transform,
            **kwargs
        )


class DeepForecaster(DeepModelBase):
    def __init__(self, config):
        pass
