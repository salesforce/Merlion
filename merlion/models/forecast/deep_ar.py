#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import copy
import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from typing import List, Optional, Tuple, Union
from abc import abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)


from merlion.models.base import NormalizingConfig
from merlion.models.deep_base import TorchModel
from merlion.models.forecast.deep_models.deep_forecast_base import DeepForecasterConfig, DeepForecaster
from merlion.utils.misc import initializer

logger = logging.getLogger(__name__)


class DeepARConfig(DeepForecasterConfig):
    pass


class DeepARModel(TorchModel):
    pass


class DeepARForecaster(DeepForecaster):
    config_class = DeepARConfig
    deep_model_class = DeepARModel

    def __init__(self, config: ETSformerModel):
        super().__init__(config)
