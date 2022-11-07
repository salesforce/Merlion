#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Automatic seasonality detection.
"""
from abc import abstractmethod
from enum import Enum, auto
import logging
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import argrelmax
from scipy.stats import norm
import statsmodels.api as sm

from merlion.models.automl.base import AutoMLMixIn
from merlion.models.base import ModelBase
from merlion.models.layers import LayeredModelConfig
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


class PeriodicityStrategy(Enum):
    """
    Strategy to choose the seasonality if multiple candidates are detected.
    """

    ACF = auto()
    """
    Select the seasonality value with the highest autocorrelation.
    """
    Min = auto()
    """
    Select the minimum seasonality.
    """
    Max = auto()
    """
    Select the maximum seasonality.
    """
    All = auto()
    """
    Use all seasonalities. Only valid for models which support multiple seasonalities.
    """


class SeasonalityModel(metaclass=AutodocABCMeta):
    """
    Class provides simple implementation to set the seasonality in a model. Extend this class to implement custom
    behavior for seasonality processing.
    """

    @abstractmethod
    def set_seasonality(self, theta, train_data: UnivariateTimeSeries):
        """
        Implement this method to do any model-specific adjustments on the seasonality that was provided by
        `SeasonalityLayer`.

        :param theta: Seasonality processed by `SeasonalityLayer`.
        :param train_data: Training data (or numpy array representing the target univariate)
            for any model-specific adjustments you might want to make.
        """
        raise NotImplementedError


class SeasonalityConfig(LayeredModelConfig):
    """
    Config object for an automatic seasonality detection layer.
    """

    _default_transform = TemporalResample()

    def __init__(
        self, model, periodicity_strategy=PeriodicityStrategy.ACF, pval: float = 0.05, max_lag: int = None, **kwargs
    ):
        """
        :param periodicity_strategy: Strategy to choose the seasonality if multiple candidates are detected.
        :param pval: p-value for deciding whether a detected seasonality is statistically significant.
        :param max_lag: max lag considered for seasonality detection.
        """
        self.periodicity_strategy = periodicity_strategy
        assert 0 < pval < 1
        self.pval = pval
        self.max_lag = max_lag
        super().__init__(model=model, **kwargs)

    @property
    def multi_seasonality(self):
        """
        :return: Whether the model supports multiple seasonalities. ``False`` unless explicitly overridden.
        """
        return False

    @property
    def periodicity_strategy(self) -> PeriodicityStrategy:
        """
        :return: Strategy to choose the seasonality if multiple candidates are detected.
        """
        return self._periodicity_strategy

    @periodicity_strategy.setter
    def periodicity_strategy(self, p: Union[PeriodicityStrategy, str]):
        if not isinstance(p, PeriodicityStrategy):
            valid = {k.lower(): k for k in PeriodicityStrategy.__members__}
            assert p.lower() in valid, f"Unsupported PeriodicityStrategy {p}. Supported values: {valid.values()}"
            p = PeriodicityStrategy[valid[p.lower()]]

        if p is PeriodicityStrategy.All and not self.multi_seasonality:
            raise ValueError(
                "Periodicity strategy All is not supported for a model which does not support multiple seasonalities."
            )

        self._periodicity_strategy = p


class SeasonalityLayer(AutoMLMixIn):
    """
    Seasonality Layer that uses automatically determines the seasonality of your data. Can be used directly on
    any model that implements `SeasonalityModel` class. The algorithmic idea is from the
    `theta method <https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R>`__. We find a set of
    multiple candidate seasonalites, and we return the best one(s) based on the `PeriodicityStrategy`.
    """

    config_class = SeasonalityConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self):
        return getattr(self.config, "target_seq_index", None) is None

    @property
    def multi_seasonality(self):
        """
        :return: Whether the model supports multiple seasonalities.
        """
        return self.config.multi_seasonality

    @property
    def periodicity_strategy(self):
        """
        :return: Strategy to choose the seasonality if multiple candidates are detected.
        """
        return self.config.periodicity_strategy

    @property
    def pval(self):
        """
        :return: p-value for deciding whether a detected seasonality is statistically significant.
        """
        return self.config.pval

    @property
    def max_lag(self):
        """
        :return: max_lag for seasonality detection
        """
        return self.config.max_lag

    @staticmethod
    def detect_seasonality(
        x: np.array,
        max_lag: int = None,
        pval: float = 0.05,
        periodicity_strategy: PeriodicityStrategy = PeriodicityStrategy.ACF,
    ) -> List[int]:
        """
        Helper method to detect the seasonality of a time series.

        :param x: The numpy array of values whose seasonality we want to detect. Must be univariate & flattened.
        :param periodicity_strategy: Strategy to choose the seasonality if multiple candidates are detected.
        :param pval: p-value for deciding whether a detected seasonality is statistically significant.
        :param max_lag: max lag considered for seasonality detection.
        """
        # compute max lag & acf function
        # compute max lag & acf function
        max_lag = max(min(int(10 * np.log10(x.shape[0])), x.shape[0] - 1), 40) if max_lag is None else max_lag
        xacf = sm.tsa.acf(x, nlags=max_lag, fft=False)
        xacf[np.isnan(xacf)] = 0

        # select the local maximum points with acf > 0
        candidates = np.intersect1d(np.where(xacf > 0), argrelmax(xacf)[0])

        # the periods should be smaller than one half of the length of time series
        candidates = candidates[candidates < int(x.shape[0] / 2)]
        if candidates.shape[0] != 0:
            candidates_idx = []
            if candidates.shape[0] == 1:
                candidates_idx += [0]
            else:
                if xacf[candidates[0]] > xacf[candidates[1]]:
                    candidates_idx += [0]
                if xacf[candidates[-1]] > xacf[candidates[-2]]:
                    candidates_idx += [-1]
                candidates_idx += argrelmax(xacf[candidates])[0].tolist()
            candidates = candidates[candidates_idx]

            # statistical test if acf is significant w.r.t a normal distribution
            xacf = xacf[1:]
            tcrit = norm.ppf(1 - pval / 2)
            clim = tcrit / np.sqrt(x.shape[0]) * np.sqrt(np.cumsum(np.insert(np.square(xacf) * 2, 0, 1)))
            candidates = candidates[xacf[candidates - 1] > clim[candidates - 1]]

            # sort candidates by ACF value
            candidates = sorted(candidates.tolist(), key=lambda c: xacf[c - 1], reverse=True)
        if len(candidates) == 0:
            candidates = [1]

        # choose the desired candidates based on periodicity strategy
        if periodicity_strategy is PeriodicityStrategy.ACF:
            candidates = [candidates[0]]
        elif periodicity_strategy is PeriodicityStrategy.Min:
            candidates = [min(candidates)]
        elif periodicity_strategy is PeriodicityStrategy.Max:
            candidates = [max(candidates)]
        elif periodicity_strategy is PeriodicityStrategy.All:
            candidates = candidates
        else:
            raise ValueError(f"Periodicity strategy {periodicity_strategy} not supported.")
        return candidates

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        model.set_seasonality(theta, train_data.univariates[self.target_name])

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None
    ) -> Tuple[Any, Optional[ModelBase], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:
        # If multiple seasonalities are supported, return a list of all detected seasonalities
        return (list(thetas) if self.config.multi_seasonality else next(thetas)), None, None

    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        x = train_data.univariates[self.target_name].np_values
        candidates = self.detect_seasonality(
            x=x, max_lag=self.max_lag, pval=self.pval, periodicity_strategy=self.periodicity_strategy
        )
        if candidates[: None if self.config.multi_seasonality else 1] != [1]:
            logger.info(f"Automatically detect the periodicity is {candidates}")
        return iter(candidates)
