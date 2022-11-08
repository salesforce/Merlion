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
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

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
        # Use the residuals of an ETS model fit to the data, to handle any trend. This makes the ACF more robust.
        # For each candidate seasonality, the ACF we assign is the higher of the raw ACF and residual ACF.
        candidate2score = {}
        y = x if len(x) < 10 else ETSModel(x, error="add", trend="add").fit(disp=False).resid
        for x in [x] if x is y else [x, y]:
            # compute max lag & ACF function
            max_lag = max(min(int(10 * np.log10(len(x))), len(x) - 1), 40) if max_lag is None else max_lag
            xacf = sm.tsa.acf(x, nlags=max_lag, fft=False)
            xacf[np.isnan(xacf)] = 0

            # select the local maximum points with acf > 0, and smaller than 1/2 the length of the time series
            xacf = xacf[: np.ceil(len(x) / 2).astype(int)]
            candidates = np.intersect1d(np.where(xacf > 0), argrelmax(xacf)[0])

            if len(candidates) > 0:
                # filter out potential harmonics by applying peak-finding on the peaks of the ACF
                if len(candidates) > 1:
                    candidates_idx = []
                    if xacf[candidates[0]] > xacf[candidates[1]]:
                        candidates_idx += [0]
                    candidates_idx += argrelmax(xacf[candidates])[0].tolist()
                    if xacf[candidates[-1]] > xacf[candidates[-2]]:
                        candidates_idx += [-1]
                    candidates = candidates[candidates_idx]

                # statistical test if ACF is significant with respect to a normal distribution
                xacf = xacf[1:]
                xacf_var = np.cumsum(np.concatenate(([1], 2 * xacf[:-1] ** 2))) / len(x)
                z_scores = xacf / np.sqrt(xacf_var)
                candidates = candidates[z_scores[candidates - 1] > norm.ppf(1 - pval / 2)]
                for c in candidates.tolist():
                    candidate2score[c] = max(candidate2score.get(c, -np.inf), z_scores[c - 1])

        # sort the candidates by z-score and choose the desired candidates based on periodicity strategy
        candidates = sorted(candidate2score.keys(), key=lambda c: candidate2score[c], reverse=True)
        for c, s in candidate2score.items():
            logger.info(f"Detected seas = {c:3d} with z-score = {s:5.2f}.")
        if len(candidates) == 0:
            candidates = [1]
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
