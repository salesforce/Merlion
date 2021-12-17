#
# Copyright (c) 2021 salesforce.com, inc.
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
from typing import Any, Iterator, Optional, Tuple, Union

from merlion.models.automl.base import AutoMLMixIn
from merlion.models.base import ModelBase
from merlion.models.layers import LayeredModelConfig
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries, UnivariateTimeSeries, autosarima_utils
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

    def __init__(self, model, periodicity_strategy=PeriodicityStrategy.ACF, pval: float = 0.05, **kwargs):
        """
        :param periodicity_strategy: Strategy to choose the seasonality if multiple candidates are detected.
        :param pval: p-value for deciding whether a detected seasonality is statistically significant.
        """
        self.periodicity_strategy = periodicity_strategy
        assert 0 < pval < 1
        self.pval = pval
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
            assert p.lower() in valid, f"Unsupported PeriodicityStrategy {p}. Supported strategies are: {valid.keys()}"
            p = PeriodicityStrategy[valid[p.lower()]]

        if p is PeriodicityStrategy.All and not self.multi_seasonality:
            raise ValueError(
                "Periodicity strategy All is not supported for a model which does not support multiple seasonalities."
            )

        self._periodicity_strategy = p

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"periodicity_strategy"}))
        if "periodicity_strategy" not in _skipped_keys:
            config_dict["periodicity_strategy"] = self.periodicity_strategy.name
        return config_dict


class SeasonalityLayer(AutoMLMixIn, metaclass=AutodocABCMeta):
    """
    Seasonality Layer that uses AutoSARIMA-like methods to determine seasonality of your data. Can be used directly on
    any model that implements `SeasonalityModel` class.
    """

    config_class = SeasonalityConfig
    require_even_sampling = False

    @property
    def require_univariate(self):
        return getattr(self.config, "target_seq_index", None) is not None

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

    def set_theta(self, model, theta, train_data: TimeSeries = None):
        model.set_seasonality(theta, train_data.univariates[self.target_name])

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None
    ) -> Tuple[Any, Optional[ModelBase], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:
        # If multiple seasonalities are supported, return a list of all detected seasonalities
        thetas = list(thetas)
        if self.periodicity_strategy is PeriodicityStrategy.ACF:
            thetas = [thetas[0]]
        elif self.periodicity_strategy is PeriodicityStrategy.Min:
            thetas = [min(thetas)]
        elif self.periodicity_strategy is PeriodicityStrategy.Max:
            thetas = [max(thetas)]
        elif self.periodicity_strategy is PeriodicityStrategy.All:
            thetas = thetas
        else:
            raise ValueError(f"Periodicity strategy {self.periodicity_strategy} not supported.")
        theta = thetas if self.config.multi_seasonality else thetas[0]
        if thetas != [1]:
            logger.info(f"Automatically detect the periodicity is {str(thetas)}")
        return theta, None, None

    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        y = train_data.univariates[self.target_name]
        periods = autosarima_utils.multiperiodicity_detection(y, pval=self.pval)
        if len(periods) == 0:
            periods = [1]
        return iter(periods)
