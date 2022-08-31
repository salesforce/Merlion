#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Base class/mixin for AutoML hyperparameter search.
"""
from abc import abstractmethod
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Iterator, Optional, Tuple, Union

from merlion.models.layers import Config, ModelBase, LayeredModel, ForecastingDetectorBase
from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta


class InformationCriterion(Enum):
    AIC = auto()
    r"""
    Akaike information criterion. Computed as
    
    .. math::
        \mathrm{AIC} = 2k - 2\mathrm{ln}(L)
    
    where k is the number of parameters, and L is the model's likelihood.
    """

    BIC = auto()
    r"""
    Bayesian information criterion. Computed as
    
    .. math::
        k \mathrm{ln}(n) - 2 \mathrm{ln}(L)
    
    where n is the sample size, k is the number of parameters, and L is the model's likelihood.
    """

    AICc = auto()
    r"""
    Akaike information criterion with correction for small sample size. Computed as
    
    .. math::
        \mathrm{AICc} = \mathrm{AIC} + \frac{2k^2 + 2k}{n - k - 1}
    
    where n is the sample size, and k is the number of paramters.
    """


class InformationCriterionConfigMixIn(Config):
    """
    Mix-in to add an information criterion parameter to a model config.
    """

    def __init__(self, information_criterion: InformationCriterion = InformationCriterion.AIC, **kwargs):
        """
        :param information_criterion: information criterion to select the best model.
        """
        super().__init__(**kwargs)
        self.information_criterion = information_criterion

    @property
    def information_criterion(self):
        return self._information_criterion

    @information_criterion.setter
    def information_criterion(self, ic: Union[InformationCriterion, str]):
        if not isinstance(ic, InformationCriterion):
            valid = {k.lower(): k for k in InformationCriterion.__members__}
            assert ic.lower() in valid, f"Unsupported InformationCriterion {ic}. Supported values: {valid.values()}"
            ic = InformationCriterion[valid[ic.lower()]]
        self._information_criterion = ic


class AutoMLMixIn(LayeredModel, metaclass=AutodocABCMeta):
    """
    Abstract base class which converts `LayeredModel` into an AutoML model.
    """

    def train_model(self, train_data: TimeSeries, train_config=None, **kwargs):
        """
        Generates a set of candidate models and picks the best one.

        :param train_data: the data to train on.
        :param train_config: the train config of the underlying model (optional).
        """
        # don't call train_pre_process() in generate/evaluate theta. get model.train_data for the original train data.
        processed_train_data = self.model.train_pre_process(train_data)
        candidate_thetas = self.generate_theta(processed_train_data)
        theta, model, train_result = self.evaluate_theta(candidate_thetas, processed_train_data, **kwargs)
        if model is not None:
            self.model = model
            return model.train_post_process(train_result, **kwargs)
        else:
            model = deepcopy(self.model)
            model.reset()
            self.set_theta(model, theta, processed_train_data)
            self.model = model
            return super().train_model(train_data, **kwargs)

    @abstractmethod
    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        r"""
        :param train_data: Pre-processed training data to use for generation of hyperparameters :math:`\theta`

        Returns an iterator of hyperparameter candidates for consideration with th underlying model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None, **kwargs
    ) -> Tuple[Any, Optional[ModelBase], Optional[Tuple[TimeSeries, Optional[TimeSeries]]]]:
        r"""
        :param thetas: Iterator of the hyperparameter candidates
        :param train_data: Pre-processed training data
        :param train_config: Training configuration

        Return the optimal hyperparameter, as well as optionally a model and result of the training procedure.
        """
        raise NotImplementedError

    @abstractmethod
    def set_theta(self, model, theta, train_data: TimeSeries = None):
        r"""
        :param model: Underlying base model to which the new theta is applied
        :param theta: Hyperparameter to apply
        :param train_data: Pre-processed training data (Optional)

        Sets the hyperparameter to the provided ``model``. This is used to apply the :math:`\theta` to the model, since
        this behavior is custom to every model. Oftentimes in internal implementations, ``model`` is the optimal model.
        """
        raise NotImplementedError
