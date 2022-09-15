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
import logging
from typing import Any, Iterator, Optional, Tuple, Union
import time

import pandas as pd

from merlion.models.layers import Config, ModelBase, LayeredModel, ForecasterBase
from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


class AutoMLMixIn(LayeredModel, metaclass=AutodocABCMeta):
    """
    Abstract base class which converts `LayeredModel` into an AutoML model.
    """

    @property
    def _pandas_train(self):
        return False

    def _train_with_exog(self, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None):
        """
        Generates a set of candidate models and picks the best one.

        :param train_data: the data to train on.
        :param train_config: the train config of the underlying model (optional).
        """
        # don't call train_pre_process() in generate/evaluate theta. get model.train_data for the original train data.
        candidate_thetas = self.generate_theta(train_data)
        theta, model, train_result = self.evaluate_theta(candidate_thetas, train_data, exog_data=exog_data)
        if model is not None:
            self.model = model
            return train_result
        else:
            model = deepcopy(self.model)
            model.reset()
            self.set_theta(model, theta, train_data)
            self.model = model
            train_data = train_data.to_pd() if self.model._pandas_train else train_data
            exog_data = exog_data.to_pd() if exog_data is not None and self.model._pandas_train else exog_data
            if exog_data is None:
                return self.model._train(train_data, train_config=train_config)
            else:
                return self.model._train_with_exog(train_data, train_config=train_config, exog_data=exog_data)

    def _train(self, train_data: TimeSeries, train_config=None):
        return self._train_with_exog(train_data, train_config=train_config, exog_data=None)

    @abstractmethod
    def generate_theta(self, train_data: TimeSeries) -> Iterator:
        r"""
        :param train_data: Pre-processed training data to use for generation of hyperparameters :math:`\theta`

        Returns an iterator of hyperparameter candidates for consideration with th underlying model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None
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


class ICConfig(Config):
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


class ICAutoMLForecaster(AutoMLMixIn, ForecasterBase, metaclass=AutodocABCMeta):
    """
    AutoML model which uses an information criterion to determine which model paramters are best.
    """

    config_class = ICConfig

    @property
    def information_criterion(self):
        return self.config.information_criterion

    @abstractmethod
    def get_ic(
        self, model, train_data: pd.DataFrame, train_result: Tuple[pd.DataFrame, Optional[pd.DataFrame]]
    ) -> float:
        """
        Returns the information criterion of the model based on the given training data & the model's train result.

        :param model: One of the models being tried. Must be trained.
        :param train_data: The target sequence of the training data as a ``pandas.DataFrame``.
        :param train_result: The result of calling ``model._train()``.
        :return: The information criterion evaluating the model's goodness of fit.
        """
        raise NotImplementedError

    @abstractmethod
    def _model_name(self, theta) -> str:
        """
        :return: a string describing the current model.
        """
        raise NotImplementedError

    def evaluate_theta(
        self, thetas: Iterator, train_data: TimeSeries, train_config=None, exog_data: TimeSeries = None
    ) -> Tuple[Any, ModelBase, Tuple[TimeSeries, Optional[TimeSeries]]]:
        best = None
        y = train_data.to_pd() if self.model._pandas_train else train_data
        y_exog = exog_data.to_pd() if exog_data is not None and self.model._pandas_train else exog_data
        y_target = pd.DataFrame(y[self.model.target_name])
        for theta in thetas:
            # Start timer & fit model using the current theta
            start = time.time()
            model = deepcopy(self.model)
            self.set_theta(model, theta, train_data)
            if exog_data is None:
                train_result = model._train(y, train_config=train_config)
            else:
                train_result = model._train_with_exog(y, train_config=train_config, exog_data=y_exog)
            fit_time = time.time() - start
            ic = float(self.get_ic(model=model, train_data=y_target, train_result=train_result))
            logger.debug(f"{self._model_name(theta)}: {self.information_criterion.name}={ic:.3f}, Time={fit_time:.2f}s")

            # Determine if current model is better than the best seen yet
            curr = {"theta": theta, "model": model, "train_result": train_result, "ic": ic}
            if best is None:
                best = curr
                logger.debug("First best model found (%.3f)" % ic)
            current_ic = best["ic"]
            if ic < current_ic:
                logger.debug("New best model found (%.3f < %.3f)" % (ic, current_ic))
                best = curr

        # Return best model after post-processing its train result
        theta, model, train_result = best["theta"], best["model"], best["train_result"]
        return theta, model, train_result
