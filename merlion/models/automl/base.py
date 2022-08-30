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
from typing import Any, Iterator, Optional, Tuple

from merlion.models.layers import ModelBase, LayeredModel, ForecastingDetectorBase
from merlion.utils import TimeSeries
from merlion.utils.misc import AutodocABCMeta


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
