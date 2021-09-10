#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Rules for combining the outputs of multiple time series models.
"""
from abc import abstractmethod
from collections import OrderedDict
import copy
import logging
from typing import List, Optional, Union

import numpy as np

from merlion.evaluate.anomaly import TSADMetric
from merlion.evaluate.forecast import ForecastMetric
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.utils.misc import AutodocABCMeta

logger = logging.getLogger(__name__)


def _align_outputs(all_model_outs: List[TimeSeries], target: TimeSeries) -> List[Optional[TimeSeries]]:
    """
    Aligns the outputs of each model to the time series ``target``.
    """
    if all(out is None for out in all_model_outs):
        return [None for _ in all_model_outs]
    if target is None:
        time_stamps = np.unique(np.concatenate([out.to_pd().index for out in all_model_outs if out is not None]))
    else:
        t0 = min(min(v.index[0] for v in out.univariates) for out in all_model_outs if out is not None)
        tf = max(max(v.index[-1] for v in out.univariates) for out in all_model_outs if out is not None)
        time_stamps = target.to_pd()[t0:tf].index
    return [None if out is None else out.align(reference=time_stamps) for out in all_model_outs]


class CombinerBase(metaclass=AutodocABCMeta):
    """
    Abstract base class for combining the outputs of multiple models. Subclasses
    should implement the abstract method ``_combine_univariates``. All combiners
    are callable objects.

    .. automethod:: __call__
    """

    def __init__(self, abs_score=False):
        """
        :param abs_score: whether to take the absolute value of the model
            outputs. Useful for anomaly detection.
        """
        self.abs_score = abs_score
        self.n_models = None

    @property
    def requires_training(self):
        return False

    def to_dict(self, _skipped_keys=None):
        skipped_keys = set() if _skipped_keys is None else _skipped_keys
        state = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in skipped_keys}
        state["name"] = type(self).__name__
        return state

    @classmethod
    def from_dict(cls, state):
        state = copy.copy(state)
        state.pop("name", None)
        n_models = state.pop("n_models", None)
        ret = cls(**state)
        ret.n_models = n_models
        return ret

    def __copy__(self):
        return self.from_dict(self.to_dict())

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    @abstractmethod
    def _combine_univariates(self, univariates: List[UnivariateTimeSeries]):
        raise NotImplementedError

    @property
    def models_used(self) -> List[bool]:
        """
        :return: which models are actually used to make predictions.
        """
        assert self.n_models is not None, "Combiner must be trained to determine which models are used"
        return [True] * self.n_models

    def train(self, all_model_outs: List[TimeSeries], target: TimeSeries = None) -> TimeSeries:
        """
        Trains the model combination rule.

        :param all_model_outs: a list of time series, with each time series
            representing the output of a single model.
        :param target: a target time series (e.g. labels)
        :return: a single time series of combined model outputs on this training data.
        """
        self.n_models = len(all_model_outs)
        return self(all_model_outs, target, _check_dim=False)

    def __call__(self, all_model_outs: List[TimeSeries], target: TimeSeries, _check_dim=True) -> TimeSeries:
        """
        Applies the model combination rule to combine multiple model outputs.

        :param all_model_outs: a list of time series, with each time series
            representing the output of a single model.
        :param target: a target time series (e.g. labels)
        :return: a single time series of combined model outputs on this training data.
        """
        if isinstance(target, list):
            new_all_model_outs = []
            for i, out in enumerate(all_model_outs):
                if out is None:
                    new_all_model_outs.append(out)
                else:
                    assert isinstance(out, list) and len(out) == len(target), (
                        f"If target is a list of time series, each model output should be a "
                        f"list with the same length, but target has length {len(target)}, "
                        f"while model output {i} is a {type(out).__name__} of length {len(out)}"
                    )
                    new_all_model_outs.append(sum(out[1:], out[0]))
            target = sum(target[1:], target[0])
            all_model_outs = new_all_model_outs

        js = [j for j, out in enumerate(all_model_outs) if out is not None]
        assert len(js) > 0, "`all_model_outs` cannot all be `None`"
        j = js[0]
        assert all(out.dim == all_model_outs[j].dim for out in all_model_outs if out is not None)
        if self.n_models is None:
            self.n_models = len(all_model_outs)

        models_used = self.models_used
        if len(all_model_outs) == self.n_models:
            j = 0
            all_model_outs = [x for x, used in zip(all_model_outs, models_used) if used]
        elif len(all_model_outs) != sum(models_used):
            raise RuntimeError(
                f"Expected either {self.n_models} or {sum(models_used)} "
                f"model outputs, but got {len(all_model_outs)} model outputs "
                f"instead."
            )

        all_model_outs = _align_outputs(all_model_outs, target)
        if all(out is None for out in all_model_outs):
            return None

        combined = OrderedDict()
        for i in range(all_model_outs[j].dim):
            name = all_model_outs[j].names[i]
            all_i = [None if ts is None else ts.univariates[ts.names[i]] for ts in all_model_outs]
            combined[name] = self._combine_univariates(all_i)

        return TimeSeries(combined)


class Mean(CombinerBase):
    """
    Combines multiple models by taking their mean prediction.
    """

    @property
    def weights(self) -> np.ndarray:
        n = sum(self.models_used)
        return np.full(shape=n, fill_value=1 / n)

    def _combine_univariates(self, univariates: List[UnivariateTimeSeries]) -> UnivariateTimeSeries:
        non_none = [var for var in univariates if var is not None]
        weights = np.asarray([w for w, var in zip(self.weights, univariates) if var is not None])
        weights = weights / weights.sum()
        v = non_none[0]
        if self.abs_score and sum(self.models_used) > 1:
            signs = np.median(np.sign([var.np_values for var in non_none]), axis=0)
            signs[signs == 0] = -1
            new_vals = signs * np.dot(weights, [np.abs(var.np_values) for var in non_none])
        else:
            new_vals = np.dot(weights, [var.np_values for var in non_none])
        return UnivariateTimeSeries(v.time_stamps, new_vals, v.name)


class Median(CombinerBase):
    """
    Combines multiple models by taking their median prediction.
    """

    def _combine_univariates(self, univariates: List[UnivariateTimeSeries]) -> UnivariateTimeSeries:
        non_none = [var for var in univariates if var is not None]
        v = non_none[0]
        if self.abs_score and sum(self.models_used) > 1:
            signs = np.median(np.sign([var.np_values for var in non_none]), axis=0)
            signs[signs == 0] = -1
            new_vals = signs * np.median([np.abs(var.np_values) for var in non_none], axis=0)
        else:
            new_vals = np.median([var.np_values for var in non_none], axis=0)
        return UnivariateTimeSeries(v.time_stamps, new_vals, v.name)


class Max(CombinerBase):
    """
    Combines multiple models by taking their max prediction.
    """

    def _combine_univariates(self, univariates: List[UnivariateTimeSeries]) -> UnivariateTimeSeries:
        non_none = [var for var in univariates if var is not None]
        v = non_none[0]
        if self.abs_score and sum(self.models_used) > 1:
            signs = np.median(np.sign([var.np_values for var in non_none]), axis=0)
            signs[signs == 0] = -1
            new_vals = signs * np.median([np.abs(var.np_values) for var in non_none], axis=0)
        else:
            new_vals = np.max([var.np_values for var in non_none], axis=0)
        return UnivariateTimeSeries(v.time_stamps, new_vals, v.name)


class ModelSelector(Mean):
    """
    Takes the mean of the best models, where the models are ranked according to
    the value of an evaluation metric.
    """

    def __init__(self, metric: Union[str, TSADMetric, ForecastMetric], abs_score=False):
        """
        :param metric: the evaluation metric to use
        :param abs_score: whether to take the absolute value of the model
            outputs. Useful for anomaly detection.
        """
        super().__init__(abs_score=abs_score)
        if isinstance(metric, str):
            metric_cls, name = metric.split(".", maxsplit=1)
            metric_cls = {c.__name__: c for c in [ForecastMetric, TSADMetric]}[metric_cls]
            metric = metric_cls[name]
        self.metric = metric
        self.metric_values = None

    @property
    def invert(self):
        if isinstance(self.metric, ForecastMetric):
            return True
        if self.metric is TSADMetric.MeanTimeToDetect:
            return True
        return False

    @property
    def requires_training(self):
        return True

    def to_dict(self, _skipped_keys=None):
        skipped_keys = set() if _skipped_keys is None else _skipped_keys
        state = super().to_dict(skipped_keys.union({"metric"}))
        state["metric"] = f"{type(self.metric).__name__}.{self.metric.name}"
        return state

    @classmethod
    def from_dict(cls, state):
        # Extract the metric values from the state (to set manually later)
        metric_values = state.pop("metric_values", None)
        ret = super().from_dict(state)
        ret.metric_values = metric_values
        return ret

    @property
    def models_used(self) -> List[bool]:
        assert self.n_models is not None, "Combiner must be trained to determine which models are used"
        metric_values = np.asarray(self.metric_values)
        val = np.min(metric_values) if self.invert else np.max(metric_values)
        return (metric_values == val).tolist()

    def train(self, all_model_outs: List[TimeSeries], target: TimeSeries = None, **kwargs) -> TimeSeries:
        metric_values = []
        assert all(x is not None for x in all_model_outs), f"None of `all_model_outs` can be `None`"
        self.n_models = len(all_model_outs)
        for i, model_out in enumerate(all_model_outs):
            if target is None and self.metric_values is None:
                metric_values.append(1)
            elif target is not None and not isinstance(target, list):
                metric_values.append(self.metric.value(ground_truth=target, predict=model_out, **kwargs))
            elif isinstance(target, list):
                assert isinstance(model_out, list) and len(model_out) == len(target), (
                    f"If target is a list of time series, each model output should be a "
                    f"list with the same length, but target has length {len(target)}, "
                    f"while model output {i} is a {type(model_out).__name__} of length "
                    f"{len(model_out)}"
                )
                vals = [self.metric.value(ground_truth=y, predict=yhat, **kwargs) for y, yhat in zip(target, model_out)]
                metric_values.append(np.mean(vals))

        if len(metric_values) == len(all_model_outs):
            self.metric_values = metric_values
        return self(all_model_outs, target)


class MetricWeightedMean(ModelSelector):
    """
    Computes a weighted average of model outputs with weights proportional to
    the metric values (or their inverses).
    """

    @property
    def models_used(self) -> List[bool]:
        return CombinerBase.models_used.fget(self)

    @property
    def weights(self) -> np.ndarray:
        w = np.asarray(self.metric_values)
        w = 1 / w if self.invert else w
        return w / w.sum()


class CombinerFactory(object):
    """
    Factory object for creating combiner objects.
    """

    @classmethod
    def create(cls, name: str, **kwargs) -> CombinerBase:
        alias = {cls.__name__: cls for cls in [Mean, Median, Max, ModelSelector, MetricWeightedMean]}
        combiner_class = alias[name]
        return combiner_class.from_dict(kwargs)
