#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Metrics and utilities for evaluating forecasting models in a continuous sense.
"""
from enum import Enum
from functools import partial
from typing import List, Union, Tuple
import warnings

import numpy as np

from merlion.evaluate.base import EvaluatorBase, EvaluatorConfig
from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries
from merlion.utils.resample import granularity_str_to_seconds


# TODO: support multivariate time series
class ForecastScoreAccumulator:
    """
    Accumulator which maintains summary statistics describing a forecasting
    algorithm's performance. Can be used to compute many different forecasting metrics.
    """

    def __init__(
        self,
        ground_truth: TimeSeries,
        predict: TimeSeries,
        insample: TimeSeries = None,
        periodicity: int = 1,
        ub: TimeSeries = None,
        lb: TimeSeries = None,
    ):
        """
        :param ground_truth: ground truth time series
        :param predict: predicted truth time series
        :param insample (optional): time series used for training model.
            This value is used for computing MSES, MSIS
        :param periodicity (optional): periodicity. m=1 indicates the non-seasonal time series,
            whereas m>1 indicates seasonal time series.
            This value is used for computing MSES, MSIS.
        :param ub (optional): upper bound of 95% prediction interval. This value is used for
            computing MSIS
        :param lb (optional): lower bound of 95% prediction interval. This value is used for
            computing MSIS
        """
        t0, tf = predict.t0, predict.tf
        ground_truth = ground_truth.window(t0, tf, include_tf=True).align()
        self.ground_truth = ground_truth
        self.predict = predict.align(reference=ground_truth.time_stamps)
        self.insample = insample
        self.periodicity = periodicity
        self.ub = ub
        self.lb = lb

    def check_before_eval(self):
        # Make sure time series is univariate
        assert self.predict.dim == self.ground_truth.dim == 1
        # Make sure the timestamps of preds and targets are identical
        assert self.predict.time_stamps == self.ground_truth.time_stamps

    def mae(self):
        """
        Mean Absolute Error (MAE)

        For ground truth time series :math:`y` and predicted time series :math:`\\hat{y}`
        of length :math:`T`, it is computed as

        ..  math:: \\frac{1}{T}\\sum_{t=1}^T{(|y_t - \\hat{y}_t|)}.
        """
        self.check_before_eval()
        predict_values = self.predict.univariates[self.predict.names[0]].np_values
        ground_truth_values = self.ground_truth.univariates[self.ground_truth.names[0]].np_values
        return np.mean(np.abs(ground_truth_values - predict_values))

    def marre(self):
        """
        Mean Absolute Ranged Relative Error (MARRE)

        For ground truth time series :math:`y` and predicted time series :math:`\\hat{y}`
        of length :math:`T`, it is computed as

        .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} -
              \\min_t{y_t}} \\right|}.
        """
        self.check_before_eval()
        predict_values = self.predict.univariates[self.predict.names[0]].np_values
        ground_truth_values = self.ground_truth.univariates[self.ground_truth.names[0]].np_values
        assert ground_truth_values.max() > ground_truth_values.min()
        true_range = ground_truth_values.max() - ground_truth_values.min()
        return 100.0 * np.mean(np.abs((ground_truth_values - predict_values) / true_range))

    def rmse(self):
        """
        Root Mean Squared Error (RMSE)

        For ground truth time series :math:`y` and predicted time series :math:`\\hat{y}`
        of length :math:`T`, it is computed as

        .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}}.
        """
        self.check_before_eval()
        predict_values = self.predict.univariates[self.predict.names[0]].np_values
        ground_truth_values = self.ground_truth.univariates[self.ground_truth.names[0]].np_values
        return np.sqrt(np.mean((ground_truth_values - predict_values) ** 2))

    def smape(self):
        """
        symmetric Mean Absolute Percentage Error (sMAPE). For ground truth time series :math:`y`
        and predicted time series :math:`\\hat{y}` of length :math:`T`, it is computed as

        .. math::
            200 \\cdot \\frac{1}{T}
            \\sum_{t=1}^{T}{\\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right|
            + \\left| \\hat{y}_t \\right|}}.
        """
        self.check_before_eval()
        predict_values = self.predict.univariates[self.predict.names[0]].np_values
        ground_truth_values = self.ground_truth.univariates[self.ground_truth.names[0]].np_values

        errors = np.abs(ground_truth_values - predict_values)
        scale = np.abs(ground_truth_values) + np.abs(predict_values)

        # Make sure the divisor is not close to zero at each timestamp
        if (scale < 1e-8).any():
            warnings.warn("Some values very close to 0, sMAPE might not be estimated accurately.")
        return np.mean(200.0 * errors / (scale + 1e-8))

    def mase(self):
        """
        Mean Absolute Scaled Error (MASE)
        For ground truth time series :math:`y` and predicted time series :math:`\\hat{y}`
        of length :math:`T`. In sample time series :math:`\\hat{x}` of length :math:`N`
        and periodicity :math:`m` it is computed as

        .. math::
            \\frac{1}{T}\\cdot\\frac{\\sum_{t=1}^{T}\\left| y_t
            - \\hat{y}_t \\right|}{\\frac{1}{N-m}\\sum_{t=m+1}^{N}\\left| x_t - x_{t-m} \\right|}.
        """
        self.check_before_eval()
        assert self.insample.dim == 1
        insample_values = self.insample.univariates[self.insample.names[0]].np_values
        predict_values = self.predict.univariates[self.predict.names[0]].np_values
        ground_truth_values = self.ground_truth.univariates[self.ground_truth.names[0]].np_values
        errors = np.abs(ground_truth_values - predict_values)
        scale = np.mean(np.abs(insample_values[self.periodicity :] - insample_values[: -self.periodicity]))

        # Make sure the divisor is not close to zero at each timestamp
        if (scale < 1e-8).any():
            warnings.warn("Some values very close to 0, MASE might not be estimated accurately.")
        return np.mean(errors / (scale + 1e-8))

    def msis(self):
        """
        Mean Scaled Interval Score (MSIS)
        This metric evaluates the quality of 95% prediction intervals.
        For ground truth time series :math:`y` and predicted time series :math:`\\hat{y}`
        of length :math:`T`, the lower and upper bounds of the prediction intervals
        :math:`L` and :math:`U`. Given in sample time series :math:`\\hat{x}` of length :math:`N`
        and periodicity :math:`m`, it is computed as

        .. math::
            \\frac{1}{T}\\cdot\\frac{\\sum_{t=1}^{T} (U_t - L_t) + 100 \\cdot (L_t - y_t)[y_t<L_t]
            + 100\\cdot(y_t - U_t)[y_t > U_t]}{\\frac{1}{N-m}\\sum_{t=m+1}^{N}\\left| x_t - x_{t-m} \\right|}.
        """
        self.check_before_eval()
        assert self.insample.dim == 1
        insample_values = self.insample.univariates[self.insample.names[0]].np_values
        lb_values = self.lb.univariates[self.lb.names[0]].np_values
        ub_values = self.ub.univariates[self.ub.names[0]].np_values
        ground_truth_values = self.ground_truth.univariates[self.ground_truth.names[0]].np_values
        errors = (
            np.sum(ub_values - lb_values)
            + 100 * np.sum((lb_values - ground_truth_values)[lb_values > ground_truth_values])
            + 100 * np.sum((ground_truth_values - ub_values)[ground_truth_values > ub_values])
        )
        scale = np.mean(np.abs(insample_values[self.periodicity :] - insample_values[: -self.periodicity]))

        # Make sure the divisor is not close to zero at each timestamp
        if (scale < 1e-8).any():
            warnings.warn("Some values very close to 0, MSIS might not be estimated accurately.")
        return errors / (scale + 1e-8) / len(ground_truth_values)


def accumulate_forecast_score(
    ground_truth: TimeSeries,
    predict: TimeSeries,
    insample: TimeSeries = None,
    periodicity=1,
    ub: TimeSeries = None,
    lb: TimeSeries = None,
    metric=None,
) -> Union[ForecastScoreAccumulator, float]:
    acc = ForecastScoreAccumulator(
        ground_truth=ground_truth, predict=predict, insample=insample, periodicity=periodicity, ub=ub, lb=lb
    )
    return acc if metric is None else metric(acc)


class ForecastMetric(Enum):
    """
    Enumeration of evaluation metrics for time series forecasting. For each value,
    the name is the metric, and the value is a partial function of form
    ``f(ground_truth, predict, **kwargs)``. Here, ``ground_truth`` is the
    original time series, and ``predict`` is the result returned by a
    `ForecastEvaluator`.
    """

    MAE = partial(accumulate_forecast_score, metric=ForecastScoreAccumulator.mae)
    """
    Mean Absolute Error (MAE) is formulated as:

    ..  math:: 
        \\frac{1}{T}\\sum_{t=1}^T{(|y_t - \\hat{y}_t|)}.
    """
    MARRE = partial(accumulate_forecast_score, metric=ForecastScoreAccumulator.marre)
    """
    Mean Absolute Ranged Relative Error (MARRE) is formulated as:

    .. math:: 
        100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t
        - \\hat{y}_t} {\\max_t{y_t} - \\min_t{y_t}} \\right|}.
    """
    RMSE = partial(accumulate_forecast_score, metric=ForecastScoreAccumulator.rmse)
    """
    Root Mean Squared Error (RMSE) is formulated as:
    
    .. math::
        \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}}.
    """
    sMAPE = partial(accumulate_forecast_score, metric=ForecastScoreAccumulator.smape)
    """
    symmetric Mean Absolute Percentage Error (sMAPE) is formulated as:

    .. math::
        200 \\cdot \\frac{1}{T}\\sum_{t=1}^{T}{\\frac{\\left| y_t
        - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|}}.
    """
    MASE = partial(accumulate_forecast_score, metric=ForecastScoreAccumulator.mase)
    """
    Mean Absolute Scaled Error (MASE) is formulated as:

    .. math:: 
        \\frac{1}{T}\\cdot\\frac{\\sum_{t=1}^{T}\\left| y_t
          - \\hat{y}_t \\right|}{\\frac{1}{N-m}\\sum_{t=m+1}^{N}\\left| x_t - x_{t-m} \\right|}.
    """
    MSIS = partial(accumulate_forecast_score, metric=ForecastScoreAccumulator.msis)
    """
    Mean Scaled Interval Score (MSIS) is formulated as:

    .. math::
        \\frac{1}{T}\\cdot\\frac{\\sum_{t=1}^{T} (U_t - L_t) + 100 \\cdot (L_t - y_t)[y_t<L_t]
          + 100\\cdot(y_t - U_t)[y_t > U_t]}{\\frac{1}{N-m}\\sum_{t=m+1}^{N}\\left| x_t - x_{t-m} \\right|}.
    
    """


class ForecastEvaluatorConfig(EvaluatorConfig):
    """
    Configuration class for a `ForecastEvaluator`
    """

    _timedelta_keys = EvaluatorConfig._timedelta_keys + ["horizon"]

    def __init__(self, horizon: float = None, **kwargs):
        """
        :param horizon: the model's prediction horizon. Whenever the model makes
            a prediction, it will predict ``horizon`` seconds into the future.
        """
        super().__init__(**kwargs)
        self.horizon = horizon

    @property
    def horizon(self):
        """
        :return: the horizon (number of seconds) our model is predicting into
            the future. Defaults to the retraining frequency.
        """
        if self._horizon is None:
            return self.retrain_freq
        return self._horizon

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = granularity_str_to_seconds(horizon)

    @property
    def cadence(self):
        """
        :return: the cadence (interval, in number of seconds) at which we are
            having our model produce new predictions. Defaults to the predictive
            horizon if there is one, and the retraining frequency otherwise.
        """
        if self._cadence is None:
            return self.horizon
        return self._cadence

    @cadence.setter
    def cadence(self, cadence):
        self._cadence = granularity_str_to_seconds(cadence)


class ForecastEvaluator(EvaluatorBase):
    """
    Simulates the live deployment of an forecaster model.
    """

    config_class = ForecastEvaluatorConfig

    def __init__(self, model, config):
        assert isinstance(model, ForecasterBase)
        super().__init__(model=model, config=config)

    @property
    def horizon(self):
        return self.config.horizon

    @property
    def cadence(self):
        return self.config.cadence

    def _call_model(
        self, time_series: TimeSeries, time_series_prev: TimeSeries, return_err: bool = False
    ) -> Union[Tuple[TimeSeries, TimeSeries], TimeSeries]:
        if self.model.target_seq_index is not None:
            name = time_series.names[self.model.target_seq_index]
            time_stamps = time_series.univariates[name].time_stamps
        else:
            time_stamps = time_series.time_stamps
        forecast, err = self.model.forecast(time_stamps, time_series_prev)
        return (forecast, err) if return_err else forecast

    def evaluate(
        self,
        ground_truth: TimeSeries,
        predict: Union[TimeSeries, List[TimeSeries]],
        metric: ForecastMetric = ForecastMetric.sMAPE,
    ):
        """
        :param ground_truth: the series of test data
        :param predict: the series of predicted values
        :param metric: the evaluation metric.
        """
        if self.model.target_seq_index is not None:
            name = ground_truth.names[self.model.target_seq_index]
            ground_truth = ground_truth.univariates[name].to_ts()
        if isinstance(predict, TimeSeries):
            if metric is not None:
                return metric.value(ground_truth, predict)
            return accumulate_forecast_score(ground_truth, predict)
        else:
            if metric is not None:
                weights = np.asarray([len(p) for p in predict if not p.is_empty()])
                vals = [metric.value(ground_truth, p) for p in predict if not p.is_empty()]
                return np.dot(weights / weights.sum(), vals)
            return [accumulate_forecast_score(ground_truth, p) for p in predict if not p.is_empty()]
