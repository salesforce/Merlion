#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
import logging
import inspect
from collections import OrderedDict
from merlion.models.factory import ModelFactory
from merlion.evaluate.forecast import ForecastEvaluator, ForecastMetric
from merlion.utils.time_series import TimeSeries

from .utils import ModelMixin, DataMixin
from ..utils.log import DashLogger

dash_logger = DashLogger(stream=sys.stdout)


class ForecastModel(ModelMixin, DataMixin):
    algorithms = [
        "DefaultForecaster",
        "Arima",
        "ETS",
        "AutoETS",
        "LSTM",
        "Prophet",
        "AutoProphet",
        "Sarima",
        "VectorAR",
        "RandomForestForecaster",
        "ExtraTreesForecaster",
        "LGBMForecaster"
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)

    @staticmethod
    def get_available_algorithms():
        return ForecastModel.algorithms

    @staticmethod
    def get_parameter_info(algorithm):
        param_info = OrderedDict()
        valid_types = [int, float, str, bool, list, tuple, dict]

        model_class = ModelFactory.get_model_class(algorithm)
        signature = inspect.signature(model_class.config_class.__init__).parameters
        for name, param in signature.items():
            if name in ["self", "target_seq_index"]:
                continue
            value = param.default
            if value == param.empty:
                value = ""
            if type(param.default) in valid_types:
                param_info[name] = {"type": type(param.default), "default": value}
            elif param.annotation in valid_types:
                param_info[name] = {"type": param.annotation, "default": value}

        if "max_forecast_steps" in param_info:
            if not param_info["max_forecast_steps"]["default"]:
                param_info["max_forecast_steps"]["default"] = 100
        return param_info

    @staticmethod
    def _compute_metrics(evaluator, ts, predictions):
        metrics = {}
        for metric_name, metric in [
            ("MAE", ForecastMetric.MAE),
            ("MARRE", ForecastMetric.MAE),
            ("RMSE", ForecastMetric.RMSE),
            ("sMAPE", ForecastMetric.sMAPE),
            ("RMSPE", ForecastMetric.RMSPE)
        ]:
            m = evaluator.evaluate(ground_truth=ts, predict=predictions, metric=metric)
            metrics[metric_name] = round(m, 5)
        return metrics

    def train(self, algorithm, train_df, test_df, target_column, params, set_progress):
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."

        columns = list(train_df.columns)
        params["target_seq_index"] = columns.index(target_column)
        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class(**params))

        self.logger.info(f"Training the forecasting model: {algorithm}...")
        set_progress(("2", "10"))

        train_ts = TimeSeries.from_pd(train_df)
        predictions = model.train(train_ts)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        self.logger.info("Computing training performance metrics...")
        set_progress(("6", "10"))

        evaluator = ForecastEvaluator(model, config=ForecastEvaluator.config_class())
        train_metrics = ForecastModel._compute_metrics(evaluator, train_ts, predictions)
        set_progress(("7", "10"))

        test_ts = TimeSeries.from_pd(test_df)
        if "max_forecast_steps" in params:
            n = min(len(test_ts) - 1, int(params["max_forecast_steps"]))
            test_ts, _ = test_ts.bisect(t=test_ts.time_stamps[n])

        self.logger.info("Computing test performance metrics...")
        test_pred, test_err = model.forecast(time_stamps=test_ts.time_stamps)
        test_metrics = ForecastModel._compute_metrics(evaluator, test_ts, test_pred)
        set_progress(("8", "10"))

        self.logger.info("Plotting forecasting results...")
        figure = model.plot_forecast_plotly(
            time_series=test_ts,
            time_series_prev=train_ts,
            plot_forecast_uncertainty=True
        )
        figure.update_layout(width=None, height=500)
        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return model, train_metrics, test_metrics, figure
