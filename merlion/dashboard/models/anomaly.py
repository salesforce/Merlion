#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
import logging
import inspect
import importlib
from collections import OrderedDict
from merlion.models.factory import ModelFactory
from merlion.evaluate.anomaly import TSADMetric
from merlion.utils.time_series import TimeSeries
from merlion.plot import MTSFigure

from merlion.dashboard.models.utils import ModelMixin, DataMixin
from merlion.dashboard.utils.log import DashLogger

dash_logger = DashLogger(stream=sys.stdout)


class AnomalyModel(ModelMixin, DataMixin):
    univariate_algorithms = [
        "DefaultDetector",
        "ArimaDetector",
        "DynamicBaseline",
        "IsolationForest",
        "ETSDetector",
        "LSTMDetector",
        "MSESDetector",
        "ProphetDetector",
        "RandomCutForest",
        "SarimaDetector",
        "WindStats",
        "SpectralResidual",
        "ZMS",
        "DeepPointAnomalyDetector",
    ]
    multivariate_algorithms = ["IsolationForest", "AutoEncoder", "VAE", "DAGMM", "LSTMED"]
    thresholds = ["Threshold", "AggregateAlarms"]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)

    @staticmethod
    def get_available_algorithms(num_input_metrics):
        if num_input_metrics <= 0:
            return []
        elif num_input_metrics == 1:
            return AnomalyModel.univariate_algorithms
        else:
            return AnomalyModel.multivariate_algorithms

    @staticmethod
    def get_available_thresholds():
        return AnomalyModel.thresholds

    @staticmethod
    def _param_info(function):
        param_info = OrderedDict()
        valid_types = [int, float, str, bool, list, tuple, dict]

        signature = inspect.signature(function).parameters
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
        return param_info

    @staticmethod
    def get_parameter_info(algorithm):
        model_class = ModelFactory.get_model_class(algorithm)
        param_info = AnomalyModel._param_info(model_class.config_class.__init__)
        if "max_forecast_steps" in param_info:
            if not param_info["max_forecast_steps"]["default"]:
                param_info["max_forecast_steps"]["default"] = 5
        if "max_forecast_steps" in param_info:
            if algorithm in ["ArimaDetector", "SarimaDetector"]:
                del param_info["max_forecast_steps"]
            elif not param_info["max_forecast_steps"]["default"]:
                param_info["max_forecast_steps"]["default"] = 5
        return param_info

    @staticmethod
    def get_threshold_info(threshold):
        module = importlib.import_module("merlion.post_process.threshold")
        model_class = getattr(module, threshold)
        param_info = AnomalyModel._param_info(model_class.__init__)
        if not param_info["alm_threshold"]["default"]:
            param_info["alm_threshold"]["default"] = 3.0
        return param_info

    @staticmethod
    def _compute_metrics(labels, predictions):
        metrics = {}
        for metric_name, metric in [
            ("Precision", TSADMetric.Precision),
            ("Recall", TSADMetric.Recall),
            ("F1", TSADMetric.F1),
            ("MeanTimeToDetect", TSADMetric.MeanTimeToDetect),
        ]:
            m = metric.value(ground_truth=labels, predict=predictions)
            metrics[metric_name] = round(m, 5) if metric_name != "MeanTimeToDetect" else str(m)
        return metrics

    @staticmethod
    def _plot_anomalies(model, ts, scores):
        title = f"{type(model).__name__}: Anomalies in Time Series"
        fig = MTSFigure(y=ts, y_prev=None, anom=scores)
        return fig.plot_plotly(title=title)

    @staticmethod
    def _check(df, columns, label_column):
        if label_column and label_column not in df:
            label_column = int(label_column)
            assert label_column in df, f"The label column {label_column} is not in the time series."
        for i in range(len(columns)):
            if columns[i] not in df:
                columns[i] = int(columns[i])
            assert columns[i] in df, f"The variable {columns[i]} is not in the time series."
        return columns, label_column

    def train(self, algorithm, df, columns, label_column, params, threshold_params, set_progress):
        columns, label_column = AnomalyModel._check(df, columns, label_column)

        if threshold_params is not None:
            thres_class, thres_params = threshold_params
            module = importlib.import_module("merlion.post_process.threshold")
            model_class = getattr(module, thres_class)
            params["threshold"] = model_class(**thres_params)

        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class(**params))
        train_ts, label_ts = TimeSeries.from_pd(df[columns]), None
        if label_column is not None and label_column != "":
            label_ts = TimeSeries.from_pd(df[[label_column]])

        self.logger.info(f"Training the anomaly detector: {algorithm}...")
        set_progress(("2", "10"))

        scores = model.train(train_data=train_ts)
        set_progress(("7", "10"))

        self.logger.info("Computing training performance metrics...")
        predictions = model.post_rule(scores) if model.post_rule is not None else scores
        metrics = AnomalyModel._compute_metrics(label_ts, predictions) if label_ts is not None else None
        set_progress(("8", "10"))

        self.logger.info("Plotting anomaly scores...")
        figure = AnomalyModel._plot_anomalies(model, train_ts, scores)
        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return model, metrics, figure

    def test(self, model, df, columns, label_column, threshold_params, set_progress):
        columns, label_column = AnomalyModel._check(df, columns, label_column)

        threshold = None
        if threshold_params is not None:
            thres_class, thres_params = threshold_params
            module = importlib.import_module("merlion.post_process.threshold")
            model_class = getattr(module, thres_class)
            threshold = model_class(**thres_params)
        if threshold is not None:
            model.threshold = threshold

        self.logger.info(f"Detecting anomalies...")
        set_progress(("2", "10"))

        test_ts, label_ts = TimeSeries.from_pd(df[columns]), None
        if label_column is not None and label_column != "":
            label_ts = TimeSeries.from_pd(df[[label_column]])
        predictions = model.get_anomaly_label(time_series=test_ts)
        set_progress(("7", "10"))

        self.logger.info("Computing test performance metrics...")
        metrics = AnomalyModel._compute_metrics(label_ts, predictions) if label_ts is not None else None
        set_progress(("8", "10"))

        self.logger.info("Plotting anomaly labels...")
        figure = AnomalyModel._plot_anomalies(model, test_ts, predictions)
        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return metrics, figure
