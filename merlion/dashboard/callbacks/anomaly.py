#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import dash
from dash import Input, Output, State, callback, dcc
from merlion.dashboard.utils.file_manager import FileManager
from merlion.dashboard.models.anomaly import AnomalyModel
from merlion.dashboard.pages.utils import create_param_table, create_metric_table, create_emtpy_figure

file_manager = FileManager()


@callback(Output("anomaly-select-file", "options"), Input("anomaly-select-file-parent", "n_clicks"))
def update_select_file_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-file-parent":
            files = file_manager.uploaded_files()
            for filename in files:
                options.append({"label": filename, "value": filename})
    return options


@callback(Output("anomaly-select-test-file", "options"), Input("anomaly-select-test-file-parent", "n_clicks"))
def update_select_test_file_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-test-file-parent":
            files = file_manager.uploaded_files()
            for filename in files:
                options.append({"label": filename, "value": filename})
    return options


@callback(
    Output("anomaly-select-metric", "options"),
    Input("anomaly-select-metric-parent", "n_clicks"),
    [State("anomaly-select-file", "value"), State("anomaly-select-test-file", "value")],
)
def select_metric(n_clicks, train_file, test_file):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-metric-parent":
            file_path = None
            if train_file:
                file_path = os.path.join(file_manager.data_directory, train_file)
            elif test_file:
                file_path = os.path.join(file_manager.data_directory, test_file)
            if file_path:
                df = AnomalyModel().load_data(file_path, nrows=2)
                options += [{"label": s, "value": s} for s in df.columns]
    return options


@callback(
    Output("anomaly-select-label", "options"),
    Input("anomaly-select-label-parent", "n_clicks"),
    [State("anomaly-select-file", "value"), State("anomaly-select-test-file", "value")],
)
def select_metric(n_clicks, train_file, test_file):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-label-parent":
            file_path = None
            if train_file:
                file_path = os.path.join(file_manager.data_directory, train_file)
            elif test_file:
                file_path = os.path.join(file_manager.data_directory, test_file)
            if file_path:
                df = AnomalyModel().load_data(file_path, nrows=2)
                options += [{"label": s, "value": s} for s in df.columns]
    return options


@callback(
    Output("anomaly-select-algorithm", "options"),
    Input("anomaly-select-algorithm-parent", "n_clicks"),
    State("anomaly-select-metric", "value"),
)
def select_algorithm_parent(n_clicks, selected_metrics):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-algorithm-parent":
            algorithms = AnomalyModel.get_available_algorithms(len(selected_metrics))
            options += [{"label": s, "value": s} for s in algorithms]
    return options


@callback(Output("anomaly-param-table", "children"), Input("anomaly-select-algorithm", "value"))
def select_algorithm(algorithm):
    param_table = create_param_table()
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-algorithm":
            param_info = AnomalyModel.get_parameter_info(algorithm)
            param_table = create_param_table(param_info)
    return param_table


@callback(Output("anomaly-select-threshold", "options"), Input("anomaly-select-threshold-parent", "n_clicks"))
def select_threshold_parent(n_clicks):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-threshold-parent":
            algorithms = AnomalyModel.get_available_thresholds()
            options += [{"label": s, "value": s} for s in algorithms]
    return options


@callback(Output("anomaly-threshold-param-table", "children"), Input("anomaly-select-threshold", "value"))
def select_threshold(threshold):
    param_table = create_param_table(height=80)
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-select-threshold" and threshold:
            param_info = AnomalyModel.get_threshold_info(threshold)
            param_table = create_param_table(param_info, height=80)
    return param_table


@callback(
    Output("anomaly-training-metrics", "children"),
    Output("anomaly-test-metrics", "children"),
    Output("anomaly-plots", "children"),
    Output("anomaly-exception-modal", "is_open"),
    Output("anomaly-exception-modal-content", "children"),
    [
        Input("anomaly-train-btn", "n_clicks"),
        Input("anomaly-test-btn", "n_clicks"),
        Input("anomaly-exception-modal-close", "n_clicks"),
    ],
    [
        State("anomaly-select-file", "value"),
        State("anomaly-select-test-file", "value"),
        State("anomaly-select-metric", "value"),
        State("anomaly-select-algorithm", "value"),
        State("anomaly-select-label", "value"),
        State("anomaly-param-table", "children"),
        State("anomaly-select-threshold", "value"),
        State("anomaly-threshold-param-table", "children"),
    ],
    running=[
        (Output("anomaly-train-btn", "disabled"), True, False),
        (Output("anomaly-test-btn", "disabled"), True, False),
        (Output("anomaly-cancel-btn", "disabled"), False, True),
    ],
    cancel=[Input("anomaly-cancel-btn", "n_clicks")],
    background=True,
    manager=file_manager.get_long_callback_manager(),
    progress=[Output("anomaly-progressbar", "value"), Output("anomaly-progressbar", "max")],
)
def click_train(
    set_progress,
    train_clicks,
    test_clicks,
    modal_close,
    train_filename,
    test_filename,
    columns,
    algorithm,
    label_column,
    param_table,
    threshold_class,
    threshold_table,
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    train_metric_table = create_metric_table()
    test_metric_table = create_metric_table()
    figure = create_emtpy_figure()
    set_progress((str(0), str(10)))

    try:
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "anomaly-train-btn" and train_clicks > 0:
                assert train_filename, "The training file is empty!"
                assert columns, "Please select variables/metrics for analysis."
                assert algorithm, "Please select a anomaly detector to train."

                df = AnomalyModel().load_data(os.path.join(file_manager.data_directory, train_filename))
                alg_params = AnomalyModel.parse_parameters(
                    param_info=AnomalyModel.get_parameter_info(algorithm),
                    params={p["Parameter"]: p["Value"] for p in param_table["props"]["data"]},
                )
                if threshold_class:
                    threshold_params = (
                        threshold_class,
                        AnomalyModel.parse_parameters(
                            param_info=AnomalyModel.get_threshold_info(threshold_class),
                            params={p["Parameter"]: p["Value"] for p in threshold_table["props"]["data"]},
                        ),
                    )
                else:
                    threshold_params = None

                model, metrics, figure = AnomalyModel().train(
                    algorithm, df, columns, label_column, alg_params, threshold_params, set_progress
                )
                AnomalyModel.save_model(file_manager.model_directory, model, algorithm)
                if metrics is not None:
                    train_metric_table = create_metric_table(metrics)
                figure = dcc.Graph(figure=figure)

            elif prop_id == "anomaly-test-btn" and test_clicks > 0:
                assert test_filename, "The test file is empty!"
                assert columns, "Please select variables/metrics for analysis."
                assert algorithm, "Please select a trained anomaly detector."

                df = AnomalyModel().load_data(os.path.join(file_manager.data_directory, test_filename))
                model = AnomalyModel.load_model(file_manager.model_directory, algorithm)
                if threshold_class:
                    threshold_params = (
                        threshold_class,
                        AnomalyModel.parse_parameters(
                            param_info=AnomalyModel.get_threshold_info(threshold_class),
                            params={p["Parameter"]: p["Value"] for p in threshold_table["props"]["data"]},
                        ),
                    )
                else:
                    threshold_params = None

                metrics, figure = AnomalyModel().test(model, df, columns, label_column, threshold_params, set_progress)
                if metrics is not None:
                    test_metric_table = create_metric_table(metrics)
                figure = dcc.Graph(figure=figure)

    except Exception as error:
        modal_is_open = True
        modal_content = str(error)

    return train_metric_table, test_metric_table, figure, modal_is_open, modal_content
