#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import os
import traceback

import dash
from dash import Input, Output, State, callback, dcc
from merlion.dashboard.utils.file_manager import FileManager
from merlion.dashboard.models.anomaly import AnomalyModel
from merlion.dashboard.pages.utils import create_param_table, create_metric_table, create_empty_figure

logger = logging.getLogger(__name__)
file_manager = FileManager()


@callback(
    Output("anomaly-select-file", "options"),
    Output("anomaly-select-features", "value"),
    Output("anomaly-select-label", "value"),
    Input("anomaly-select-file-parent", "n_clicks"),
    Input("anomaly-select-file", "value"),
    [State("anomaly-select-features", "value"), State("anomaly-select-label", "value")],
)
def update_select_file_dropdown(n_clicks, filename, features, label):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_ids = {p["prop_id"].split(".")[0]: p["value"] for p in ctx.triggered}
        if "anomaly-select-file-parent" in prop_ids:
            files = file_manager.uploaded_files()
            for f in files:
                options.append({"label": f, "value": f})
        if "anomaly-select-file" in prop_ids:
            features, label = None, None
    return options, features, label


@callback(Output("anomaly-select-test-file", "options"), Input("anomaly-select-test-file-parent", "n_clicks"))
def update_select_test_file_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "anomaly-select-test-file-parent":
        files = file_manager.uploaded_files()
        for filename in files:
            options.append({"label": filename, "value": filename})
    return options


@callback(
    Output("anomaly-select-features", "options"),
    Input("anomaly-select-features-parent", "n_clicks"),
    [
        State("anomaly-select-file", "value"),
        State("anomaly-select-test-file", "value"),
        State("anomaly-select-label", "value"),
    ],
)
def select_features(n_clicks, train_file, test_file, label_name):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "anomaly-select-features-parent":
        file_path = None
        if train_file:
            file_path = os.path.join(file_manager.data_directory, train_file)
        elif test_file:
            file_path = os.path.join(file_manager.data_directory, test_file)
        if file_path:
            df = AnomalyModel().load_data(file_path, nrows=2)
            options += [{"label": s, "value": s} for s in df.columns if s != label_name]
    return options


@callback(
    Output("anomaly-select-label", "options"),
    Input("anomaly-select-label-parent", "n_clicks"),
    [
        State("anomaly-select-file", "value"),
        State("anomaly-select-test-file", "value"),
        State("anomaly-select-features", "value"),
    ],
)
def select_label(n_clicks, train_file, test_file, features):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "anomaly-select-label-parent":
        file_path = None
        if train_file:
            file_path = os.path.join(file_manager.data_directory, train_file)
        elif test_file:
            file_path = os.path.join(file_manager.data_directory, test_file)
        if file_path:
            df = AnomalyModel().load_data(file_path, nrows=2)
            options += [{"label": s, "value": s} for s in df.columns if s not in (features or [])]
    return options


@callback(
    Output("anomaly-select-algorithm", "options"),
    Input("anomaly-select-algorithm-parent", "n_clicks"),
    State("anomaly-select-features", "value"),
    prevent_initial_call=True,
)
def select_algorithm_parent(n_clicks, selected_metrics):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "anomaly-select-algorithm-parent":
        algorithms = AnomalyModel.get_available_algorithms(len(selected_metrics))
        options += [{"label": s, "value": s} for s in algorithms]
    return options


@callback(Output("anomaly-param-table", "children"), Input("anomaly-select-algorithm", "value"))
def select_algorithm(algorithm):
    param_table = create_param_table()
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "anomaly-select-algorithm":
        param_info = AnomalyModel.get_parameter_info(algorithm)
        param_table = create_param_table(param_info)
    return param_table


@callback(Output("anomaly-select-threshold", "options"), Input("anomaly-select-threshold-parent", "n_clicks"))
def select_threshold_parent(n_clicks):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "anomaly-select-threshold-parent":
        algorithms = AnomalyModel.get_available_thresholds()
        options += [{"label": s, "value": s} for s in algorithms]
    return options


@callback(Output("anomaly-threshold-param-table", "children"), Input("anomaly-select-threshold", "value"))
def select_threshold(threshold):
    param_table = create_param_table(height=80)
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
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
        State("anomaly-select-features", "value"),
        State("anomaly-select-algorithm", "value"),
        State("anomaly-select-label", "value"),
        State("anomaly-param-table", "children"),
        State("anomaly-select-threshold", "value"),
        State("anomaly-threshold-param-table", "children"),
        State("anomaly-training-slider", "value"),
        State("anomaly-training-metrics", "children"),
        State("anomaly-file-radio", "value"),
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
def click_train_test(
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
    train_percentage,
    train_metrics,
    file_mode,
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    train_metric_table = create_metric_table()
    test_metric_table = create_metric_table()
    figure = create_empty_figure()
    set_progress((str(0), str(10)))

    try:
        if ctx.triggered:
            prop_id = ctx.triggered_id
            if prop_id == "anomaly-train-btn" and train_clicks > 0:
                assert train_filename, "The training file is empty!"
                assert columns, "Please select variables/metrics for analysis."
                assert algorithm, "Please select a anomaly detector to train."

                df = AnomalyModel().load_data(os.path.join(file_manager.data_directory, train_filename))
                if file_mode == "single":
                    n = int(int(train_percentage) * len(df) / 100)
                    train_df = df.iloc[:n]
                    test_df = df.iloc[n:]
                else:
                    assert test_filename, "The test file is empty!"
                    train_df = df
                    test_df = AnomalyModel().load_data(os.path.join(file_manager.data_directory, test_filename))

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

                model, train_metrics, test_metrics, figure = AnomalyModel().train(
                    algorithm, train_df, test_df, columns, label_column, alg_params, threshold_params, set_progress
                )
                AnomalyModel.save_model(file_manager.model_directory, model, algorithm)
                if train_metrics is not None:
                    train_metric_table = create_metric_table(train_metrics)
                if test_metrics is not None:
                    test_metric_table = create_metric_table(test_metrics)
                figure = dcc.Graph(figure=figure)

            elif prop_id == "anomaly-test-btn" and test_clicks > 0:
                assert columns, "Please select variables/metrics for analysis."
                assert algorithm, "Please select a trained anomaly detector."

                if file_mode == "single":
                    df = AnomalyModel().load_data(os.path.join(file_manager.data_directory, train_filename))
                    n = int(int(train_percentage) * len(df) / 100)
                    df = df.iloc[n:]
                else:
                    assert test_filename, "The test file is empty!"
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

                train_metrics = train_metrics[0] if isinstance(train_metrics, list) else train_metrics
                train_metric_table = create_metric_table(train_metrics["props"]["data"][0])
                metrics, figure = AnomalyModel().test(model, df, columns, label_column, threshold_params, set_progress)
                if metrics is not None:
                    test_metric_table = create_metric_table(metrics)
                figure = dcc.Graph(figure=figure)

    except Exception:
        error = traceback.format_exc()
        modal_is_open = True
        modal_content = error
        logger.error(error)

    return train_metric_table, test_metric_table, figure, modal_is_open, modal_content


@callback(
    Output("anomaly-slider-collapse", "is_open"),
    Output("anomaly-test-file-collapse", "is_open"),
    Input("anomaly-file-radio", "value"),
)
def set_file_mode(value):
    if value == "single":
        return True, False
    else:
        return False, True
