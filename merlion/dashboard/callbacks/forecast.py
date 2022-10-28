#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import os
import traceback
import dash
from dash import Input, Output, State, dcc, callback
from merlion.dashboard.utils.file_manager import FileManager
from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.pages.utils import create_param_table, create_metric_table, create_empty_figure

logger = logging.getLogger(__name__)
file_manager = FileManager()


@callback(Output("forecasting-select-file", "options"), Input("forecasting-select-file-parent", "n_clicks"))
def update_select_file_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "forecasting-select-file-parent":
            files = file_manager.uploaded_files()
            for filename in files:
                options.append({"label": filename, "value": filename})
    return options


@callback(
    Output("forecasting-select-target", "options"),
    Input("forecasting-select-target-parent", "n_clicks"),
    State("forecasting-select-file", "value"),
)
def select_target(n_clicks, filename):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "forecasting-select-target-parent":
            if filename is not None:
                file_path = os.path.join(file_manager.data_directory, filename)
                df = ForecastModel().load_data(file_path, nrows=2)
                options += [{"label": s, "value": s} for s in df.columns]
    return options


@callback(
    Output("forecasting-select-exog", "options"),
    Input("forecasting-select-exog-parent", "n_clicks"),
    [State("forecasting-select-file", "value"), State("forecasting-select-target", "value")],
)
def select_exog(n_clicks, filename, target_name):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "forecasting-select-exog-parent":
            if filename is not None and target_name is not None:
                file_path = os.path.join(file_manager.data_directory, filename)
                df = ForecastModel().load_data(file_path, nrows=2)
                options += [{"label": s, "value": s} for s in df.columns if s != target_name]
    return options


@callback(
    Output("forecasting-select-algorithm", "options"),
    Input("forecasting-select-algorithm-parent", "n_clicks"),
    [State("forecasting-select-target", "value")],
)
def select_algorithm_parent(n_clicks, selected_target):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "forecasting-select-algorithm-parent":
            algorithms = ForecastModel.get_available_algorithms()
            options += [{"label": s, "value": s} for s in algorithms]
    return options


@callback(Output("forecasting-param-table", "children"), Input("forecasting-select-algorithm", "value"))
def select_algorithm(algorithm):
    param_table = create_param_table()
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "forecasting-select-algorithm":
            param_info = ForecastModel.get_parameter_info(algorithm)
            param_table = create_param_table(param_info)
    return param_table


@callback(
    Output("forecasting-training-metrics", "children"),
    Output("forecasting-test-metrics", "children"),
    Output("forecasting-plots", "children"),
    Output("forecasting-exception-modal", "is_open"),
    Output("forecasting-exception-modal-content", "children"),
    [Input("forecasting-train-btn", "n_clicks"), Input("forecasting-exception-modal-close", "n_clicks")],
    [
        State("forecasting-select-file", "value"),
        State("forecasting-select-target", "value"),
        State("forecasting-select-exog", "value"),
        State("forecasting-select-algorithm", "value"),
        State("forecasting-param-table", "children"),
        State("forecasting-training-slider", "value"),
    ],
    running=[
        (Output("forecasting-train-btn", "disabled"), True, False),
        (Output("forecasting-cancel-btn", "disabled"), False, True),
    ],
    cancel=[Input("forecasting-cancel-btn", "n_clicks")],
    background=True,
    manager=file_manager.get_long_callback_manager(),
    progress=[Output("forecasting-progressbar", "value"), Output("forecasting-progressbar", "max")],
)
def click_train_test(
    set_progress, n_clicks, modal_close, filename, target_col, exog_cols, algorithm, table, train_percentage
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    train_metric_table = create_metric_table()
    test_metric_table = create_metric_table()
    figure = create_empty_figure()
    set_progress(("0", "10"))

    try:
        if ctx.triggered and n_clicks > 0:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "forecasting-train-btn":
                assert filename, "Data file is empty!"
                assert target_col, "Please select a target variable/metric for forecasting."
                assert algorithm, "Please select a forecasting algorithm."
                exog_cols = exog_cols or []

                df = ForecastModel().load_data(os.path.join(file_manager.data_directory, filename))
                assert len(df) > 20, f"The input time series length ({len(df)}) is too small."
                n = int(int(train_percentage) * len(df) / 100)
                train_df, test_df = df.iloc[:n], df.iloc[n:]

                params = ForecastModel.parse_parameters(
                    param_info=ForecastModel.get_parameter_info(algorithm),
                    params={p["Parameter"]: p["Value"] for p in table["props"]["data"] if p["Parameter"]},
                )
                model, train_metrics, test_metrics, figure = ForecastModel().train(
                    algorithm, train_df, test_df, target_col, exog_cols, params, set_progress
                )
                ForecastModel.save_model(file_manager.model_directory, model, algorithm)
                train_metric_table = create_metric_table(train_metrics)
                test_metric_table = create_metric_table(test_metrics)
                figure = dcc.Graph(figure=figure)

    except Exception as error:
        modal_is_open = True
        modal_content = str(error)
        logger.error(traceback.format_exc())

    return train_metric_table, test_metric_table, figure, modal_is_open, modal_content
