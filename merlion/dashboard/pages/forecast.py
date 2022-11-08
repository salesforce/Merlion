#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from merlion.dashboard.pages.utils import create_modal, create_param_table, create_metric_table, create_empty_figure


def create_control_panel() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select Training Data File"),
            html.Div(
                id="forecasting-select-file-parent",
                children=[
                    dbc.RadioItems(
                        id="forecasting-file-radio",
                        options=[
                            {"label": "Single data file", "value": "single"},
                            {"label": "Separate train/test files", "value": "separate"},
                        ],
                        value="single",
                        inline=True,
                    ),
                    dcc.Dropdown(id="forecasting-select-file", options=[], style={"width": "100%"}),
                ],
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Training Data Percentage"),
                        dcc.Slider(
                            id="forecasting-training-slider",
                            min=5,
                            max=95,
                            step=1,
                            marks={t * 10: str(t * 10) for t in range(1, 10)},
                            value=80,
                        ),
                    ],
                ),
                id="forecasting-slider-collapse",
                is_open=True,
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Select Test Data File"),
                        html.Div(
                            id="forecasting-select-test-file-parent",
                            children=[
                                dcc.Dropdown(id="forecasting-select-test-file", options=[], style={"width": "100%"})
                            ],
                        ),
                    ],
                ),
                id="forecasting-test-file-collapse",
                is_open=False,
            ),
            html.Br(),
            html.P("Select Target Column"),
            html.Div(
                id="forecasting-select-target-parent",
                children=[dcc.Dropdown(id="forecasting-select-target", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Other Features (Optional)"),
            html.Div(
                id="forecasting-select-features-parent",
                children=[
                    dcc.Dropdown(id="forecasting-select-features", options=[], multi=True, style={"width": "100%"})
                ],
            ),
            html.Br(),
            html.P("Select Exogenous Variables (Optional; Known A Priori)"),
            html.Div(
                id="forecasting-select-exog-parent",
                children=[dcc.Dropdown(id="forecasting-select-exog", options=[], multi=True, style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Forecasting Algorithm"),
            html.Div(
                id="forecasting-select-algorithm-parent",
                children=[dcc.Dropdown(id="forecasting-select-algorithm", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Algorithm Setting"),
            html.Div(id="forecasting-param-table", children=[create_param_table()]),
            html.Progress(id="forecasting-progressbar", style={"width": "100%"}),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="forecasting-train-btn", children="Train", n_clicks=0),
                    html.Button(id="forecasting-cancel-btn", children="Cancel", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),
            create_modal(
                modal_id="forecasting-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="forecasting-exception-modal-content",
                button_id="forecasting-exception-modal-close",
            ),
        ],
    )


def create_right_column() -> html.Div:
    return html.Div(
        id="right-column-data",
        children=[
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Forecasting Results"),
                    html.Hr(),
                    html.Div(id="forecasting-plots", children=[create_empty_figure()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Testing Metrics"),
                    html.Hr(),
                    html.Div(id="forecasting-test-metrics", children=[create_metric_table()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Training Metrics"),
                    html.Hr(),
                    html.Div(id="forecasting-training-metrics", children=[create_metric_table()]),
                ],
            ),
        ],
    )


def create_forecasting_layout() -> html.Div:
    return html.Div(
        id="forecasting_views",
        children=[
            # Left column
            html.Div(id="left-column-data", className="three columns", children=[create_control_panel()]),
            # Right column
            html.Div(className="nine columns", children=create_right_column()),
        ],
    )
