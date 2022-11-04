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
                id="anomaly-select-file-parent",
                children=[
                    dbc.RadioItems(
                        id="anomaly-file-radio",
                        options=[
                            {"label": "Single data file", "value": "single"},
                            {"label": "Separate train/test files", "value": "separate"},
                        ],
                        value="single",
                        inline=True,
                    ),
                    dcc.Dropdown(id="anomaly-select-file", options=[], style={"width": "100%"}),
                ],
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Training Data Percentage"),
                        dcc.Slider(
                            id="anomaly-training-slider",
                            min=5,
                            max=95,
                            step=1,
                            marks={t * 10: str(t * 10) for t in range(1, 10)},
                            value=50,
                        ),
                    ],
                ),
                id="anomaly-slider-collapse",
                is_open=True,
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Select Test Data File"),
                        html.Div(
                            id="anomaly-select-test-file-parent",
                            children=[dcc.Dropdown(id="anomaly-select-test-file", options=[], style={"width": "100%"})],
                        ),
                    ],
                ),
                id="anomaly-test-file-collapse",
                is_open=False,
            ),
            html.Br(),
            html.P("Select Feature Column(s)"),
            html.Div(
                id="anomaly-select-features-parent",
                children=[dcc.Dropdown(id="anomaly-select-features", options=[], multi=True, style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Label Column for Evaluation (Optional)"),
            html.Div(
                id="anomaly-select-label-parent",
                children=[dcc.Dropdown(id="anomaly-select-label", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Anomaly Detection Algorithm"),
            html.Div(
                id="anomaly-select-algorithm-parent",
                children=[dcc.Dropdown(id="anomaly-select-algorithm", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Algorithm Setting"),
            html.Div(id="anomaly-param-table", children=[create_param_table()]),
            html.Progress(id="anomaly-progressbar", style={"width": "100%"}),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="anomaly-train-btn", children="Train", n_clicks=0),
                    html.Button(id="anomaly-cancel-btn", children="Cancel", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            html.P("Threshold Setting"),
            html.Div(
                id="anomaly-select-threshold-parent",
                children=[dcc.Dropdown(id="anomaly-select-threshold", options=[], style={"width": "100%"})],
            ),
            html.Div(id="anomaly-threshold-param-table", children=[create_param_table(height=80)]),
            html.Div(
                children=[html.Button(id="anomaly-test-btn", children="Update Threshold", n_clicks=0)],
                style={"textAlign": "center"},
            ),
            create_modal(
                modal_id="anomaly-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="anomaly-exception-modal-content",
                button_id="anomaly-exception-modal-close",
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
                    html.B("Anomaly Detection Results"),
                    html.Hr(),
                    html.Div(id="anomaly-plots", children=[create_empty_figure()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Testing Metrics"),
                    html.Hr(),
                    html.Div(id="anomaly-test-metrics", children=[create_metric_table()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Training Metrics"),
                    html.Hr(),
                    html.Div(id="anomaly-training-metrics", children=[create_metric_table()]),
                ],
            ),
        ],
    )


def create_anomaly_layout() -> html.Div:
    return html.Div(
        id="anomaly_views",
        children=[
            # Left column
            html.Div(id="left-column-data", className="three columns", children=[create_control_panel()]),
            # Right column
            html.Div(className="nine columns", children=create_right_column()),
        ],
    )
