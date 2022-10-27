#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html


tab_style = {"borderBottom": "1px solid #d6d6d6", "padding": "6px", "fontWeight": "bold"}

tab_selected_style = {
    "borderTop": "1px solid #d6d6d6",
    "borderBottom": "1px solid #d6d6d6",
    "backgroundColor": "#119DFF",
    "color": "white",
    "padding": "6px",
    "fontWeight": "bold",
}


def create_banner(app):
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Img(src=app.get_asset_url("merlion_small.svg")),
            html.Plaintext("  Powered by Salesforce AI Research"),
        ],
    )


def create_layout() -> html.Div:
    children, values = [], []
    # Data analysis tab
    children.append(
        dcc.Tab(label="File Manager", value="file-manager", style=tab_style, selected_style=tab_selected_style)
    )
    values.append("file-manager")
    # Forecasting tab
    children.append(
        dcc.Tab(label="Forecasting", value="forecasting", style=tab_style, selected_style=tab_selected_style)
    )
    values.append("forecasting")
    # Anomaly detection tab
    children.append(
        dcc.Tab(label="Anomaly Detection", value="anomaly", style=tab_style, selected_style=tab_selected_style)
    )
    values.append("anomaly")

    layout = html.Div(
        id="app-content",
        children=[dcc.Tabs(id="tabs", value=values[0] if values else "none", children=children), html.Div(id="plots")],
    )
    return layout
