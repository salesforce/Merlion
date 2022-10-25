#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash_bootstrap_components as dbc
from dash import html, dash_table
from ..settings import *

styles = {
    'json-output': {
        'overflow-y': 'scroll',
        'height': 'calc(90% - 25px)',
        'border': 'thin lightgrey solid'
    },
    'tab': {'height': 'calc(98vh - 80px)'},
    'log-output': {
            'overflow-y': 'scroll',
            'height': 'calc(90% - 25px)',
            'border': 'thin lightgrey solid',
            'white-space': 'pre-wrap'
        },
}


def create_modal(modal_id, header, content, content_id, button_id):
    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(header)),
                    dbc.ModalBody(content, id=content_id),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id=button_id, className="ml-auto", n_clicks=0
                        )
                    ),
                ],
                id=modal_id,
                is_open=False,
            ),
        ]
    )
    return modal


def create_param_table(params=None, height=100):
    if params is None or len(params) == 0:
        data = [{"Parameter": "", "Value": ""}]
    else:
        data = [{"Parameter": key, "Value": str(value["default"])}
                for key, value in params.items()]

    table = dash_table.DataTable(
        data=data,
        columns=[
            {"id": "Parameter", "name": "Parameter"},
            {"id": "Value", "name": "Value"}
        ],
        editable=True,
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_table={
            "overflowX": "scroll",
            "overflowY": "scroll",
            "height": height
        },
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR)
    )
    return table


def create_metric_table(metrics=None):
    if metrics is None or len(metrics) == 0:
        data, columns = {}, []
        for i in range(4):
            data[f"Metric {i}"] = "-"
            columns.append({"id": f"Metric {i}", "name": f"Metric {i}"})

    else:
        data = metrics
        columns = [{"id": key, "name": key} for key in metrics.keys()]

    if not isinstance(data, list):
        data = [data]
    table = dash_table.DataTable(
        data=data,
        columns=columns,
        editable=False,
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_table={"overflowX": "scroll"},
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR)
    )
    return table


def create_emtpy_figure():
    import numpy as np
    import pandas as pd
    from ..utils.plot import plot_timeseries

    x = np.arange(500) * 0.1
    df = pd.DataFrame({"x": np.sin(x), "y": np.cos(x + 1.57)})
    df.index = pd.to_datetime(df.index * 60, unit="s")
    df.index.rename("timestamp", inplace=True)
    return plot_timeseries(df)
