#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dash_table, dcc
from merlion.dashboard.settings import *


def data_table(df, n=1000, page_size=10):
    if df is not None:
        df = df.head(n)
        columns = [{"name": "Index", "id": "Index"}] + [{"name": c, "id": c} for c in df.columns]
        data = []
        for i in range(df.shape[0]):
            d = {c: v for c, v in zip(df.columns, df.values[i])}
            d.update({"Index": df.index[i]})
            data.append(d)

        table = dash_table.DataTable(
            id="table",
            columns=columns,
            data=data,
            style_cell_conditional=[{"textAlign": "center"}],
            style_table={"overflowX": "scroll"},
            editable=False,
            column_selectable="single",
            page_action="native",
            page_size=page_size,
            page_current=0,
            style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
            style_data=dict(backgroundColor=TABLE_DATA_COLOR),
        )
        return table
    else:
        return dash_table.DataTable()


def plot_timeseries(ts, figure_height=500):
    traces = []
    color_list = plotly.colors.qualitative.Dark24
    for i, col in enumerate(ts.columns):
        v = ts[col]
        if v.dtype in ["int", "float", "bool"]:
            v = v.astype(float)
            color = color_list[i % len(color_list)]
            traces.append(go.Scatter(name=col, x=v.index, y=v.values.flatten(), mode="lines", line=dict(color=color)))

    layout = dict(
        showlegend=True,
        xaxis=dict(
            title="Time",
            type="date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
        ),
    )
    fig = make_subplots(figure=go.Figure(layout=layout))
    fig.update_yaxes(title_text="Time Series")
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(height=figure_height, xaxis_rangeselector_font_color='white', xaxis_rangeselector_activecolor='#0176D3', xaxis_rangeselector_bgcolor='#1B96FF', xaxis_rangeselector_font_family='Salesforce Sans')
    return dcc.Graph(figure=fig)
