#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Module for visualizing model predictions.
"""
import logging
from typing import Dict
from copy import copy

from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merlion.utils import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)
try:
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    logger.warning(
        "plotly not installed, so plotly visualizations will not work. Try installing Merlion with optional "
        "dependencies using `pip install salesforce-merlion[plot]` or `pip install `salesforce-merlion[all]`."
    )


def plot_anoms(ax: plt.Axes, anomaly_labels: TimeSeries):
    """
    Plots anomalies as pink windows on the matplotlib ``Axes`` object ``ax``.
    """
    anomaly_labels = anomaly_labels.to_pd()
    t, y = anomaly_labels.index, anomaly_labels.values
    splits = np.where(y[1:] != y[:-1])[0] + 1
    splits = np.concatenate(([0], splits, [len(y) - 1]))
    for k in range(len(splits) - 1):
        if y[splits[k]]:  # If splits[k] is anomalous
            ax.axvspan(t[splits[k]], t[splits[k + 1]], color="#e07070", alpha=0.5)


def plot_anoms_plotly(fig, anomaly_labels: TimeSeries):
    """
    Plots anomalies as pink windows on the plotly ``Figure`` object ``fig``.
    """
    anomaly_labels = anomaly_labels.to_pd()
    t, y = anomaly_labels.index, anomaly_labels.values
    splits = np.where(y[1:] != y[:-1])[0] + 1
    splits = np.concatenate(([0], splits, [len(y) - 1]))
    for k in range(len(splits) - 1):
        if y[splits[k]]:  # If splits[k] is anomalous
            fig.add_vrect(t[splits[k]], t[splits[k + 1]], line_width=0, fillcolor="#e07070", opacity=0.4)


class Figure:
    """
    Class for visualizing predictions of univariate anomaly detection & forecasting models.
    """

    _default_label_alias = dict(yhat="Forecast", anom="Anomaly Score")

    def __init__(
        self,
        y: UnivariateTimeSeries = None,
        anom: UnivariateTimeSeries = None,
        yhat: UnivariateTimeSeries = None,
        yhat_lb: UnivariateTimeSeries = None,
        yhat_ub: UnivariateTimeSeries = None,
        y_prev: UnivariateTimeSeries = None,
        yhat_prev: UnivariateTimeSeries = None,
        yhat_prev_lb: UnivariateTimeSeries = None,
        yhat_prev_ub: UnivariateTimeSeries = None,
    ):
        """
        :param y: the true value of the time series
        :param anom: anomaly scores returned by a model
        :param yhat: forecast returned by a model
        :param yhat_lb: lower bound on ``yhat`` (if model supports uncertainty estimation)
        :param yhat_ub: upper bound on ``yhat`` (if model supports uncertainty estimation)
        :param y_prev: portion of time series preceding ``y``
        :param yhat_prev: model's forecast of ``y_prev``
        :param yhat_prev_lb: lower bound on ``yhat_prev`` (if model supports uncertainty estimation)
        :param yhat_prev_ub: upper bound on ``yhat_prev`` (if model supports uncertainty estimation)
        """
        assert not (anom is not None and y is None), "If `anom` is given, `y` must also be given"

        if yhat is None:
            assert yhat_lb is None and yhat_ub is None, "Can only give `yhat_lb` and `yhat_ub` if `yhat` is given"
        else:
            assert (yhat_lb is None and yhat_ub is None) or (
                yhat_lb is not None and yhat_ub is not None
            ), "Must give both or neither of `yhat_lb` and `yhat_ub`"

        if yhat_prev is None:
            assert (
                yhat_prev_lb is None and yhat_prev_ub is None
            ), "Can only give `yhat_prev_lb` and `yhat_prev_ub` if `yhat_prev` is given"
        else:
            assert (yhat_prev_lb is None and yhat_prev_ub is None) or (
                yhat_prev_lb is not None and yhat_prev_ub is not None
            ), "Must give both or neither of `yhat_prev_lb` and `yhat_prev_ub`"

        self.y = y
        self.anom = anom
        self.yhat = yhat
        if yhat_lb is not None and yhat_ub is not None:
            self.yhat_iqr = TimeSeries({"lb": yhat_lb, "ub": yhat_ub}).align()
        else:
            self.yhat_iqr = None

        self.y_prev = y_prev
        self.yhat_prev = yhat_prev
        if yhat_prev_lb is not None and yhat_prev_ub is not None:
            self.yhat_prev_iqr = TimeSeries({"lb": yhat_prev_lb, "ub": yhat_prev_ub}).align()
        else:
            self.yhat_prev_iqr = None

    @property
    def t0(self):
        """
        :return: First time being plotted.
        """
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return min(y.index[0] for y in ys if y is not None and len(y) > 0)

    @property
    def tf(self):
        """
        :return: Final time being plotted.
        """
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return max(y.index[-1] for y in ys if y is not None and len(y) > 0)

    @property
    def t_split(self):
        """
        :return: Time splitting train from test.
        """
        if self.y_prev is not None:
            return self.y_prev.index[-1]
        if self.yhat_prev is not None:
            return self.yhat_prev.index[-1]
        return None

    def get_y(self):
        """Get all y's (actual values)"""
        if self.y is not None and self.y_prev is not None:
            return self.y_prev.concat(self.y)
        elif self.y_prev is not None:
            return self.y_prev
        elif self.y is not None:
            return self.y
        else:
            return None

    def get_yhat(self):
        """Get all yhat's (predicted values)."""
        if self.yhat is not None and self.yhat_prev is not None:
            return self.yhat_prev.concat(self.yhat)
        elif self.yhat_prev is not None:
            return self.yhat_prev
        elif self.yhat is not None:
            return self.yhat
        else:
            return None

    def get_yhat_iqr(self):
        """Get IQR of predicted values."""
        if self.yhat_iqr is not None and self.yhat_prev_iqr is not None:
            return self.yhat_prev_iqr + self.yhat_iqr
        elif self.yhat_prev_iqr is not None:
            return self.yhat_prev_iqr
        elif self.yhat_iqr is not None:
            return self.yhat_iqr
        else:
            return None

    def plot(self, title=None, metric_name=None, figsize=(1000, 600), ax=None, label_alias: Dict[str, str] = None):
        """
        Plots the figure in matplotlib.

        :param title: title of the plot.
        :param metric_name: name of the metric (y axis)
        :param figsize: figure size in pixels
        :param ax: matplotlib axes to add the figure to.
        :param label_alias: dict which maps entities in the figure,
            specifically ``y_hat`` and ``anom`` to their label names.

        :return: (fig, ax): matplotlib figure & matplotlib axes
        """
        # determine full label alias
        label_alias = {} if label_alias is None else label_alias
        full_label_alias = copy(self._default_label_alias)
        full_label_alias.update(label_alias)

        # Get the figure
        figsize = (figsize[0] / 100, figsize[1] / 100)
        if ax is None:
            fig = plt.figure(facecolor="w", figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.set_facecolor((0.9, 0.9, 0.9))

        # Get & plot the actual value (if applicable)
        lines = []
        y = self.get_y()
        if y is not None:
            metric_name = y.name if metric_name is None else metric_name
            ln = ax.plot(y.index, y.np_values, c="k", alpha=0.8, lw=1, zorder=1, label=metric_name)
            lines.extend(ln)

        # Dotted line to cordon off previous times from current ones
        t_split = self.t_split
        if t_split is not None:
            ax.axvline(t_split, ls="--", lw=2, c="k")

        # Get & plot the prediction (if applicable)
        yhat = self.get_yhat()
        if yhat is not None:
            metric_name = yhat.name if metric_name is None else metric_name
            yhat_label = full_label_alias.get("yhat")
            ln = ax.plot(yhat.index, yhat.np_values, ls="-", c="#0072B2", zorder=0, label=yhat_label)
            lines.extend(ln)

        # Get & plot the uncertainty of the prediction (if applicable)
        iqr = self.get_yhat_iqr()
        if iqr is not None:
            lb, ub = iqr.univariates["lb"], iqr.univariates["ub"]
            ax.fill_between(lb.index, lb.values, ub.values, color="#0072B2", alpha=0.2, zorder=2)

        # Plot anomaly scores if desired
        if self.anom is not None and self.y is not None:
            ax2 = ax.twinx()
            anom_vals = self.anom.np_values
            anom_label = full_label_alias.get("anom")
            ln = ax2.plot(self.anom.index, anom_vals, color="r", label=anom_label)
            ax2.set_ylabel(anom_label)
            minval, maxval = min(anom_vals), max(anom_vals)
            delta = maxval - minval
            if delta > 0:
                ax2.set_ylim(minval - delta / 8, maxval + 2 * delta)
            else:
                ax2.set_ylim(minval - 1 / 30, maxval + 1)
            lines.extend(ln)

        # Format the axes before returning the figure
        locator = AutoDateLocator(interval_multiples=False)
        formatter = AutoDateFormatter(locator)
        ax.set_xlim(self.t0 - (self.tf - self.t0) / 20, self.tf + (self.tf - self.t0) / 20)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric_name)
        ax.set_title(title if title else metric_name)
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
        fig.tight_layout()
        return fig, ax

    def plot_plotly(self, title=None, metric_name=None, figsize=(1000, 600), label_alias: Dict[str, str] = None):
        """
        Plots the figure in plotly.

        :param title: title of the plot.
        :param metric_name: name of the metric (y axis)
        :param figsize: figure size in pixels
        :param label_alias: dict which maps entities in the figure,
            specifically ``y_hat`` and ``anom`` to their label names.

        :return: plotly figure.
        """
        # determine full label alias
        label_alias = {} if label_alias is None else label_alias
        full_label_alias = copy(self._default_label_alias)
        full_label_alias.update(label_alias)

        prediction_color = "#0072B2"
        error_color = "rgba(0, 114, 178, 0.2)"  # '#0072B2' with 0.2 opacity
        actual_color = "black"
        anom_color = "red"
        line_width = 2

        traces = []
        y = self.get_y()
        yhat = self.get_yhat()
        iqr = self.get_yhat_iqr()
        if metric_name is None:
            if y is not None:
                metric_name = y.name
            elif yhat is not None:
                metric_name = yhat.name

        if iqr is not None:
            lb = iqr.univariates["lb"]
            traces.append(
                go.Scatter(
                    x=lb.index, y=lb.np_values, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
                )
            )

        if yhat is not None:
            fill_mode = "tonexty" if iqr is not None else "none"
            yhat_label = full_label_alias.get("yhat")
            traces.append(
                go.Scatter(
                    name=yhat_label,
                    x=yhat.index,
                    y=yhat.np_values,
                    mode="lines",
                    line=dict(color=prediction_color, width=line_width),
                    fillcolor=error_color,
                    fill=fill_mode,
                )
            )

        if iqr is not None:
            ub = iqr.univariates["ub"]
            traces.append(
                go.Scatter(
                    x=ub.index,
                    y=ub.np_values,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    fillcolor=error_color,
                    fill="tonexty",
                )
            )

        if y is not None:
            traces.append(
                go.Scatter(
                    name=y.name, x=y.index, y=y.values, mode="lines", line=dict(color=actual_color, width=line_width)
                )
            )

        anom_trace = None
        if self.anom is not None:
            anom_label = full_label_alias.get("anom")
            anom_trace = go.Scatter(
                name=anom_label,
                x=self.anom.index,
                y=self.anom.np_values,
                mode="lines",
                line=dict(color=anom_color, width=line_width),
            )

        layout = dict(
            showlegend=True,
            width=figsize[0],
            height=figsize[1],
            yaxis=dict(title=metric_name),
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
                rangeslider=dict(visible=True),
            ),
        )
        title = title if title else metric_name
        if title is not None:
            layout["title"] = title
        fig = make_subplots(
            specs=[[{"secondary_y": anom_trace is not None}]], figure=go.Figure(data=traces, layout=layout)
        )
        if self.t_split is not None:
            fig.add_vline(x=self.t_split, line_dash="dot", line_color="black", line_width=2)
        if anom_trace is not None:
            fig.add_trace(anom_trace, secondary_y=True)
            minval, maxval = min(self.anom.np_values), max(self.anom.np_values)
            delta = maxval - minval
            if delta > 0:
                minval, maxval = minval - delta / 8, maxval + 2 * delta
            else:
                minval, maxval = minval - 1 / 30, maxval + 1
            fig.update_yaxes(title_text=anom_label, range=[minval, maxval], secondary_y=True)

        return fig


class MTSFigure:
    def __init__(
        self,
        y: TimeSeries = None,
        anom: TimeSeries = None,
        yhat: TimeSeries = None,
        yhat_lb: TimeSeries = None,
        yhat_ub: TimeSeries = None,
        y_prev: TimeSeries = None,
        yhat_prev: TimeSeries = None,
        yhat_prev_lb: TimeSeries = None,
        yhat_prev_ub: TimeSeries = None,
    ):
        assert y is not None, "`y` must be given"

        if yhat is None:
            assert yhat_lb is None and yhat_ub is None, "Can only give `yhat_lb` and `yhat_ub` if `yhat` is given"
        else:
            assert (yhat_lb is None and yhat_ub is None) or (
                yhat_lb is not None and yhat_ub is not None
            ), "Must give both or neither of `yhat_lb` and `yhat_ub`"

        if yhat_prev is None:
            assert (
                yhat_prev_lb is None and yhat_prev_ub is None
            ), "Can only give `yhat_prev_lb` and `yhat_prev_ub` if `yhat_prev` is given"
        else:
            assert (yhat_prev_lb is None and yhat_prev_ub is None) or (
                yhat_prev_lb is not None and yhat_prev_ub is not None
            ), "Must give both or neither of `yhat_prev_lb` and `yhat_prev_ub`"

        self.y = y
        self.anom = anom
        self.yhat = yhat
        self.yhat_lb = yhat_lb
        self.yhat_ub = yhat_ub

        self.y_prev = y_prev
        self.yhat_prev = yhat_prev
        self.yhat_prev_lb = yhat_prev_lb
        self.yhat_prev_ub = yhat_prev_ub

    @property
    def t0(self):
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return min(y.t0 for y in ys if y is not None and len(y) > 0)

    @property
    def tf(self):
        ys = [self.anom, self.y, self.yhat, self.y_prev, self.yhat_prev]
        return max(y.tf for y in ys if y is not None and len(y) > 0)

    @property
    def t_split(self):
        if self.y_prev is not None:
            return pd.to_datetime(self.y_prev.tf, unit="s")
        if self.yhat_prev is not None:
            return pd.to_datetime(self.yhat_prev.tf, unit="s")
        return None

    @staticmethod
    def _combine_prev(x, x_prev):
        if x is not None and x_prev is not None:
            return x_prev + x
        elif x_prev is not None:
            return x_prev
        elif x is not None:
            return x
        else:
            return None

    def get_y(self):
        """Get all y's (actual values)"""
        return self._combine_prev(self.y, self.y_prev)

    def get_yhat(self):
        """Get all yhat's (predicted values)."""
        return self._combine_prev(self.yhat, self.yhat_prev)

    def get_yhat_iqr(self):
        """Get IQR of predicted values."""
        return self._combine_prev(self.yhat_lb, self.yhat_prev_lb), self._combine_prev(self.yhat_ub, self.yhat_prev_ub)

    @staticmethod
    def _get_layout(title, figsize):
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
                rangeslider=dict(visible=True),
            ),
        )
        layout["title"] = title if title else "Untitled"
        if figsize is not None:
            assert len(figsize) == 2, "figsize should be (width, height)."
            layout["width"] = figsize[0]
            layout["height"] = figsize[1]
        return layout

    def plot_plotly(self, title=None, figsize=None):
        """
        Plots the figure in plotly.
        :param title: title of the plot.
        :param figsize: figure size in pixels
        :return: plotly figure.
        """
        prediction_color = "#0072B2"
        error_color = "rgba(0, 114, 178, 0.2)"  # '#0072B2' with 0.2 opacity
        anom_color = "red"

        traces = []
        y = self.get_y()
        yhat = self.get_yhat()
        lb, ub = self.get_yhat_iqr()

        color_list = plotly.colors.qualitative.Dark24
        valid_idx = [i for i in range(len(color_list)) if i not in [3, 12]]  # exclude red to make anom trace clearer
        for i, name in enumerate(y.names):
            v = y.univariates[name]
            color = color_list[valid_idx[i % len(valid_idx)]]
            traces.append(go.Scatter(name=name, x=v.index, y=v.np_values, mode="lines", line=dict(color=color)))
            if lb is not None and name in lb.names:
                v = lb.univariates[name]
                traces.append(
                    go.Scatter(
                        x=v.index, y=v.np_values, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
                    )
                )
            if yhat is not None and name in yhat.names:
                v = yhat.univariates[name]
                fill_mode = "tonexty" if lb is not None or ub is not None else "none"
                traces.append(
                    go.Scatter(
                        name=f"{name}_forecast",
                        x=v.index,
                        y=v.np_values,
                        mode="lines",
                        line=dict(color=prediction_color),
                        fillcolor=error_color,
                        fill=fill_mode,
                    )
                )
            if ub is not None and name in ub.names:
                v = ub.univariates[name]
                traces.append(
                    go.Scatter(
                        x=v.index,
                        y=v.np_values,
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        fillcolor=error_color,
                        fill="tonexty",
                    )
                )

        anom_trace = None
        if self.anom is not None:
            v = self.anom.univariates[self.anom.names[0]]
            anom_trace = go.Scatter(
                name="Anomaly Score", x=v.index, y=v.np_values, mode="lines", line=dict(color=anom_color)
            )

        fig = make_subplots(
            specs=[[{"secondary_y": anom_trace is not None}]], figure=go.Figure(layout=self._get_layout(title, figsize))
        )
        if anom_trace is not None:
            fig.add_trace(anom_trace, secondary_y=True)
            v = self.anom.univariates[self.anom.names[0]]
            minval, maxval = min(v.np_values), max(v.np_values)
            delta = maxval - minval
            if delta > 0:
                minval, maxval = minval - delta / 8, maxval + 2 * delta
            else:
                minval, maxval = minval - 1 / 30, maxval + 1
            fig.update_yaxes(title_text="Anomaly Score", range=[minval, maxval], secondary_y=True)
        for trace in traces:
            fig.add_trace(trace)
        if self.t_split is not None:
            fig.add_vline(x=self.t_split, line_dash="dot", line_color="black", line_width=2)
        return fig
