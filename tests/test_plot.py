#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import math
import os
from os.path import abspath, dirname, join
import sys
import unittest

from merlion.transform.base import Identity
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.normalize import BoxCoxTransform
from merlion.transform.resample import TemporalResample
from merlion.models.anomaly.forecast_based.prophet import ProphetDetector, ProphetDetectorConfig
from merlion.models.forecast.trees import LGBMForecaster, LGBMForecasterConfig
from merlion.plot import plot_anoms, plot_anoms_plotly
from merlion.utils.data_io import csv_to_time_series

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(abspath(__file__)))


class TestPlot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.csv_name = join(rootdir, "data", "example.csv")
        data = csv_to_time_series(self.csv_name, timestamp_unit="ms")
        self.data = TemporalResample("15min")(data.univariates[data.names[0]].to_ts())
        self.labels = data.univariates[data.names[1]].to_ts()
        logger.info(f"Data looks like:\n{self.data[:5]}")

        # Test Prophet
        self.test_len = math.ceil(len(self.data) / 5)
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]

    def test_plot(self):
        print("-" * 80)
        logger.info("test_plot\n" + "-" * 80 + "\n")
        self.model = ProphetDetector(
            ProphetDetectorConfig(transform=Identity(), invert_transform=False, uncertainty_samples=1000)
        )
        self._test_plot(subdir="basic")

    def test_plot_transform_inv(self):
        print("-" * 80)
        logger.info("test_plot_transform_inv\n" + "-" * 80 + "\n")
        self.model = ProphetDetector(
            ProphetDetectorConfig(transform=BoxCoxTransform(), invert_transform=True, uncertainty_samples=1000)
        )
        self._test_plot(subdir="transform_inv")

    def test_plot_transform_no_inv(self):
        print("-" * 80)
        logger.info("test_plot_transform_no_inv\n" + "-" * 80 + "\n")
        self.model = ProphetDetector(
            ProphetDetectorConfig(transform=DifferenceTransform(), invert_transform=False, uncertainty_samples=1000)
        )
        self._test_plot(subdir="transform_no_inv")

    def test_no_uncertainty(self):
        self.model = LGBMForecaster(
            LGBMForecasterConfig(transform=TemporalResample("1h"), maxlags=24 * 7, prediction_stride=24)
        )
        self.model.train(self.vals_train)
        fig, _ = self.model.plot_forecast(time_series=self.vals_test, plot_forecast_uncertainty=True)
        figdir = join(rootdir, "tmp", "plot", "no_uncertainty")
        os.makedirs(figdir, exist_ok=True)
        fig.savefig(join(figdir, "lgbm_forecast.png"))

    def _test_plot(self, subdir):
        logger.info("Training model...\n")
        self.model.train(self.vals_train)

        figdir = join(rootdir, "tmp", "plot", subdir)
        os.makedirs(figdir, exist_ok=True)

        # Test various plots with matplotlib
        fig, ax = self.model.plot_anomaly(
            time_series=self.vals_test, filter_scores=False, plot_forecast=True, plot_forecast_uncertainty=True
        )
        plot_anoms(ax, self.labels)
        fig.savefig(join(figdir, "prophet_anom_raw.png"))

        fig, ax = self.model.plot_anomaly(
            time_series=self.vals_test, filter_scores=True, plot_forecast=True, plot_forecast_uncertainty=True
        )
        plot_anoms(ax, self.labels)
        fig.savefig(join(figdir, "prophet_anom_filtered.png"))

        fig, _ = self.model.plot_forecast(
            time_stamps=self.vals_test.time_stamps,
            time_series_prev=self.vals_train,
            plot_time_series_prev=True,
            plot_forecast_uncertainty=True,
        )
        fig.savefig(join(figdir, "prophet_forecast.png"))

        # Test plotting with plotly
        try:
            import plotly

            fig1 = self.model.plot_anomaly_plotly(
                time_series=self.vals_test,
                time_series_prev=self.vals_train,
                plot_forecast=True,
                plot_forecast_uncertainty=True,
                plot_time_series_prev=True,
            )
            plot_anoms_plotly(fig1, self.labels)

            fig2 = self.model.plot_forecast_plotly(time_series=self.vals_test, plot_forecast_uncertainty=False)
            try:
                import kaleido

                fig1.write_image((join(figdir, "prophet_plotly.png")), engine="kaleido")
                fig2.write_image((join(figdir, "prophet_forecast_plotly.png")), engine="kaleido")
            except ImportError:
                logger.info("kaleido not installed, not trying to save image")
        except ImportError:
            logger.info("plotly not installed, not trying plotly visualization")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
