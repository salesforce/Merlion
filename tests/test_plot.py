import logging
import math
import os
from os.path import abspath, dirname, join
import sys
import unittest

from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample
from merlion.models.anomaly.forecast_based.prophet import (
    ProphetDetector,
    ProphetDetectorConfig,
)
from merlion.utils.time_series import ts_csv_load

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(abspath(__file__)))


class TestPlot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.csv_name = join(rootdir, "data", "example.csv")
        self.data = TemporalResample("15min")(ts_csv_load(self.csv_name, n_vars=1))
        logger.info(f"Data looks like:\n{self.data[:5]}")

        # Test Prophet
        self.test_len = math.ceil(len(self.data) / 5)
        self.vals_train = self.data[: -self.test_len]
        self.vals_test = self.data[-self.test_len :]
        self.model = ProphetDetector(
            ProphetDetectorConfig(transform=Identity(), uncertainty_samples=1000)
        )

    def test_plot(self):
        print("-" * 80)
        logger.info("test_plot\n" + "-" * 80 + "\n")
        logger.info("Training model...\n")
        self.model.train(self.vals_train)

        figdir = join(rootdir, "tmp", "plot")
        os.makedirs(figdir, exist_ok=True)

        # Test various plots with matplotlib
        fig, _ = self.model.plot_anomaly(
            time_series=self.vals_test,
            filter_scores=False,
            plot_forecast=True,
            plot_forecast_uncertainty=True,
        )
        fig.savefig(join(figdir, "prophet_anom_raw.png"))

        fig, _ = self.model.plot_anomaly(
            time_series=self.vals_test,
            filter_scores=True,
            plot_forecast=True,
            plot_forecast_uncertainty=True,
        )
        fig.savefig(join(figdir, "prophet_anom_filtered.png"))

        var = self.vals_test.univariates[self.vals_test.names[0]]
        fig, _ = self.model.plot_forecast(
            time_stamps=var.time_stamps,
            time_series_prev=self.vals_train,
            plot_time_series_prev=True,
            plot_forecast_uncertainty=True,
        )
        fig.savefig(join(figdir, "prophet_forecast.png"))

        # Test plotting with plotly
        try:
            import plotly

            fig = self.model.plot_anomaly_plotly(
                time_series=self.vals_test,
                time_series_prev=self.vals_train,
                plot_forecast=True,
                plot_forecast_uncertainty=True,
                plot_time_series_prev=True,
            )
            try:
                import kaleido

                fig.write_image((join(figdir, "prophet_plotly.png")), engine="kaleido")
            except ImportError:
                logger.info("kaleido not installed, not trying to save image")
        except ImportError:
            logger.info("plotly not installed, not trying plotly visualization")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    unittest.main()
