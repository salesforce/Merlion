import unittest
import numpy as np
import pandas as pd
from merlion.models.factory import ModelFactory
from merlion.utils.time_series import TimeSeries
from merlion.evaluate.forecast import ForecastEvaluator, ForecastMetric


class TestForecast(unittest.TestCase):

    def setUp(self) -> None:
        t = np.arange(500) * 0.1
        df = pd.DataFrame({"x": np.sin(t), "y": np.cos(t)})
        df.index = pd.to_datetime(df.index * 60, unit="s")
        df.index.rename("timestamp", inplace=True)
        self.df = df

    def test(self):
        algorithm = "Arima"
        max_forecast_steps = 100
        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class(
            target_seq_index=0,
            max_forecast_steps=max_forecast_steps
        ))
        evaluator = ForecastEvaluator(model, config=ForecastEvaluator.config_class())

        ts = TimeSeries.from_pd(self.df)
        predictions = model.train(ts)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        rmses = evaluator.evaluate(ground_truth=ts, predict=predictions, metric=ForecastMetric.RMSE)
        smapes = evaluator.evaluate(ground_truth=ts, predict=predictions, metric=ForecastMetric.sMAPE)
        print((rmses, smapes))

        n = len(ts) // 2
        ts_a, ts_b = ts.bisect(t=ts.time_stamps[n], t_in_left=True)
        if hasattr(model, "max_forecast_steps"):
            n = min(len(ts_b) - 1, model.max_forecast_steps)
            ts_b, _ = ts_b.bisect(t=ts_b.time_stamps[n])

        figure = model.plot_forecast_plotly(
            time_series=ts_b,
            time_series_prev=ts_a,
            plot_forecast_uncertainty=True
        )
        figure.show()


if __name__ == "__main__":
    unittest.main()
