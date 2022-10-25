import unittest
import numpy as np
import pandas as pd
from merlion.models.factory import ModelFactory
from merlion.utils.time_series import TimeSeries


class TestAnomaly(unittest.TestCase):

    def setUp(self) -> None:
        t = 1000
        x = np.random.randn(t) * 0.2
        y = np.zeros(t)
        for i in range(t):
            if (i + 1) % 100 == 0:
                x[i: i + 5] = x[i: i + 5] + (np.random.rand() + 1) * 2
                y[i: i + 5] = 1
        df = pd.DataFrame({"x": x, "y": y})
        df.index = pd.to_datetime(df.index * 60, unit="s")
        df.index.rename("timestamp", inplace=True)
        self.df = df

    def test(self):
        algorithm = "ArimaDetector"
        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class())

        df = self.df[["x"]]
        train_df = df.iloc[:500]
        test_df = df.iloc[500:]

        train_ts = TimeSeries.from_pd(train_df)
        test_ts = TimeSeries.from_pd(test_df)

        scores = model.train(train_data=train_ts)
        predictions = model.post_rule(scores) if model.post_rule is not None else scores
        predictions = model.get_anomaly_label(time_series=test_ts)


if __name__ == "__main__":
    unittest.main()
