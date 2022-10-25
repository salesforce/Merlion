import unittest
from dashboard.models.forecast import ForecastModel
from dashboard.models.anomaly import AnomalyModel


class TestForecastModel(unittest.TestCase):

    def test_forecast(self):
        algorithm = "LSTM"
        param_info = ForecastModel.get_parameter_info(algorithm)
        for name, info in param_info.items():
            print(name)
            print(info)

    def test_anomaly(self):
        algorithm = "VAE"
        param_info = AnomalyModel.get_parameter_info(algorithm)
        for name, info in param_info.items():
            print(name)
            print(info)


if __name__ == "__main__":
    unittest.main()
