import logging
from typing import List, Tuple

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.utils.time_series import to_pd_datetime, TimeSeries
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset, max_feasible_forecast_steps

logger = logging.getLogger(__name__)


class AutoRegressiveForecasterConfig(ForecasterConfig):
    """
    Configuration class for auto-regressive forecaster model.
    """

    def __init__(
        self,
        maxlags: int,
        max_forecast_steps: int = None,
        target_seq_index: int = None,
        prediction_stride: int = 1,
        **kwargs,
    ):
        """
        :param maxlags: Max # of lags for forecasting
        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        :param prediction_stride: the prediction step for training and forecasting

            - If univariate: the sequence target of the length of prediction_stride will be utilized, forecasting will
              be done autoregressively, with the stride unit of prediction_stride
            - If multivariate:

                - if = 1: the autoregression with the stride unit of 1
                - if > 1: only support sequence mode, and the model will set prediction_stride = max_forecast_steps
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.maxlags = maxlags
        self.prediction_stride = prediction_stride


class AutoRegressiveForecaster(ForecasterBase):

    config_class = AutoRegressiveForecasterConfig
    model = None

    def __init__(self, config: AutoRegressiveForecasterConfig):
        super().__init__(config)

    @property
    def maxlags(self) -> int:
        return self.config.maxlags

    @property
    def prediction_stride(self) -> int:
        return self.config.prediction_stride

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def require_univariate(self) -> bool:
        return False

    @property
    def _default_train_config(self):
        return dict()

    def _train(self, train_data: pd.DataFrame, train_config=None):
        train_data = TimeSeries.from_pd(train_data)
        fit = train_config.get("fit", True)

        assert self.dim == train_data.dim

        if self.dim == 1:
            logger.info(
                f"Model is working on a univariate dataset, "
                f"hybrid of sequence and autoregression training strategy will be adopted "
                f"with prediction_stride = {self.prediction_stride} "
            )
        else:
            assert self.prediction_stride == 1, \
                "AutoRegressive model only handles prediction_stride == 1 for multivariate"
            logger.info(
                f"Model is working on a multivariate dataset with prediction_stride = 1, "
                f"autoregression training strategy will be adopted "
            )

        # process train data to the rolling window dataset
        rolling_window_data = RollingWindowDataset(train_data,
                                                   self.target_seq_index,
                                                   self.maxlags,
                                                   self.prediction_stride)
        if self.dim == 1:
            # hybrid of seq and autoregression for univariate
            inputs_train, labels_train, labels_train_ts = rolling_window_data.process_rolling_train_data()
        else:
            # autoregression for multivariate
            inputs_train, labels_train, labels_train_ts = rolling_window_data.process_regressive_train_data()

        # fitting
        if fit:
            self.model.fit(inputs_train, labels_train)

        # forecasting
        inputs_train = np.atleast_2d(inputs_train)
        if self.dim == 1:
            pred = self._hybrid_forecast(inputs_train, self.max_forecast_steps or len(inputs_train) - self.maxlags)
        else:
            pred = self._autoregressive_forecast(inputs_train, max(self.max_forecast_steps or 0, 1)
            )
        # since the model may predict multiple steps, we concatenate all the first steps together
        pred = pred[:, 0].reshape(-1)
        return pd.DataFrame(pred, index=labels_train_ts, columns=[self.target_name]), None

    def _forecast(
            self, time_stamps: List[int], time_series_prev: pd.DataFrame = None, return_prev=False
    ) -> Tuple[pd.DataFrame, None]:
        if time_series_prev is not None:
            assert len(time_series_prev) >= self.maxlags, (
                f"time_series_prev has a data length of "
                f"{len(time_series_prev)} that is shorter than the maxlags "
                f"for the model"
            )
        if self.dim > 1:
            assert self.prediction_stride == 1, \
                "AutoRegressive model only handles prediction_stride == 1 for multivariate"

        n = len(time_stamps)
        prev_pred, prev_err = None, None
        if time_series_prev is None:
            time_series_prev = self.transform(self.train_data)
        elif time_series_prev is not None and return_prev:
            prev_pred, prev_err = self._train(time_series_prev, train_config=dict(fit=False))

        time_series_prev = TimeSeries.from_pd(time_series_prev)
        assert self.dim == time_series_prev.dim

        rolling_window_data = RollingWindowDataset(time_series_prev,
                                                   self.target_seq_index,
                                                   self.maxlags,
                                                   self.prediction_stride)
        time_series_prev_no_ts = rolling_window_data.process_one_step_prior()
        if self.dim == 1:
            yhat = self._hybrid_forecast(np.atleast_2d(time_series_prev_no_ts), n).reshape(-1)
        else:
            yhat = self._autoregressive_forecast(time_series_prev_no_ts, n).reshape(-1)

        forecast = pd.DataFrame(yhat, index=to_pd_datetime(time_stamps), columns=[self.target_name])
        if prev_pred is not None:
            forecast = pd.concat((prev_pred, forecast))
        return forecast, None

    def _hybrid_forecast(self, inputs, steps=None):
        """
        n-step autoregression method for univairate data, each regression step updates n_prediction_steps data points
        :return: pred of target_seq_index for steps [n_samples, steps]
        """
        if steps is None:
            steps = self.max_forecast_steps

        inputs = np.atleast_2d(inputs)

        pred = np.empty((len(inputs), (int((steps - 1) / self.prediction_stride) + 1) * self.prediction_stride))
        start = 0
        while True:
            next_forecast = self.model.predict(inputs)
            if len(next_forecast.shape) == 1:
                next_forecast = np.expand_dims(next_forecast, axis=1)
            pred[:, start: start + self.prediction_stride] = next_forecast
            start += self.prediction_stride
            if start >= steps:
                break
            inputs = self._update_prior_1d(inputs, next_forecast)
        return pred[:, :steps]

    def _autoregressive_forecast(self, inputs, steps=None):
        """
        1-step auto-regression method for multivariate data, each regression step updates one data point for each sequence
        :return: pred of target_seq_index for steps [n_samples, steps]
        """

        if steps is None:
            steps = self.max_forecast_steps

        inputs = np.atleast_2d(inputs)

        pred = np.empty((len(inputs), steps))

        for i in range(steps):
            # next forecast shape: [n_samples, self.dim]
            next_forecast = self.model.predict(inputs)
            pred[:, i] = next_forecast[:, self.target_seq_index]
            if i == steps - 1:
                break
            inputs = self._update_prior_nd(inputs, next_forecast)
        return pred

    def _update_prior_nd(self, prior: np.ndarray, next_forecast: np.ndarray):
        """
        regressively update the prior by concatenate prior with next_forecast on the sequence dimension,
        :param prior: the prior [n_samples, n_seq * maxlags]
        :param next_forecast: the next forecasting result [n_samples, n_seq]
        :return: updated prior
        """
        assert isinstance(prior, np.ndarray) and len(prior.shape) == 2
        assert isinstance(next_forecast, np.ndarray) and len(next_forecast.shape) == 2

        # unsqueeze the sequence dimension so prior and next_forecast can be concatenated along sequence dimension
        # for example,
        # prior = [[1,2,3,4,5,6,7,8,9], [10,20,30,40,50,60,70,80,90]], after the sequence dimension is expanded
        # prior = [[[1,2,3], [4,5,6], [7,8,9]],
        #          [[10,20,30],[40,50,60],[70,80,90]]
        #         ]
        # next_forcast = [[0.1,0.2,0.3],[0.4,0.5,0.6]], after the sequence dimension is expanded
        # next_forecast = [[[0.1],[0.2],[0.3]],
        #                  [[0.4],[0.5],[0.6]]
        #                 ]
        prior = prior.reshape(len(prior), self.dim, -1)
        next_forecast = np.expand_dims(next_forecast, axis=2)
        prior = np.concatenate([prior, next_forecast], axis=2)[:, :, -self.maxlags:]
        return prior.reshape(len(prior), -1)

    def _update_prior_1d(self, prior: np.ndarray, next_forecast: np.ndarray):
        """
        regressively update the prior by concatenate prior with next_forecast for the univariate,
        :param prior: the prior [n_samples, maxlags]
        :param next_forecast: the next forecasting result [n_samples, n_prediction_steps],
            if n_prediciton_steps ==1, maybe [n_samples,]
        :return: updated prior
        """
        assert isinstance(prior, np.ndarray) and len(prior.shape) == 2
        assert isinstance(next_forecast, np.ndarray)

        if len(next_forecast.shape) == 1:
            next_forecast = np.expand_dims(next_forecast, axis=1)
        prior = np.concatenate([prior, next_forecast], axis=1)[:, -self.maxlags:]
        return prior.reshape(len(prior), -1)



