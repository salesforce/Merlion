#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Base class for Deep Learning Forecasting Models
"""
import pdb
import copy
import time

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

from typing import List, Optional, Tuple, Union
from abc import abstractmethod

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)


from merlion.models.deep_base import DeepConfig, DeepModelBase
from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.models.utils.rolling_window_dataset import RollingWindowDataset
from merlion.utils.timefeatures import time_features

from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.utils.misc import initializer
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy

logger = logging.getLogger(__name__)


class DeepForecasterConfig(DeepConfig, ForecasterConfig):
    def __init__(
        self,
        n_past: int,
        max_forecast_steps: int = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 1,
        optim_name: str = "adam",
        criterion: str = "mse",
        target_seq_index: int = None,
        invert_transform=None,
        start_token_len: int = 0,
        ts_encoding: bool = True,
        ts_freq: str = "h",
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            optim_name=optim_name,
            criterion=criterion,
            max_forecast_steps=max_forecast_steps,
            target_seq_index=target_seq_index,
            invert_transform=invert_transform,
            **kwargs,
        )
        self.n_past = n_past

        self.ts_freq = ts_freq
        self.ts_encoding = ts_encoding
        self.start_token_len = start_token_len


class DeepForecaster(DeepModelBase, ForecasterBase):
    """
    Base class for a deep forecaster model
    """

    config_class = DeepForecasterConfig

    def __init__(self, config: DeepForecasterConfig):
        super().__init__(config)

    def _create_model(self):

        self.config.device = torch.device("cpu")

        self.deep_model = self.deep_model_class(self.config)
        print("I am at the creating model section")

        # initialize optimizer
        optim_dict = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
        }
        if self.config.optim_name in optim_dict:
            self.optimizer = optim_dict[self.config.optim_name](
                self.deep_model.parameters(),
                lr=self.config.lr,
            )
        else:
            raise NotImplementedError("%s optimizer is not supported yet..." % (self.config.optim_name))

        # initialize loss function
        if self.config.criterion == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError("%s loss is not supported..." % (self.config.criterion))

    def evaluate(self, evaluate_data: pd.DataFrame, metric: str = "mse"):
        config = self.config
        self.deep_model.eval()
        eval_dataset = RollingWindowDataset(
            evaluate_data,
            n_past=config.n_past,
            n_future=config.max_forecast_steps,
            batch_size=config.batch_size,
            target_seq_index=None,  # have to set None, we use config.target_seq_index later in the training
            ts_encoding=config.ts_encoding,
            ts_freq=config.ts_freq,
            start_token_len=config.start_token_len,
            flatten=False,
            shuffle=False,
        )

        all_preds = []
        all_trues = []

        for i, batch in enumerate(eval_dataset):
            with torch.no_grad():
                loss, outputs, y_true = self._deep_batch_iter(batch, self.config)
                pred = outputs.detach().cpu().numpy()
                true = y_true.detach().cpu().numpy()
                all_preds.append(pred)
                all_trues.append(true)

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        logger.info("test shape:" + str(preds.shape) + str(trues.shape))

        if metric == "mse":
            pred_err = np.mean(np.sum((preds - trues) ** 2, axis=-1), axis=-1)
        else:
            raise NotImplementedError

        return preds, pred_err

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:

        if train_config is not None:
            self.config = copy.deepcopy(train_config)
        config = self.config

        # creating model before the training
        self._create_model()

        train_dataset = RollingWindowDataset(
            train_data,
            n_past=config.n_past,
            n_future=config.max_forecast_steps,
            batch_size=config.batch_size,
            target_seq_index=None,  # have to set None, we use config.target_seq_index later in the training, if not this is a bug
            ts_encoding=config.ts_encoding,
            ts_freq=config.ts_freq,
            start_token_len=config.start_token_len,
            flatten=False,
            shuffle=True,
        )

        time_now = time.time()
        train_steps = len(train_dataset)
        logger.info("Training steps each epoch: %d" % (train_steps))
        # start training
        for epoch in range(config.num_epochs):
            iter_count = 0
            train_loss = []

            self.deep_model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_dataset):
                iter_count += 1
                self.optimizer.zero_grad()

                loss, _, _ = self._deep_batch_iter(batch, config)
                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if (i + 1) % 200 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((config.num_epochs - epoch) * train_steps - i)
                    logger.info("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            train_loss = np.average(train_loss)
            logger.info(
                "Epoch: {0}, Steps: {1}, cost time: {2}| Train Loss: {3:.7f}, ".format(
                    epoch + 1, train_steps, time.time() - epoch_time, train_loss
                )
            )

        # add adjusting learnign rate scheduling later.
        logger.info("Ending of the train loop")

        pred, pred_err = self.evaluate(train_data)

        # since the model predicts multiple steps, we concatenate all the first steps together
        return pd.DataFrame(pred[:, 0], columns=train_data.columns), None

    def _deep_batch_iter(self, batch, config):
        """
        For both loss calculation and data prediction
        """
        past, past_timestamp, future, future_timestamp = batch
        past = torch.FloatTensor(past, device=config.device)

        past_timestamp = torch.FloatTensor(past_timestamp, device=config.device)
        future_timestamp = torch.FloatTensor(future_timestamp, device=config.device)

        # if future is None, then we only need to do inference
        if future is None:
            start_token = past[:, -config.start_token_len :]
            dec_inp = torch.zeros(past.shape[0], config.max_forecast_steps, config.dec_in).float().to(config.device)
            dec_inp = torch.cat([start_token, dec_inp], dim=1)

        else:
            future = torch.FloatTensor(future, device=config.device)
            dec_inp = torch.zeros_like(future[:, -config.max_forecast_steps :, :]).float().to(config.device)
            dec_inp = torch.cat([future[:, : config.start_token_len, :], dec_inp], dim=1)

        model_output = self.deep_model(past, past_timestamp, dec_inp, future_timestamp)

        if future is None:
            return None, model_output, None

        if config.target_seq_index is None:
            target_future = future[:, -config.max_forecast_steps :].to(config.device)
        else:
            # choose specific target_seq_index for regression
            target_idx = config.target_seq_index
            target_future = future[:, -config.max_forecast_steps :, target_idx : target_idx + 1].to(config.device)

        loss = self.loss_fn(model_output, target_future)
        return loss, model_output, target_future

    def train_pre_process(
        self, train_data: TimeSeries, exog_data: TimeSeries = None, return_exog=None
    ) -> Union[TimeSeries, Tuple[TimeSeries, Union[TimeSeries, None]]]:
        train_data = super(ForecasterBase, self).train_pre_process(train_data)

        if self.dim == 1:
            self.config.target_seq_index = 0
        if self.config.target_seq_index is None:
            self.target_name = "Multi dimensional forecasting"
        else:
            assert 0 <= self.config.target_seq_index < train_data.dim, (
                f"Expected `target_seq_index` to be between 0 and {train_data.dim} "
                f"(the dimension of the transformed data), but got {self.config.target_seq_index}"
            )

            self.target_name = train_data.names[self.target_seq_index]

        return (train_data, exog_data) if return_exog else train_data

    # TODO: need to check and discuss with this one
    @property
    def require_even_sampling(self) -> bool:
        return False

    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame, return_prev=False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        FIXME: Need to discuss about this part API.
        Currently the implementation is make sure the batch size = 1
        Not sure about the meaning of the arguments
        """

        pred_datetime = to_pd_datetime(time_stamps)
        prev_datetime = to_pd_datetime(time_series_prev.index)

        # convert to vector feature
        prev_timestamp = time_features(prev_datetime).transpose(1, 0)

        if self.config.start_token_len != 0:
            pred_datetime = prev_datetime[self.config.start_token_len :].append(pred_datetime)
        future_timestamp = time_features(pred_datetime).transpose(1, 0)

        # preparing data
        past = np.expand_dims(time_series_prev.values, 0)
        past_timestamp = np.expand_dims(prev_timestamp, 0)
        future_timestamp = np.expand_dims(future_timestamp, 0)

        self.deep_model.eval()
        batch = (past, past_timestamp, None, future_timestamp)
        _, model_output, _ = self._deep_batch_iter(batch, self.config)

        preds = model_output.detach().cpu().numpy().squeeze()
        pd_pred = pd.DataFrame(preds, index=to_pd_datetime(time_stamps), columns=time_series_prev.columns)

        return pd_pred, None

    def batch_forecast(
        self,
        time_stamps_list: List[List[int]],
        time_series_prev_list: List[TimeSeries],
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Tuple[
        Union[
            Tuple[List[TimeSeries], List[Optional[TimeSeries]]],
            Tuple[List[TimeSeries], List[TimeSeries], List[TimeSeries]],
        ]
    ]:
        """
        Need to do re-implementation
        """
        raise NotImplementedError
