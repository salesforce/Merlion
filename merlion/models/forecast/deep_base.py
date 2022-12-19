#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
    Base class for Deep Learning Forecasting Models
"""
import copy
import time

import logging
import numpy as np
import pandas as pd


from scipy.stats import norm
from typing import List, Optional, Tuple, Union

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
from merlion.models.utils.time_features import get_time_features
from merlion.models.utils.early_stopping import EarlyStopping

from merlion.transform.base import TransformBase, Identity
from merlion.transform.factory import TransformFactory
from merlion.utils.misc import initializer, ProgressBar
from merlion.utils.time_series import to_pd_datetime, to_timestamp, TimeSeries, AggregationPolicy, MissingValuePolicy


logger = logging.getLogger(__name__)


class DeepForecasterConfig(DeepConfig, ForecasterConfig):
    """
    Config object used to define a forecaster with deep model
    """

    def __init__(
        self,
        n_past: int,
        start_token_len: int = 0,
        **kwargs,
    ):
        """
        :param n_past: # of past steps used for forecasting future.
        :param start_token_len: Length of start token for deep transformer encoder-decoder based models such as
            ``Autoformer`` or ``informer``. The start token is similar to the special tokens for NLP models (e.g., bos, sep, eos tokens).
            For non-transformer based models, we set the ``start_token_len = 0``.
        """
        super().__init__(
            **kwargs,
        )
        self.n_past = n_past
        self.start_token_len = start_token_len


class DeepForecaster(DeepModelBase, ForecasterBase):
    """
    Base class for a deep forecaster model
    """

    config_class = DeepForecasterConfig

    def __init__(self, config: DeepForecasterConfig):
        super().__init__(config)

    def _get_np_loss_and_prediction(self, eval_dataset: RollingWindowDataset):
        """
        Get numpy prediction and loss with evaluation mode for a given dataset or data

        :param evaluate_data: Evaluation data or dataset

        :return: The numpy prediction of the model and the average loss for the given dataset.
        """
        config = self.config
        self.deep_model.eval()

        all_preds = []
        all_trues = []
        total_loss = []

        for i, batch in enumerate(eval_dataset):
            with torch.no_grad():
                loss, outputs, y_true = self._get_batch_model_loss_and_outputs(batch)
                pred = outputs.detach().cpu().numpy()
                true = y_true.detach().cpu().numpy()

                all_preds.append(pred)
                all_trues.append(true)
                total_loss.append(loss.item())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        return preds, np.average(total_loss)

    @property
    def support_multivariate_output(self) -> bool:
        """
        Deep models support multivariate output by default.
        """
        return True

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        config = self.config

        # creating model before the training
        self._create_model()
        if config.use_gpu:
            self.to_gpu()
        else:
            self.to_cpu()

        total_data = copy.deepcopy(train_data)
        data_size = len(total_data)

        # splitting the data into training and valdation set
        if config.validation_rate > 0:
            training_data = total_data[: int(data_size * (1 - config.validation_rate))]
            validation_data = total_data[int(data_size * (1 - config.validation_rate)) :]
        else:
            # if validation rate is zero, we use all training data as validation
            training_data = copy.deepcopy(total_data)
            validation_data = copy.deepcopy(total_data)

        training_dataset = RollingWindowDataset(
            training_data,
            n_past=config.n_past,
            n_future=config.max_forecast_steps,
            batch_size=config.batch_size,
            target_seq_index=None,  # have to set None, we use config.target_seq_index later in the training, if not this is a bug
            ts_encoding=config.ts_encoding,
            flatten=False,
            shuffle=True,
        )

        validation_dataset = RollingWindowDataset(
            validation_data,
            n_past=config.n_past,
            n_future=config.max_forecast_steps,
            batch_size=config.batch_size,
            target_seq_index=None,
            ts_encoding=config.ts_encoding,
            flatten=False,
            shuffle=False,
        )

        train_steps = len(training_dataset)
        logger.info(f"Training steps each epoch: {train_steps}")

        bar = ProgressBar(total=config.num_epochs)

        if config.early_stop_patience:
            early_stopping = EarlyStopping(patience=config.early_stop_patience)

        # start training
        for epoch in range(config.num_epochs):
            train_loss = []

            self.deep_model.train()
            epoch_time = time.time()

            for i, batch in enumerate(training_dataset):
                self.optimizer.zero_grad()

                loss, _, _ = self._get_batch_model_loss_and_outputs(batch)
                train_loss.append(loss.item())

                loss.backward()
                if config.clip_gradient is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), config.clip_gradient)

                self.optimizer.step()

            train_loss = np.average(train_loss)
            _, val_loss = self._get_np_loss_and_prediction(validation_dataset)

            if bar is not None:
                bar.print(
                    epoch + 1, prefix="", suffix=f"Train Loss: {train_loss: .4f}, Validation Loss: {val_loss: .4f}"
                )

            if config.early_stop_patience:
                early_stopping(val_loss, self.deep_model)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping with {config.early_stop_patience} patience")
                    break

        if config.early_stop_patience:
            early_stopping.load_best_model(self.deep_model)
            logger.info(f"Load the best model with validation loss: {early_stopping.val_loss_min: .4f}")

        logger.info("End of the training loop")

        pred, _ = self._get_np_loss_and_prediction(training_dataset)

        # since the model predicts multiple steps, we concatenate all the first steps together
        return pd.DataFrame(pred[:, 0], columns=total_data.columns), None

    def _get_batch_model_loss_and_outputs(self, batch):
        """
        For loss calculation and output prediction

        :param batch: a batch contains `(past, past_timestamp, future, future_timestamp)` used for calculating loss and model outputs

        :return: calculated loss, deep model outputs and targeted ground truth future
        """
        config = self.config
        device = self.deep_model.device

        past, past_timestamp, future, future_timestamp = batch

        past = torch.tensor(past, dtype=torch.float, device=device)
        future = future if future is None else torch.tensor(future, dtype=torch.float, device=device)

        past_timestamp = torch.tensor(past_timestamp, dtype=torch.float, device=device)
        future_timestamp = torch.tensor(future_timestamp, dtype=torch.float, device=device)

        model_output = self.deep_model(past, past_timestamp, future_timestamp)

        if future is None:
            return None, model_output, None

        if config.target_seq_index is None and self.support_multivariate_output:
            target_future = future[:, -config.max_forecast_steps :].to(device)
        else:
            # choose specific target_seq_index for regression
            target_idx = config.target_seq_index
            target_future = future[:, -config.max_forecast_steps :, target_idx : target_idx + 1].to(device)

        loss = self.loss_fn(model_output, target_future)
        return loss, model_output, target_future

    @property
    def require_even_sampling(self) -> bool:
        return False

    def _forecast(
        self, time_stamps: List[int], time_series_prev: pd.DataFrame, return_prev=False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        # convert to vector feature
        prev_timestamp = get_time_features(time_series_prev.index, self.config.ts_encoding)
        future_timestamp = get_time_features(to_pd_datetime(time_stamps), self.config.ts_encoding)

        # preparing data
        past = np.expand_dims(time_series_prev.values, 0)
        past_timestamp = np.expand_dims(prev_timestamp, 0)
        future_timestamp = np.expand_dims(future_timestamp, 0)

        self.deep_model.eval()
        batch = (past, past_timestamp, None, future_timestamp)
        _, model_output, _ = self._get_batch_model_loss_and_outputs(batch)

        preds = model_output.detach().cpu().numpy().squeeze()
        pd_pred = pd.DataFrame(preds, index=to_pd_datetime(time_stamps), columns=time_series_prev.columns)

        return pd_pred, None
