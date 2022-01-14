#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""Mixture of Expert forecasters."""
__author__ = "Devansh Arpit"

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from merlion.models.base import NormalizingConfig
from merlion.models.ensemble.base import EnsembleConfig, EnsembleTrainConfig, EnsembleBase
from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.models.ensemble.MoE_networks import TransformerModel, LRScheduler

logger = logging.getLogger(__name__)

########################## Helper functions ##########################


class myDataset(Dataset):
    def __init__(self, data, lookback, forecast=1, target_seq_index=0, include_ts=False):
        """
        Creates a pytorch dataset.

        :param data: TimeSeries object
        :param lookback: number of time steps to lookback in order to forecast
        :param forecast: number of steps to forecast in the future
        :param target_seq_index: dimension of the timeseries that will be forecasted
        :param include_ts: Bool. If True, __getitem__ also returns a TimeSeries version of the data,
            excludes it otherwise

        :meta private:
        """
        self.total_sz = len(data)
        self.include_ts = include_ts

        if include_ts:
            self.data_ts = data
        self.data = data.to_pd().values
        self.timestamps = data.univariates[data.names[0]].time_stamps

        self.lookback = lookback
        self.forecast = forecast
        self.sample_length = lookback + forecast
        self.target_seq_index = target_seq_index

    def __len__(self):
        return self.total_sz - self.sample_length

    def __getitem__(self, idx):
        """
        :return: [idx, pred_timestamps, x, y] or [idx, pred_timestamps, x_ts, x, y]
            idx: scalar index of the time series data treated as the start if the lookback window
            pred_timestamps: list of timestamps of the target that needs to be forecasted
            x_ts: TimeSeries object; window of time series of size lookback extracted from data starting at index idx
            x: same as x_ts but of type numpy array
            y: 1D numpy array; target vector of length forecast extracted from data dimension target_seq_index

        :note: Calling this from a Pytorch Dataloader will return each of these items as a list of size batch size
        """

        start_idx = idx
        end_idx = idx + self.sample_length

        data = self.data[start_idx:end_idx]
        pred_timestamps_full = self.timestamps[start_idx:end_idx]
        pred_timestamps = pred_timestamps_full[self.lookback :]
        x = data[: self.lookback]
        y = data[self.lookback :, self.target_seq_index]  # - data.univariates[data.names[0]][self.lookback-1:-1]

        if self.include_ts:
            x_ts = self.data_ts[start_idx : start_idx + self.lookback]
            return idx, pred_timestamps, x_ts, x, y
        return idx, pred_timestamps, x, y


def collate(items):
    """
    :meta private:
    """
    return items


def sorted_preds(preds):
    """
    :meta private:
    """
    if preds == []:
        return []
    s = sorted(range(len(preds)), key=lambda x: preds[x])
    out = [preds[i][1] for i in s]
    return out


def get_expert_output(expert_preds, free_expert_vals):
    """
    :param expert_preds: (B x nexperts x max_forecast_steps) or []
    :param free_expert_vals: (nexperts x max_forecast_steps) -> (1 x nexperts x max_forecast_steps), or None

    :return: expert_vals. Size: (* x nexperts x max_forecast_steps)

    :meta private:
    """

    if free_expert_vals is not None:
        free_expert_vals = free_expert_vals.unsqueeze(0)
        expert_vals = free_expert_vals
    else:
        expert_vals = expert_preds
    return expert_vals


def get_expert_regression_loss(logits, expert_preds, free_expert_vals, targets):
    """
    :param logits: (B x nexperts x max_forecast_steps)
    :param expert_preds: (B x nexperts x max_forecast_steps) or []
    :param free_expert_vals: (nexperts x max_forecast_steps) -> (1 x nexperts x max_forecast_steps), or None
    :param targets: B x max_forecast_steps -> B x 1 x max_forecast_steps

    :return: (cross_entropy_loss, regression_loss, expert_vals). Note: cross_entropy_loss and regression_loss
    are scalar values

    :meta private:
    """
    batch_size = logits.size(0)

    expert_vals = get_expert_output(expert_preds, free_expert_vals)
    nexperts = expert_vals.size(1)
    max_forecast_steps = expert_vals.size(-1)

    targets = targets.unsqueeze(1)
    err_all = torch.abs(expert_vals - targets)
    regression_loss, idx = err_all.min(1)  # regression_loss: B; idx: B
    CE = torch.nn.CrossEntropyLoss()(logits, idx)
    return CE, regression_loss.mean(), expert_vals


def smape_f1_loss(output, std, target, thres=0.1):
    """
    :meta private:
    """
    output = output.reshape(-1)
    std = std.reshape(-1)
    target = target.reshape(-1)
    idx = torch.le((std / (torch.abs(output) + 1e-7)), thres)

    vanilla_smape = 200.0 * torch.mean(torch.abs((target - output)) / (torch.abs(target) + torch.abs(output) + 1e-7))

    if idx.sum() > 0:
        smape_conf = 200.0 * torch.mean(
            torch.abs((target[idx] - output[idx])) / (torch.abs(target[idx]) + torch.abs(output[idx]) + 1e-7)
        )
        recall = idx.type("torch.FloatTensor").sum() / output.size(0)
    else:
        recall = 0
        smape_conf = None
    idx = torch.gt((std / (torch.abs(output) + 1e-7)), thres)
    if idx.sum() > 0:
        smape_notconf = 200.0 * torch.mean(
            torch.abs((target[idx] - output[idx])) / (torch.abs(target[idx]) + torch.abs(output[idx]) + 1e-7)
        )
    else:
        smape_notconf = None
    return vanilla_smape, smape_conf, smape_notconf, 100.0 * recall


########################## End helper functions ##########################


class MoE_ForecasterEnsembleConfig(EnsembleConfig, ForecasterConfig, NormalizingConfig):
    """
    Config class for MoE (mixture of experts) forecaster.
    """

    def __init__(
        self,
        batch_size=128,
        lr=0.0001,
        warmup_steps=100,
        epoch_max=100,
        nfree_experts=0,
        lookback_len=10,
        max_forecast_steps=3,
        target_seq_index=0,
        use_gpu=True,
        **kwargs,
    ):
        """
        :param batch_size: batch_size needed since MoE uses gradient descent based learning,
            training happens over multiple epochs.
        :param lr: learning rate of the Adam optimizer used in MoE training
        :param warmup_steps: number of iterations used to reach lr
        :param epoch_max: number of epochs to train the MoE model
        :param nfree_experts: number of free expert forecast values that are trained using gradient descent
        :param lookback_len: number of past time steps to look at in order to make future forecasts
        :param max_forecast_steps: number of future steps to forecast
        :param target_seq_index: index of time series to forecast. Integer value.
        :param use_gpu: Bool. Use True if GPU available for faster speed.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.batch_size = batch_size
        self.epoch_max = epoch_max
        self.nfree_experts = nfree_experts
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.lookback_len = lookback_len


class MoE_ForecasterEnsemble(EnsembleBase, ForecasterBase):
    """
    Model-based mixture of experts for forecasting.

    The main class functions useful for users are:

        - train: used for training the MoE model (includes training external expert and training MoE model
          parameters)
        - finetune: assuming the train() function has been called once, finetune can be called if the user
          wants to train the MoE model params again for some reason (E.g. different optimization hyper-parameters).
        - forecast: given a time series, returns the forecast values and standard error as a tuple of `TimeSeries`
          objects
        - batch_forecast: same as forecast, but can operate on a batch of input data and outputs list of
          `TimeSeries` objects
        - _forecast: given a time series, returns the forecast values and confidence of all experts
        - _batch_forecast: same as _forecast, but can operate on a batch of input data
        - expert_prediction: this function operates on the output of _batch_forecast to compute a single forecast
          value per input by combining expert predictions using the user specified strategy (see expert_prediction
          function for details)
        - evaluate: mainly for development purpose. This function performs sMAPE evaluation for a given time series
          data
    """

    models: List[ForecasterBase]
    config_class = MoE_ForecasterEnsembleConfig

    _default_train_config = EnsembleTrainConfig(valid_frac=0.5)

    def __init__(
        self, config: MoE_ForecasterEnsembleConfig = None, models: List[ForecasterBase] = None, moe_model=None
    ):
        """
        :param models: list of external expert models (E.g. Sarima, Arima). Can be an empty list if nfree_experts>0 is
            specified.
        :param moe_model: pytorch model that takes torch.tensor input of size (B x lookback_len x input_dim) and
            outputs a  tuple of 2 variables.  The first variable is the logit (pre-softmax) of size
            (B x nexperts x max_forecast_steps). The second variable is None if nfree_experts=0, else has size
            (nfree_experts x max_forecast_steps) which is the forecasted values by nfree_experts number of experts.
        """
        super().__init__(config=config, models=models)
        for model in self.models:
            assert isinstance(model, ForecasterBase), (
                f"Expected all models in {type(self).__name__} to be anomaly "
                f"detectors, but got a {type(model).__name__}."
            )
        self.loss_list = []

        condition1 = self.config.nfree_experts > 0
        condition2 = len(self.models) > 0
        assert (not (condition1 and condition2)) and (condition1 or condition2), (
            f"Number of free experts (nfree_experts={self.config.nfree_experts}) "
            f"and number of external experts (#models={len(self.models)}) cannot be "
            f"greater than 0 at the same time, but one of them must be non-zero."
        )

        self.optimiser = None
        self.lr_sch = None
        self.moe_model = moe_model

    @property
    def moe_model(self):
        return self._moe_model

    @moe_model.setter
    def moe_model(self, moe_model):
        self._moe_model = moe_model
        if self.moe_model is not None:
            if self.optimiser is None:
                self.optimiser = torch.optim.Adam(self.moe_model.parameters(), lr=self.lr, weight_decay=0.00000)
            if self.lr_sch is None:
                self.lr_sch = LRScheduler(lr_i=0.0000, lr_f=self.lr, nsteps=self.warmup_steps, optimizer=self.optimiser)
        else:
            self.optimiser = None
            self.lr_sch = None

    @property
    def nexperts(self):
        return len(self.models)

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def lr(self) -> int:
        return self.config.lr

    @property
    def warmup_steps(self) -> int:
        return self.config.warmup_steps

    @property
    def epoch_max(self) -> int:
        return self.config.epoch_max

    @property
    def nfree_experts(self) -> int:
        return self.config.nfree_experts

    @property
    def use_gpu(self) -> int:
        return self.config.use_gpu

    @property
    def lookback_len(self) -> int:
        return self.config.lookback_len

    def train(
        self, train_data: TimeSeries, train_config: EnsembleTrainConfig = None
    ) -> Tuple[Optional[TimeSeries], Optional[TimeSeries]]:
        if self.use_gpu:
            torch.cuda.empty_cache()
        full_train = self.train_pre_process(train_data, False, False)
        if self.nexperts > 0:
            t0 = min(v.np_time_stamps[0] for v in full_train.univariates)
            tf = max(v.np_time_stamps[-1] for v in full_train.univariates)
            train, valid = full_train.bisect(t0 + (tf - t0) * (1.0 - train_config.valid_frac))
        else:
            valid = full_train

        # store transform mean and std as torch tensors for future use
        self.mn = torch.tensor(np.array(self.transform.transforms[-1].bias)).type(torch.FloatTensor)
        self.std = torch.tensor(np.array(self.transform.transforms[-1].scale)).type(torch.FloatTensor)
        if self.use_gpu:
            self.mn, self.std = self.mn.cuda(), self.std.cuda()
        # mn and std have dim = (timeseries input dim)

        ## Make sure Pytorch model and optimizer are properly defined
        if self.moe_model is None:
            self.moe_model = TransformerModel(
                input_dim=train_data.dim,
                lookback_len=self.lookback_len,
                nexperts=self.nexperts,
                output_dim=self.max_forecast_steps,
                nfree_experts=self.nfree_experts,
                hid_dim=256,
                dim_head=2,
                mlp_dim=256,
                pool="cls",
                dim_dropout=0,
                time_step_dropout=0,
            )
            self.optimiser = torch.optim.Adam(self.moe_model.parameters(), lr=self.lr, weight_decay=0.00000)
            self.lr_sch = LRScheduler(lr_i=0.0000, lr_f=self.lr, nsteps=self.warmup_steps, optimizer=self.optimiser)
        ## End of: make sure Pytorch model and optimizer are properly defined

        # Train individual models on the training data
        preds = []
        if len(self.models) > 0:
            for i, model in enumerate(self.models):
                logger.info(f"Training model {i+1}/{len(self.models)}...")
                try:
                    preds.append(model.train(train)[0])  # train_config=train_config
                except TypeError as e:
                    if "'NoneType' object is not subscriptable" in str(e):
                        raise RuntimeError(
                            f"train() method of {type(model).__name__} model "
                            f"does not return its fitted predictions for the "
                            f"training data. Therefore, this model cannot be "
                            f"used in a forecaster ensemble."
                        )
                    else:
                        raise e

        # Train the combiner on the validation data
        if self.use_gpu:
            self.moe_model = self.moe_model.cuda()

        _ = self.finetune(valid)

        return None, None

    def _extract_inidividual_model_feat(self, train_data: TimeSeries):
        """
        This function extracts the forecasts of each expert model provided for efficient training of MoE model
        because expert model forecasting on-the-fly is slow. Specifically, for a given train_data time series,
        it creates chunks of samples of (time_stamps, time_series_prev) that external expert models like Sarima use in
        their forecast function. This function then stores the forecasts of these experts in a list such that the i'th
        index of this list corresponds to the ith index of the dataloader object.

        :return: List (len = num of experts) of Lists. Each inner list contains tuple of predictions for full train_data
            for that expert.
        """
        logger.info(f"Extracting and storing expert predictions")

        dataset = myDataset(
            train_data,
            lookback=self.lookback_len,
            forecast=self.max_forecast_steps,
            include_ts=True,
            target_seq_index=self.target_seq_index,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate
        )

        pbar = tqdm(dataloader)
        expert_preds = [[] for _ in range(len(self.models))]
        # List (len = num of experts) of Lists. Each inner list contains tuples (idx, pred)

        for d in pbar:
            sample_idx, pred_timestamps, x_ts, x, y = list(zip(*d))
            sample_idx = list(sample_idx)

            # get model forecasts and errors
            for i, model in enumerate(self.models):
                logger.info(f"Getting model {i+1}/{len(self.models)} predictions...")
                try:
                    pred, _ = model.batch_forecast(pred_timestamps, x_ts)
                    pred = [(idx, pred_i.to_pd().values) for idx, pred_i in zip(sample_idx, pred)]
                    expert_preds[i].extend(pred)
                except TypeError as e:
                    if "'NoneType' object is not subscriptable" in str(e):
                        raise RuntimeError(
                            f"train() method of {type(model).__name__} model "
                            f"does not return its fitted predictions for the "
                            f"training data. Therefore, this model cannot be "
                            f"used in a forecaster ensemble."
                        )
                    else:
                        raise e

        final_expert_preds = []
        """  
        List (len = num of experts) of Lists. Each inner list contains tuple of preds for full train_data for 
            that expert. The sorted_preds function simply rearranges the list of preds so that the i'th index
            of final_expert_preds corresponds to the prediction of the i'th index of the data loader
        """
        for i, expert_preds_i in enumerate(expert_preds):
            final_expert_preds.append(sorted_preds(expert_preds_i))
        return final_expert_preds

    def finetune(self, train_data: TimeSeries, train_config: EnsembleTrainConfig = None):
        """
        This function expects the external experts to be already trained. This function extracts the predictions
        of external experts (if any) and stores them. It then uses them along with the training data to train the
        MoE model to perform expert selection and forecasting. This function is called internally by the train
        function.
        """
        self.moe_model.train()  # pytorch train mode
        final_expert_preds = []
        if len(self.models) > 0:
            final_expert_preds = self._extract_inidividual_model_feat(train_data)

        dataset = myDataset(
            train_data,
            lookback=self.lookback_len,
            forecast=self.max_forecast_steps,
            include_ts=False,
            target_seq_index=self.target_seq_index,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate
        )
        optimiser = self.optimiser
        lr_sch = self.lr_sch

        epoch = 0
        self.loss_list = []
        while epoch < self.epoch_max:
            epoch += 1
            pbar = tqdm(dataloader)
            loss_epoch = []
            for d in pbar:
                sample_idx, pred_timestamps, x, y = list(zip(*d))
                x = torch.tensor(np.array(x)).type(torch.FloatTensor)  # B x lookback_len x dim
                y = torch.tensor(np.array(y)).type(torch.FloatTensor)  # B x max_forecast_steps

                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()

                if self.nfree_experts > 0:
                    y = y - x[:, -1, self.target_seq_index].unsqueeze(-1)  # make y the increment from last x time step

                sample_idx = list(sample_idx)

                # get expert model forecasts
                expert_preds = []
                if len(self.models) > 0:
                    for i in range(len(self.models)):
                        minibatch_expert_preds = [final_expert_preds[i][idx] for idx in sample_idx]
                        expert_preds.append(minibatch_expert_preds)

                    expert_preds = (
                        torch.tensor(np.stack(expert_preds)).permute(1, 0, 2, 3).squeeze(3).type(torch.FloatTensor)
                    )
                    # expert_preds: B x nexperts x max_forecast_steps
                    if self.use_gpu:
                        expert_preds = expert_preds.cuda()

                logits, free_expert_vals = self.moe_model(x)

                # expert_preds is [] or torch.tensor; free_expert_vals is None or torch.tensor
                CE, regression_loss, _ = get_expert_regression_loss(logits, expert_preds, free_expert_vals, y)
                loss = CE + regression_loss

                loss_epoch.append(loss.item())
                pbar.set_description("Epoch {} Loss: {:.6f}".format(epoch, sum(loss_epoch) / len(loss_epoch)))

                optimiser.zero_grad()
                loss.backward()
                lr_sch.step()
                optimiser.step()
            self.loss_list.append(sum(loss_epoch) / len(loss_epoch))
        return self.loss_list

    def _get_expert_prediction(self, time_stamps: List[int], time_series_prev: TimeSeries, use_gpu=False):
        """
        this function is used by the batch_forecast function of this class

        returns: predictions of all the external expert models combined into 1 torch.tensor
            of shape (B=1 x nexperts x max_forecast_steps)
        """
        expert_preds = []
        if len(self.models) > 0:
            for model in self.models:
                expert_preds.append(model.forecast(time_stamps=time_stamps, time_series_prev=time_series_prev)[0])
            expert_preds = [pred_i.to_pd().values for pred_i in expert_preds]
            expert_preds = torch.tensor(np.stack(expert_preds)).unsqueeze(0).squeeze(3)
            # expert_preds: B=1 x nexperts x max_forecast_steps
            if use_gpu:
                expert_preds = expert_preds.cuda()
        return expert_preds

    def _forecast(
        self,
        time_stamps: List[int],
        time_series_prev: TimeSeries = None,
        apply_transform=True,
        expert_idx=None,
        use_gpu=False,
    ):
        """
        returns:
            tuple of 2 numpy arrays: forecast and prob. Both have size: (nexperts x max_forecast_steps)
            Note invert transforms are applied to forecasts returned by this function
        """

        time_stamps_list = [time_stamps]
        time_series_prev_list = [time_series_prev]
        time_series_prev_array = np.expand_dims(time_series_prev.to_pd().values, axis=0)
        forecast, prob = self._batch_forecast(
            time_stamps_list,
            time_series_prev_array,
            time_series_prev_list,
            apply_transform=apply_transform,
            expert_idx=expert_idx,
            use_gpu=use_gpu,
        )
        # forecast: 1 x nexperts x max_forecast_steps
        # prob: 1 x nexperts x max_forecast_steps
        forecast, prob = forecast[0], prob[0]  # remove the 0th dim
        return forecast, prob

    def _batch_forecast(
        self,
        time_stamps_list: List[List[int]],
        time_series_prev_array,
        time_series_prev_list: List[TimeSeries],
        apply_transform=True,
        expert_idx=None,
        use_gpu=False,
    ) -> Tuple[np.array, np.array]:
        """
        Returns the ensemble's forecast on a batch of timestamps given. Note invert transforms are applied to forecasts
        returned by this function

        :param time_stamps_list: a list of lists of timestamps we wish to forecast for
        :param time_series_prev_list: a list of TimeSeries immediately preceeding the time stamps in time_stamps_list
        :param time_series_prev_array: np array. Same as time_series_prev_list but in a numpy array form of size
                    (B x lookback_len x dim)
        :param apply_transform: bool. Whether or not to apply transform to the inputs. Use False if
            transform has already been applied.
        :return: (array of forecasts, array of probs)

            - ``forecasts`` (np array): the forecast for the timestamps given, of size
              (B x nexperts x max_forecast_steps)
            - ``probs`` (np array): the expert probabilities for each forecast made,
              of size (B x nexperts x max_forecast_steps), sum of probs is 1 along dim 1
        """
        self.moe_model.eval()
        time_series_prev_array = torch.from_numpy(time_series_prev_array).type(torch.FloatTensor)
        if use_gpu:
            self.mn = self.mn.cuda()
            self.std = self.std.cuda()
            self.moe_model = self.moe_model.cuda()
            time_series_prev_array = time_series_prev_array.cuda()
        else:
            self.mn = self.mn.data.cpu()
            self.std = self.std.data.cpu()
            self.moe_model = self.moe_model.cpu()

        scale = self.std[self.target_seq_index]
        bias = self.mn[self.target_seq_index]

        if expert_idx is not None and len(self.models) > 0:
            if apply_transform:
                time_series_prev_list = [self.transform(time_series_prev) for time_series_prev in time_series_prev_list]
            expert_vals = []
            for time_stamps, time_series_prev in zip(time_stamps_list, time_series_prev_list):
                forecast = (
                    self.models[expert_idx]
                    .forecast(time_stamps=time_stamps, time_series_prev=time_series_prev)[0]
                    .to_pd()
                    .values.reshape(-1)
                )
                expert_vals.append(forecast)
            expert_vals = torch.tensor(np.stack(expert_vals)).unsqueeze(1)
            expert_vals = scale * expert_vals + bias
            expert_vals = expert_vals.data.cpu().numpy()
            # expert_vals: B x nexpert=1 x forecast len

            s = forecast.shape[0]

            prob = np.ones((len(time_series_prev_list), 1, s))
            # prob: B x nexpert=1 x forecast len
            return expert_vals, prob

        expert_preds = []
        if len(self.models) > 0:
            if apply_transform:
                time_series_prev_list = [self.transform(time_series_prev) for time_series_prev in time_series_prev_list]
            for time_stamps, time_series_prev in zip(time_stamps_list, time_series_prev_list):
                expert_preds_i = self._get_expert_prediction(time_stamps, time_series_prev)

                if len(expert_preds) == 0:
                    expert_preds = expert_preds_i
                else:
                    expert_preds = torch.cat([expert_preds, expert_preds_i], dim=0)

        x = time_series_prev_array
        if apply_transform:
            x = (x - self.mn.view(1, 1, -1)) / self.std.view(1, 1, -1)
        # x: B x lookback_len x input_dim

        logits, free_expert_vals = self.moe_model(x)
        probs = nn.Softmax(dim=1)(logits).data.cpu().numpy()  # B x nexperts x forecast len

        batch_size = logits.size(0)
        expert_vals = get_expert_output(expert_preds, free_expert_vals)
        # expert_vals: * x nexperts x max_forecast_steps

        if self.nfree_experts > 0:
            # invert y from being the increment from last x time step
            expert_vals = expert_vals + x[:, -1, self.target_seq_index].unsqueeze(-1).unsqueeze(-1)
        # expert_vals: B x nexperts x max_forecast_steps

        expert_vals = expert_vals * scale + bias
        expert_vals = expert_vals.data.cpu().numpy()
        return expert_vals, probs

    def forecast(
        self,
        time_stamps: List[int],
        time_series_prev: TimeSeries = None,
        apply_transform=True,
        return_iqr: bool = False,
        return_prev: bool = False,
        expert_idx=None,
        mode="max",
        use_gpu=False,
    ):
        assert not return_iqr, "ForecasterEnsemble does not support return_iqr=True"

        time_stamps_list = [time_stamps]
        time_series_prev_list = [time_series_prev]
        time_series_prev_array = np.expand_dims(time_series_prev.to_pd().values, axis=0)
        forecast, prob = self._batch_forecast(
            time_stamps_list,
            time_series_prev_array,
            time_series_prev_list,
            apply_transform=apply_transform,
            expert_idx=expert_idx,
            use_gpu=use_gpu,
        )
        # forecast: 1 x nexperts x max_forecast_steps
        # prob: 1 x nexperts x max_forecast_steps
        y_pred, std = self.expert_prediction(forecast, prob, mode=mode, use_gpu=use_gpu)
        se = std / np.sqrt(self.nexperts) if self.nexperts > 0 else std / np.sqrt(self.nfree_experts)
        forecast = UnivariateTimeSeries(name=f"{self.target_name}", time_stamps=time_stamps, values=y_pred[0]).to_ts()
        se = UnivariateTimeSeries(name=f"{self.target_name}_err", time_stamps=time_stamps, values=se[0]).to_ts()
        return forecast, se

    def batch_forecast(
        self,
        time_stamps_list: List[List[int]],
        time_series_prev_list: List[TimeSeries],
        return_iqr: bool = False,
        return_prev: bool = False,
        apply_transform=True,
        expert_idx=None,
        mode="max",
        use_gpu=False,
    ) -> Tuple[List[TimeSeries], List[Optional[TimeSeries]]]:
        """
        Returns the ensemble's forecast on a batch of timestamps given. Note invert transforms are applied to forecasts
        returned by this function

        :param time_stamps_list: a list of lists of timestamps we wish to forecast for
        :param time_series_prev_list: a list of TimeSeries immediately preceeding the time stamps in time_stamps_list
        :param return_iqr: whether to return the inter-quartile range for the
            forecast. Note that not all models support this option.
        :param return_prev: whether to return the forecast for
            ``time_series_prev`` (and its stderr or IQR if relevant), in addition
            to the forecast for ``time_stamps``. Only used if ``time_series_prev``
            is provided.
        :param apply_transform: bool. Whether or not to apply transform to the inputs. Use False
            if transform has already been applied.
        :return: (List of TimeSeries of forecasts, List of TimeSeries of standard errors)

            - ``forecasts`` (np array): the forecast for the timestamps given, of size
              (B x nexperts x max_forecast_steps)

            - ``probs`` (np array): the expert probabilities for each forecast made,
              of size (B x nexperts x max_forecast_steps), sum of probs is 1 along dim 1
        """
        time_series_prev_array = np.stack(
            [time_series_prev.to_pd().values for time_series_prev in time_series_prev_list]
        )
        # time_series_prev_array: numpy array form of size (B x lookback_len x dim)

        expert_vals, probs = self._batch_forecast(
            time_stamps_list,
            time_series_prev_array,
            time_series_prev_list,
            apply_transform=apply_transform,
            expert_idx=expert_idx,
            use_gpu=use_gpu,
        )

        y_pred, std = self.expert_prediction(expert_vals, probs, mode=mode, use_gpu=use_gpu)
        se = std / np.sqrt(self.nexperts) if self.nexperts > 0 else std / np.sqrt(self.nfree_experts)
        forecast = [
            UnivariateTimeSeries(
                name=f"{self.target_name}_{i}", time_stamps=time_stamps_list[i], values=y_pred[i]
            ).to_ts()
            for i in range(y_pred.shape[0])
        ]
        se = [
            UnivariateTimeSeries(
                name=f"{self.target_name}_err_{i}", time_stamps=time_stamps_list[i], values=se[i]
            ).to_ts()
            for i in range(y_pred.shape[0])
        ]

        return forecast, se

    def expert_prediction(self, expert_preds, probs, mode="max", use_gpu=False):
        """
        This function can take the outputs provided by batch_forecast or forecast of this class to get
        the final forecast value and allows the user to choose which strategy to use to combine different experts.

        expert_preds: (B x nexperts x max_forecast_steps) np array
        probs: (B x nexperts x max_forecast_steps) np array
        mode: either mean or max. Max picks the expert with the highest confidence; mean computes the weighted average
        use_gpu: set True if GPU available for faster speed

        Returns:
        y_pred: B x max_forecast_steps
        std: B x max_forecast_steps
        """
        expert_preds = torch.tensor(expert_preds).type(torch.FloatTensor)
        probs = torch.tensor(probs).type(torch.FloatTensor)

        if use_gpu:
            expert_preds = expert_preds.cuda()
            probs = probs.cuda()

        max_forecast_steps = probs.size(2)
        nexperts = probs.size(1)

        mean_pred = (probs * expert_preds).sum(1)  # B x max_forecast_steps

        if mode == "mean":
            y_pred = mean_pred  # B x max_forecast_steps
        elif mode == "max":
            prob_ = probs.permute(1, 0, 2).reshape(probs.size(1), -1)
            expert_vals_ = expert_preds.permute(1, 0, 2).reshape(expert_preds.size(1), -1)
            _, idx = prob_.max(0)
            expert_vals_o = expert_vals_[idx, range(expert_vals_.size(1))]
            y_pred = expert_vals_o.reshape(expert_preds.size(0), expert_preds.size(2))  # B x max_forecast_steps
        std = (probs * (expert_preds - mean_pred.unsqueeze(1)) ** 2).sum(1)  # B x max_forecast_steps

        if not use_gpu:
            y_pred = y_pred.data.numpy()
            std = std.data.numpy()
        return y_pred, std

    def evaluate(
        self, data, mode="mean", expert_idx=None, use_gpu=True, use_batch_forecast=True, bs=64, confidence_thres=0.1
    ):
        """
        this function takes a timeseries data and performs an overall evaluation using sMAPE metric on it.
        This function uses many if-else to satisfy the use_gpu and use_batch_forecast conditions specified by user.

        :param data: TimeSeries object
        :param mode: either mean or max. Max picks the expert with the highest confidence; mean computes the weighted
            average.
        :param expert_idx: if None, MoE uses all the experts provided and uses the 'mode' strategy specified below to
            forecast. If value is int (E.g. 0), MoE only uses the external expert at the corresponding index of the
            expert models provided to MoE to make forecasts.
        :param use_gpu: set True if GPU available for faster speed
        :param use_batch_forecast: set True for higher speed
        :param bs: batch size for to go through data in chunks
        :param confidence_thres: threshold used to determine if MoE output is considered confident or not on a sample.
            MoE confident is calculated as forecast-standard-deviation/abs(forecast value). forecast-standard-deviation
            is the standard deviation of the forecasts made by all the experts.
        """
        if use_gpu:
            torch.cuda.empty_cache()
        bs = bs if use_batch_forecast else 1

        data = self.transform(data)

        myDataset_ = myDataset(
            data,
            lookback=self.lookback_len,
            forecast=self.max_forecast_steps,
            include_ts=True,
            target_seq_index=self.target_seq_index,
        )
        dataloader = torch.utils.data.DataLoader(
            myDataset_, batch_size=bs, shuffle=False, num_workers=0, collate_fn=lambda x: x
        )

        pbar = tqdm(dataloader)
        sMAPE_conf_list, sMAPE_not_conf_list, recall_list, overall_sMAPE_list = ([], [], [], [])

        y_pred_all = None
        y_all = None
        std_all = None

        scale, bias = None, None

        pbar = tqdm(dataloader)
        i = 0
        for d in pbar:
            i += 1
            _, pred_timestamps, x_ts, x, y = list(zip(*d))
            x = np.array(x)
            y = torch.tensor(np.array(y)).type(torch.FloatTensor)

            if use_gpu:
                y = y.cuda()

            if use_batch_forecast:
                forecasts, probs = self._batch_forecast(
                    pred_timestamps, x, x_ts, expert_idx=expert_idx, apply_transform=False, use_gpu=use_gpu
                )
            else:
                forecasts, probs = self._forecast(
                    pred_timestamps[0], x_ts[0], expert_idx=expert_idx, apply_transform=False, use_gpu=use_gpu
                )

                forecasts, probs = np.expand_dims(forecasts, axis=0), np.expand_dims(probs, axis=0)

            # the reason why scale and bias are set here is that self.mn, self.std are properly set
            # in self._batch_forecast
            if scale is None:
                scale = self.std[self.target_seq_index]
                bias = self.mn[self.target_seq_index]
            y = y * scale + bias
            y_pred, std = self.expert_prediction(
                forecasts, probs, mode=mode, use_gpu=use_gpu
            )  # takes np array as input

            if not use_gpu:
                y_pred, std = torch.tensor(y_pred), torch.tensor(std)
                if expert_idx is not None and self.nfree_experts == 0:
                    std = torch.zeros_like(y_pred)

                if y_pred_all is None:
                    y_pred_all = y_pred
                    y_all = y
                    std_all = std
                else:
                    y_pred_all = np.concatenate([y_pred_all, y_pred], axis=0)
                    y_all = np.concatenate([y_all, y], axis=0)
                    std_all = np.concatenate([std_all, std], axis=0)
            else:

                if expert_idx is not None and self.nfree_experts == 0:
                    std = torch.zeros_like(y_pred)

                if y_pred_all is None:
                    y_pred_all = y_pred.data.cpu().numpy()
                    y_all = y.data.cpu().numpy()
                    std_all = std.data.cpu().numpy()
                else:
                    y_pred_all = np.concatenate([y_pred_all, y_pred.data.cpu().numpy()], axis=0)
                    y_all = np.concatenate([y_all, y.data.cpu().numpy()], axis=0)
                    std_all = np.concatenate([std_all, std.data.cpu().numpy()], axis=0)

            vanilla_loss, loss, loss2, recall = smape_f1_loss(y_pred, std, y, confidence_thres)
            overall_sMAPE_list.append(vanilla_loss.item())

            if loss is not None:
                recall_list.append(recall)
                sMAPE_conf_list.append(loss.item())
            if loss2 is not None:
                sMAPE_not_conf_list.append(loss2.item())

            pbar.set_description(
                "sMAPE_conf: {:.3f} sMAPE_not_conf: {:.3f} recall: {:.3f}% | Plain sMAPE {:.3f}".format(
                    sum(sMAPE_conf_list) / (len(sMAPE_conf_list) + 1e-7),
                    sum(sMAPE_not_conf_list) / (len(sMAPE_not_conf_list) + 1e-7),
                    sum(recall_list) / (len(recall_list) + 1e-7),
                    sum(overall_sMAPE_list) / (len(overall_sMAPE_list) + 1e-7),
                )
            )

        sMAPE_conf = sum(sMAPE_conf_list) / (len(sMAPE_conf_list) + 1e-7)
        sMAPE_not_conf = sum(sMAPE_not_conf_list) / (len(sMAPE_not_conf_list) + 1e-7)
        recall = sum(recall_list) / (len(recall_list) + 1e-7)
        overall_sMAPE = sum(overall_sMAPE_list) / (len(overall_sMAPE_list) + 1e-7)
        return (y_pred_all, std_all, y_all, sMAPE_conf, sMAPE_not_conf, recall, overall_sMAPE)

    def save(self, dirname: str, **save_config):
        """
        :param dirname: directory to save the model
        :param save_config: additional configurations (if needed)
        """
        # Save MoE transformer state separately from the rest of the model state
        super().save(dirname, **save_config)
        state = {
            "model_params": self.moe_model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "mean": self.mn,
            "std": self.std,
        }
        with open(dirname + "/torch_params.pth.tar", "wb") as f:
            torch.save(state, f)

    def _save_state(
        self, state_dict: Dict[str, Any], filename: str = None, save_only_used_models=False, **save_config
    ) -> Dict[str, Any]:
        for key in ["train_data", "_moe_model", "mn", "std", "optimiser", "lr_sch"]:
            state_dict.pop(key)
        return super()._save_state(state_dict, filename, save_only_used_models=save_only_used_models, **save_config)

    @classmethod
    def load(cls, dirname: str, **kwargs):
        """
        Note: if a user specified model was used while saving the MoE ensemble, specify argument
        ``moe_model`` when calling the load function with the pytorch model that was used in the original MoE ensemble.
        If ``moe_model`` is not specified, it will be assumed that the default Pytorch network was used. Any
        discrepancy between the saved model state and model used here will raise an error.

        :param dirname: directory to load the model from
        """
        loaded_ensemble = super().load(dirname, **kwargs)

        # Load the MoE model state
        config = loaded_ensemble.config
        if "moe_model" in kwargs:
            loaded_ensemble.moe_model = kwargs["moe_model"]
        else:
            loaded_ensemble.moe_model = TransformerModel(
                input_dim=config.dim,
                lookback_len=config.lookback_len,
                nexperts=loaded_ensemble.nexperts,
                output_dim=config.max_forecast_steps,
                nfree_experts=config.nfree_experts,
                hid_dim=256,
                dim_head=2,
                mlp_dim=256,
                pool="cls",
                dim_dropout=0,
                time_step_dropout=0,
            )

        state = torch.load(dirname + "/torch_params.pth.tar", map_location="cuda:0" if config.use_gpu else "cpu")
        try:
            loaded_ensemble.moe_model.load_state_dict(state["model_params"])
            loaded_ensemble.optimiser.load_state_dict(state["optimiser"])
        except TypeError as e:
            raise RuntimeError(f"Found error while loading parameter states/optimizer states of the moe_model: {e}")
        loaded_ensemble.mn = state["mean"]
        loaded_ensemble.std = state["std"]
        return loaded_ensemble
