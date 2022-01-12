#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Deep Point Anomaly Detector algorithm.
"""
import logging
import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.utils.data as data
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.post_process.threshold import AdaptiveAggregateAlarms
from merlion.transform.moving_average import DifferenceTransform
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


def param_init(module, init="ortho"):
    """
    MLP parameter initialization function

    :meta private:
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):  # or isinstance(m, nn.Linear):
            # print('Update init of ', m)
            if init == "he":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif init == "ortho":
                nn.init.orthogonal_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.0001)
            # n = m.weight.size(1)
            # m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            # print('Update init of ', m)
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MLPNet(nn.Module):
    """
    MLP network architecture

    :meta private:
    """

    def __init__(self, dim_inp=None, dim_out=None, nhiddens=(400, 400, 400), bn=True):
        super().__init__()
        self.dim_inp = dim_inp
        self.layers = nn.ModuleList([])
        for i in range(len(nhiddens)):
            if i == 0:
                layer = nn.Linear(dim_inp, nhiddens[i], bias=False)
            else:
                layer = nn.Linear(nhiddens[i - 1], nhiddens[i], bias=False)
            self.layers.append(layer)
            bn_layer = nn.BatchNorm1d(nhiddens[i]) if bn else nn.Sequential()
            relu_layer = nn.ReLU(inplace=True)
            self.layers.extend([bn_layer, relu_layer])

        fc = nn.Linear(nhiddens[-1], dim_out, bias=True)
        self.layers.append(fc)
        self.net = nn.Sequential(*self.layers)

        self.nhiddens = nhiddens

        param_init(self)

    def forward(self, x, logit=False):
        x = x.view(-1, self.dim_inp)
        x = self.net(x)
        # out = torch.nn.Softmax(dim=1)(x)
        #   if logit:
        #   return x
        return x


def get_dnn_loss_as_anomaly_score(tensor_x, tensor_y, use_cuda=True):
    """
    train an MLP using Adam optimizer for 20 iteration on the training data provided

    :meta private:
    """
    BS = tensor_x.size(0)
    LR = 0.001
    max_epochs = 20

    model = MLPNet(dim_inp=1, dim_out=tensor_y.size(1), nhiddens=[400, 400, 400], bn=True)
    if use_cuda:
        model = model.cuda()
    epoch = 0

    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)

    my_dataset = data.TensorDataset(tensor_x, tensor_y)  # create your datset
    my_dataloader = data.DataLoader(my_dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

    # with tqdm.tqdm(total=max_epochs) as pbar:
    while epoch < max_epochs:
        # pbar.update(1)
        epoch += 1
        for x, y in my_dataloader:
            if use_cuda:
                x, y = x.cuda(), y.cuda()

            y_ = model(x)
            loss = ((y - y_) ** 2).sum(1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for x, y in my_dataloader:
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        y_ = model(x)
        loss = ((y - y_) ** 2).sum(1).view(-1)
    return loss.data.view(-1).cpu().numpy()


def normalize_data(x):
    """
    normalize data to have 0 mean and unit variance

    :meta private:
    """
    mn = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    return (x - mn) / (sd + 1e-7)


class DeepPointAnomalyDetectorConfig(DetectorConfig):
    _default_transform = DifferenceTransform()
    _default_threshold = AdaptiveAggregateAlarms()


class DeepPointAnomalyDetector(DetectorBase):
    """
    Given a time series tuple (time, signal), this algorithm trains an MLP with
    each element in time and corresponding signal as input-taget pair. Once the
    MLP is trained for a few itertions, the loss values at each time is
    regarded as the anomaly score for the corresponding signal. The intuition is
    that DNNs learn global patterns before overfitting local details. Therefore
    any point anomalies in the signal will have high MLP loss. These intuitions
    can be found in:
    Arpit, Devansh, et al. "A closer look at memorization in deep networks." ICML 2017
    Rahaman, Nasim, et al. "On the spectral bias of neural networks." ICML 2019
    """

    config_class = DeepPointAnomalyDetectorConfig

    def __init__(self, config: DeepPointAnomalyDetectorConfig):

        super().__init__(config)
        self.use_cuda = torch.cuda.is_available()  # config.use_cuda

    def _preprocess(self, x):
        """
        Pre-process the data: reshape data to pytorch data format, normalize
        data to have zero mean & unit variance, and convert to torch tensor.
        """
        # convert to numpy and reshape as (len(x), -1), to be compatible with
        # torch data format (BS, dim)
        x = np.asarray(x).reshape(len(x), -1)
        # normalize both time and value to have 0 mean and unit variance
        x = normalize_data(x)
        # convert to torch tensor
        x = torch.Tensor(x)
        return x

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)

        times, train_values = zip(*train_data.align())
        processed_times, train_values = self._preprocess(times), self._preprocess(train_values)
        # train an MLP on with (processed_times, train_values) as (input, target)
        # and report the loss vector corresponding to these points as anomaly scores
        train_scores = get_dnn_loss_as_anomaly_score(processed_times, train_values, use_cuda=self.use_cuda)
        train_scores = TimeSeries({"anom_score": UnivariateTimeSeries(times, train_scores)})
        self.train_post_rule(
            anomaly_scores=train_scores, anomaly_labels=anomaly_labels, post_rule_train_config=post_rule_train_config
        )
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        time_series, _ = self.transform_time_series(time_series, time_series_prev)
        times, values = zip(*time_series.align())
        processed_times, values = self._preprocess(times), self._preprocess(values)

        scores = get_dnn_loss_as_anomaly_score(processed_times, values, use_cuda=self.use_cuda)
        return TimeSeries({"anom_score": UnivariateTimeSeries(times, scores)})
