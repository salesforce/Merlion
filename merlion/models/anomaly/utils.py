#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
try:
    import torch
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

import random
import numpy as np
from torch.utils.data import Dataset


class InputData(Dataset):
    """
    Dataset for Pytorch models
    """

    def __init__(self, time_series, k, window_based=False):
        self.k = k
        self.window_based = window_based
        self.time_series = time_series
        assert k >= 1

        n = time_series.shape[0]
        if not window_based:
            self.num_samples = n - k + 1
        else:
            self.num_samples = n // k

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not self.window_based:
            return self.time_series[index : index + self.k, :]
        else:
            i = index * self.k
            return self.time_series[i : i + self.k, :]

    @staticmethod
    def collate_func(samples, shuffle=False):
        if shuffle:
            random.shuffle(samples)
        examples = np.stack(samples, axis=0)
        return torch.FloatTensor(examples)


def batch_detect(model, data: np.ndarray, batch_size: int = 1000):
    """
    :param model: A anomaly detection model which implements 'detect' and 'get_sequence_len'
    :param data: The test data
    :param batch_size: The prediction batch size
    """
    scores, k = [], model._get_sequence_len()
    for index in range(0, len(data), batch_size):
        a = max(index - k + 1, 0)
        b = min(index + batch_size, len(data))
        s = model._detect(data[a:b])
        scores.append(s[-(b - index) :])
    scores = np.concatenate(scores, axis=0)
    assert len(scores) == len(data)
    return scores
