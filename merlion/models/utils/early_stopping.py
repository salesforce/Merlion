#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Earlying Stopping  
"""
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

import numpy as np


logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping for deep model training
    """

    def __init__(self, patience=7, delta=0):
        """
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        """

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_state_dict = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_best_state_and_dict(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best_state_and_dict(val_loss, model)
            self.counter = 0

    def save_best_state_and_dict(self, val_loss, model):
        self.best_model_state_dict = model.state_dict()

        self.val_loss_min = val_loss

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state_dict)
