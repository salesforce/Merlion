#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains all forecasting models.

For forecasting, we define an abstract base `ForecasterBase` class which inherits from `ModelBase` and supports the
following interface, in addition to ``model.save()`` and ``ForecasterClass.load`` defined for ``ModelBase``:

1. ``model = ForecasterClass(config)``

    -   initialization with a model-specific config (which inherits from `ForecasterConfig`)
    -   configs contain:

        -   a (potentially trainable) data pre-processing transform from :py:mod:`merlion.transform`;
            note that ``model.transform`` is a property which refers to ``model.config.transform``
        -   model-specific hyperparameters
        -   **optionally, a maximum number of steps the model can forecast for**

2. ``model.forecast(time_stamps, time_series_prev=None)``

    - returns the forecast (`TimeSeries`) for future values at the time stamps specified by ``time_stamps``,
      as well as the standard error of that forecast (`TimeSeries`, may be ``None``)
    - if ``time_series_prev`` is specified, it is used as the most recent context. Otherwise, the training data is used

3.  ``model.train(train_data, train_config=None)``

    -   trains the model on the `TimeSeries` ``train_data``
    -   ``train_config`` (optional): extra configuration describing how the model should be trained (e.g. learning rate
        for `LSTM`). Not used for all models. Class-level default provided for models which do use it.
    -   returns the model's prediction ``train_data``, in the same format as if you called `ForecasterBase.forecast`
        on the time stamps of ``train_data``
"""
