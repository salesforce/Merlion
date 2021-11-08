#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains all anomaly detection models. Forecaster-based anomaly detection models
may be found in :py:mod:`merlion.models.anomaly.forecast_based`. Change-point detection models may be
found in :py:mod:`merlion.models.anomaly.change_point`.

For anomaly detection, we define an abstract `DetectorBase` class which inherits from `ModelBase` and supports the
following interface, in addition to ``model.save`` and ``DetectorClass.load`` defined for `ModelBase`:

1.  ``model = DetectorClass(config)``

    - initialization with a model-specific config
    - configs contain:

        -   a (potentially trainable) data pre-processing transform from :py:mod:`merlion.transform`;
            note that ``model.transform`` is a property which refers to ``model.config.transform``
        -   **a (potentially trainable) post-processing rule** from :py:mod:`merlion.post_process`;
            note that ``model.post_rule`` is a property which refers to ``model.config.post_rule``.
            In general, this post-rule will have two stages: :py:mod:`calibration <merlion.post_process.calibrate>`
            and :py:mod:`thresholding <merlion.post_process.threshold>`.
        -   booleans ``enable_calibrator`` and ``enable_threshold`` (both defaulting to ``True``) indicating
            whether to enable calibration and thresholding in the post-rule.
        -   model-specific hyperparameters

2.  ``model.get_anomaly_score(time_series, time_series_prev=None)``

    -   returns a time series of anomaly scores for each timestamp in ``time_series``
    -   ``time_series_prev`` (optional): the most recent context, only used for some models. If not provided, the
        training data is used as the context instead.

3.  ``model.get_anomaly_label(time_series, time_series_prev=None)``

    -   returns a time series of post-processed anomaly scores for each timestamp in ``time_series``. These scores
        are calibrated to correspond to z-scores if ``enable_calibrator`` is ``True``, and they have also been filtered
        by a thresholding rule (``model.threshold``) if ``enable_threshold`` is ``True``. ``threshold`` is specified
        manually in the config (though it may be modified by `DetectorBase.train`), .
    -   ``time_series_prev`` (optional): the most recent context, only used for some models. If not provided, the
        training data is used as the context instead.

4.  ``model.train(train_data, anomaly_labels=None, train_config=None, post_rule_train_config=None)``

    -   trains the model on the time series ``train_data``
    -   ``anomaly_labels`` (optional): a time series aligned with ``train_data``, which indicates whether each
        time stamp is anomalous
    -   ``train_config`` (optional): extra configuration describing how the model should be trained (e.g. learning rate
        for the `LSTMDetector`). Not used for all models. Class-level default provided for models which do use it.
    -   ``post_rule_train_config``: extra configuration describing how to train the model's post-rule. Class-level
        default is provided for all models.
    -   returns a time series of anomaly scores produced by the model on ``train_data``.
"""
