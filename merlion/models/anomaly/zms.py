#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Multiple z-score model (static thresholding at multiple time scales).
"""
from math import log
from typing import List

import numpy as np

from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.transform.base import Identity
from merlion.transform.moving_average import LagTransform
from merlion.transform.normalize import MeanVarNormalize
from merlion.transform.sequence import TransformSequence, TransformStack
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries, UnivariateTimeSeries


class ZMSConfig(DetectorConfig, NormalizingConfig):
    """
    Configuration class for `ZMS` anomaly detection model. The transform of this config is actually a
    pre-processing step, followed by the desired number of lag transforms, and a final mean/variance
    normalization step. This full transform may be accessed as `ZMSConfig.full_transform`. Note that
    the normalization is inherited from `NormalizingConfig`.
    """

    _default_transform = TemporalResample(trainable_granularity=True)

    def __init__(self, base: int = 2, n_lags: int = None, lag_inflation: float = 1.0, **kwargs):
        r"""
        :param base: The base to use for computing exponentially distant lags.
        :param n_lags: The number of lags to be used. If None, n_lags will be
            chosen later as the maximum number of lags possible for the initial
            training set.
        :param lag_inflation: See math below for the precise mathematical role of
            the lag inflation. Consider the lag inflation a measure of distrust
            toward higher lags, If ``lag_inflation`` > 1, the higher the lag
            inflation, the less likely the model is to select a higher lag's z-score
            as the anomaly score.

        .. math::
            \begin{align*}
            \text{Let } \space z_k(x_t) \text{ be the z-score of the } & k\text{-lag at } t, \space \Delta_k(x_t)
            \text{ and } p \text{ be the lag inflation} \\
            & \\
            \text{the anomaly score    } z(x_t) & =  z_{k^*}(x_t) \\
            \text{where } k^* & = \text{argmax}_k \space | z_k(x_t) | / k^p
            \end{align*}
        """
        assert lag_inflation >= 0.0
        self.base = base
        self.n_lags = n_lags
        self.lag_inflation = lag_inflation
        super().__init__(**kwargs)

    @property
    def full_transform(self):
        """
        Returns the full transform, including the pre-processing step, lags, and
        final mean/variance normalization.
        """
        return TransformSequence([self.transform, self.lags, self.normalize])

    def to_dict(self, _skipped_keys=None):
        # self.lags isn't trainable & is set automatically via n_lags
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        return super().to_dict(_skipped_keys.union({"lags"}))

    @property
    def n_lags(self):
        return self._n_lags

    @n_lags.setter
    def n_lags(self, n: int):
        """
        Set the number of lags. Also resets the mean/var normalization, since
        the output dimension (number of lags) will change.
        """
        self._n_lags = n
        lags = [LagTransform(self.base ** k, pad=True) for k in range(n)] if n is not None else []
        self.lags = TransformStack([Identity(), *lags])
        self.normalize = MeanVarNormalize()


class ZMS(DetectorBase):
    r"""
    Multiple Z-Score based Anomaly Detector.

    ZMS is designed to detect spikes, dips, sharp trend changes (up or down)
    relative to historical data. Anomaly scores capture not only magnitude
    but also direction. This lets one distinguish between positive (spike)
    negative (dip) anomalies for example.

    The algorithm builds models of normalcy at multiple exponentially-growing
    time scales. The zeroth order model is just a model of the values seen
    recently. The kth order model is similar except that it models not
    values, but rather their k-lags, defined as x(t)-x(t-k), for k in
    1, 2, 4, 8, 16, etc. The algorithm assigns the maximum absolute z-score
    of all the models of normalcy as the overall anomaly score.

    .. math::
        \begin{align*}
        \text{Let } \space z_k(x_t) \text{ be the z-score of the } & k\text{-lag at } t, \space \Delta_k(x_t)
        \text{ and } p \text{ be the lag inflation} \\
        & \\
        \text{the anomaly score    } z(x_t) & =  z_{k^*}(x_t) \\
        \text{where } k^* & = \text{argmax}_k \space | z_k(x_t) | / k^p
        \end{align*}
    """
    config_class = ZMSConfig

    @property
    def n_lags(self):
        return self.config.n_lags

    @n_lags.setter
    def n_lags(self, n_lags):
        self.config.n_lags = n_lags

    @property
    def lag_scales(self) -> List[int]:
        return [lag.k for lag in self.config.lags.transforms[1:]]

    @property
    def lag_inflation(self):
        return self.config.lag_inflation

    @property
    def adjust_z_scores(self) -> bool:
        return self.lag_inflation > 0.0 and len(self.lag_scales) > 1

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        if self.n_lags is None:
            self.n_lags = int(log(len(train_data), self.config.base))

        self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)
        train_scores = self.get_anomaly_score(train_data)
        self.train_post_rule(train_scores, anomaly_labels, post_rule_train_config)
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        time_series, _ = self.transform_time_series(time_series, time_series_prev)
        z_scores = time_series.to_pd().values

        if self.adjust_z_scores:
            # choose z-score according to adjusted z-scores
            adjusted_z_scores = np.hstack(
                (z_scores[:, 0:1], z_scores[:, 1:] / (np.asarray(self.lag_scales) ** self.lag_inflation))
            )
            lag_args = np.argmax(adjusted_z_scores, axis=1)
            scores = [z_scores[(i, a)] for i, a in enumerate(lag_args)]
        else:
            scores = np.nanmax(np.abs(z_scores), axis=1)

        return UnivariateTimeSeries(time_stamps=time_series.time_stamps, values=scores, name="anom_score").to_ts()
