#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Bayesian online change point detection algorithm.
"""
from enum import Enum
from typing import List, Union

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from tqdm import tqdm

from merlion.models.anomaly.base import DetectorBase, NoCalibrationDetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.conj_priors import ConjPrior, MVNormInvWishart, BayesianMVLinReg
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries


class ChangeKind(Enum):
    """
    Enum representing the kinds of changes we would like to detect. Values correspond to the Bayesian conjugate
    prior class used to detect that sort of change point.
    """

    LevelShift = MVNormInvWishart
    TrendChange = BayesianMVLinReg


class _PosteriorBeam:
    """
    Utility class to track the posterior beam in the dynamic programming for BOCPD.
    """

    def __init__(self, run_length: int, posterior: ConjPrior, cp_prior: float, logp: float):
        self.run_length: int = run_length
        self.posterior: ConjPrior = posterior
        self.cp_prior = cp_prior
        self.logp = logp  # joint probability P(r_t = self.run_length, x_{1:t})

    def update(self, x):
        # self.logp starts as log P(r_{t-1} = self.run_length, x_{1:t-1})
        n = 1 if isinstance(x, tuple) and len(x) == 2 else len(x)
        # logp_x is log P(x_t)
        logp_x, updated = self.posterior.posterior(x, log=True, return_updated=True)
        self.posterior = updated
        self.run_length += n
        # P(r_t = self.run_length + 1, x_{1:t}) = P(r_{t-1} = self.run_length, x_{1:t-1}) * P(x_t) * (1 - self.cp_prior)
        self.logp += sum(logp_x) + n * np.log1p(-self.cp_prior)


class BOCPDConfig(NoCalibrationDetectorConfig):
    _default_threshold = AggregateAlarms(alm_threshold=3, min_alm_in_window=1)

    def __init__(
        self,
        change_kind: Union[str, ChangeKind] = ChangeKind.TrendChange,
        min_likelihood=1e-4,
        cp_prior=1e-2,
        lag=20,
        **kwargs,
    ):
        """
        :param change_kind: the kind of change points we would like to detect
        :param min_likelihood: we will not consider any hypotheses with likelihood lower than this threshold
        :param cp_prior: prior belief probability of how frequently changepoints occur
        :param lag: the maximum amount of delay allowed for detecting change points
        """
        self.change_kind = change_kind
        self.min_likelihood = min_likelihood
        self.cp_prior = cp_prior  # Kats checks [0.001, 0.002, 0.005, 0.01, 0.02]
        self.lag = lag
        super().__init__(**kwargs)

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"change_kind"}))
        config_dict["change_kind"] = self.change_kind.name
        return config_dict

    @property
    def change_kind(self) -> ChangeKind:
        return self._change_kind

    @change_kind.setter
    def change_kind(self, change_kind: Union[str, ChangeKind]):
        if isinstance(change_kind, str):
            valid = set(ChangeKind.__members__.keys())
            if change_kind not in valid:
                raise KeyError(f"{change_kind} is not a valid change kind. Valid change kinds are: {valid}")
            change_kind = ChangeKind[change_kind]
        self._change_kind = change_kind


class BOCPD(DetectorBase):
    """
    Bayesian online change point detection algorithm described by
    `Adams & MacKay (2007) <https://arxiv.org/abs/0710.3742>`__.
    """

    config_class = BOCPDConfig

    def __init__(self, config: BOCPDConfig = None):
        config = BOCPDConfig() if config is None else config
        super().__init__(config)
        self.posterior_beam: List[_PosteriorBeam] = []

    @property
    def change_kind(self) -> ChangeKind:
        return self.config.change_kind

    @property
    def cp_prior(self) -> float:
        return self.config.cp_prior

    @property
    def min_likelihood(self):
        return self.config.min_likelihood

    @property
    def lag(self):
        return self.config.lag

    def create_posterior(self, sample, logp: float) -> _PosteriorBeam:
        posterior = self.change_kind.value(sample)
        return _PosteriorBeam(run_length=0, posterior=posterior, cp_prior=self.cp_prior, logp=logp)

    def update(self, time_series: TimeSeries, train=False):
        if not train and self.last_train_time is not None:
            _, time_series = time_series.bisect(self.last_train_time, t_in_left=True)

        timestamps, logps = [], []
        for t, x in tqdm(time_series.align()):
            # Update posterior beams
            for post in self.posterior_beam:
                post.update((t, x))

            # Calculate posterior probability that this is change point with
            # P_changepoint = \sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1}) * P(x_t) * cp_prior
            # After the updates, post.logp = log P(r_t, x_{1:t})
            #                              = log P(r_{t-1}, x_1{1:t-1}) + log P(x_t) - log(1 - cp_prior)
            # So we can just add log(cp_prior) - log(1 - cp_prior) to each of the logp's
            if len(self.posterior_beam) == 0:
                cp_logp = 0
            else:
                cp_delta = np.log(self.cp_prior) - np.log1p(-self.cp_prior)
                cp_logp = logsumexp([post.logp + cp_delta for post in self.posterior_beam])
            self.posterior_beam.append(self.create_posterior(sample=(t, x), logp=cp_logp))

            # P(x_{1:t}) = \sum_{r_t} P(r_t, x_{1:t})
            min_ll = -np.inf if self.min_likelihood is None else np.log(self.min_likelihood)
            evidence = logsumexp([post.logp for post in self.posterior_beam])

            # P(r_t) = P(r_t, x_{1:t}) / P(x_{1:t})
            run_length_dist = {post.run_length: post.logp - evidence for post in self.posterior_beam}

            # Remove posterior beam candidates whose run length probability is too low
            # and re-normalize all probabilities to sum to 1
            to_remove = {r: logp for r, logp in run_length_dist.items() if logp < min_ll and r > self.lag}
            if len(to_remove) > 0:
                excess_p = np.exp(logsumexp(list(to_remove.values())))
                for post in self.posterior_beam:
                    post.logp -= np.log1p(-excess_p)
                    run_length_dist[post.run_length] -= np.log1p(-excess_p)
            self.posterior_beam = [post for post in self.posterior_beam if post.run_length not in to_remove]

            # Compute the log probability that t is _not_ a change point.
            timestamps.append(t)
            logp_cp = logsumexp([logp for r, logp in run_length_dist.items() if r < self.lag])
            logps.append(np.log1p(-np.exp(min(logp_cp, -1e-16))))

        # Convert anomaly score to z-score units
        scores = norm.ppf(1 - np.exp(logps) / 2)
        return UnivariateTimeSeries(time_stamps=timestamps, values=scores, name="anom_score").to_ts()

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)
        train_scores = self.update(time_series=train_data, train=True)
        self.train_post_rule(train_scores, anomaly_labels, post_rule_train_config)
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        if time_series_prev is not None:
            self.update(time_series_prev)
        return self.update(time_series)
