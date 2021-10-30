#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Bayesian online change point detection algorithm.
"""
import copy
from enum import Enum
import logging
from typing import List, Union
import warnings

import numpy as np
import scipy.sparse
from scipy.special import logsumexp
from scipy.stats import norm
from tqdm import tqdm

from merlion.models.anomaly.base import DetectorBase, NoCalibrationDetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.conj_priors import ConjPrior, MVNormInvWishart, BayesianMVLinReg, BayesianLinReg
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class ChangeKind(Enum):
    """
    Enum representing the kinds of changes points we would like to detect.
    Enum values correspond to the Bayesian `ConjPrior` class used to detect each sort of change point.
    """

    Auto = None
    """
    Automatically choose the Bayesian conjugate prior we would like to use.
    """

    LevelShift = MVNormInvWishart
    """
    Model data points with a normal distribution, to detect level shifts.
    """

    TrendChange = BayesianMVLinReg
    """
    Model data points as a linear function of time, to detect trend changes.
    """


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
        if n == 1:
            method = getattr(self.posterior, "posterior_explicit", self.posterior.posterior)
        else:
            method = self.posterior.posterior
        logp_x, updated = method(x, log=True, return_updated=True)
        self.posterior = updated
        self.run_length += n
        # P(r_t = self.run_length + 1, x_{1:t}) = P(r_{t-1} = self.run_length, x_{1:t-1}) * P(x_t) * (1 - self.cp_prior)
        self.logp += sum(logp_x) + n * np.log1p(-self.cp_prior)


class BOCPDConfig(NoCalibrationDetectorConfig):
    _default_threshold = AggregateAlarms(alm_threshold=norm.ppf((1 + 0.5) / 2), min_alm_in_window=1)
    """
    Default threshold is for a >=50% probability that a point is a change point.
    """

    def __init__(
        self,
        change_kind: Union[str, ChangeKind] = ChangeKind.Auto,
        cp_prior=1e-2,
        lag=None,
        min_likelihood=1e-8,
        **kwargs,
    ):
        """
        :param change_kind: the kind of change points we would like to detect
        :param cp_prior: prior belief probability of how frequently changepoints occur
        :param lag: the maximum amount of delay/lookback (in number of steps) allowed for detecting change points.
            If ``lag`` is ``None``, we will consider the entire history.
        :param min_likelihood: we will discard any hypotheses whose probability of being a change point is
            lower than this threshold. Lower values improve accuracy at the cost of time and space complexity.
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
    At a high level, this algorithm models the observed data using Bayesian conjugate priors. If an observed value
    deviates too much from the current posterior distribution, it is likely a change point, and we should start
    modeling the time series from that point forwards with a freshly initialized Bayesian conjugate prior.
    """

    config_class = BOCPDConfig

    def __init__(self, config: BOCPDConfig = None):
        config = BOCPDConfig() if config is None else config
        super().__init__(config)
        self.posterior_beam: List[_PosteriorBeam] = []
        self.full_run_length_posterior = scipy.sparse.dok_matrix((0, 0), dtype=float)

    @property
    def n_seen(self):
        """
        :return: the number of data points seen so far
        """
        return self.full_run_length_posterior.get_shape()[0]

    @property
    def change_kind(self) -> ChangeKind:
        """
        :return: the kind of change points we would like to detect
        """
        return self.config.change_kind

    @property
    def cp_prior(self) -> float:
        """
        :return: prior belief probability of how frequently changepoints occur
        """
        return self.config.cp_prior

    @property
    def lag(self) -> int:
        """
        :return: the maximum amount of delay allowed for detecting change points. A higher lag can increase
            recall, but it may decrease precision.
        """
        return self.config.lag

    @property
    def min_likelihood(self) -> float:
        """
        :return: we will not consider any hypotheses (about whether a particular point is a change point)
            with likelihood lower than this threshold
        """
        return self.config.min_likelihood

    def _create_posterior(self, sample, logp: float) -> _PosteriorBeam:
        posterior = self.change_kind.value(sample)
        return _PosteriorBeam(run_length=0, posterior=posterior, cp_prior=self.cp_prior, logp=logp)

    def update(self, time_series: TimeSeries, train=False):
        """
        Updates the BOCPD model's internal state using the time series values provided.

        :param time_series: time series whose values we are using to update the internal state of the model
        :param train: whether we are performing the initial training of the model
        """
        if not train and self.last_train_time is not None:
            _, time_series = time_series.bisect(self.last_train_time, t_in_left=True)

        time_series = time_series.align()
        n_seen = self.n_seen
        T = len(time_series)
        timestamps = []
        full_run_length_posterior = scipy.sparse.dok_matrix((n_seen + T, n_seen + T), dtype=float)
        full_run_length_posterior[:n_seen, :n_seen] = self.full_run_length_posterior
        for i, (t, x) in enumerate(tqdm(time_series)):
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
            self.posterior_beam.append(self._create_posterior(sample=(t, x), logp=cp_logp))

            # P(x_{1:t}) = \sum_{r_t} P(r_t, x_{1:t})
            min_ll = -np.inf if self.min_likelihood is None else np.log(self.min_likelihood)
            evidence = logsumexp([post.logp for post in self.posterior_beam])

            # P(r_t) = P(r_t, x_{1:t}) / P(x_{1:t})
            run_length_dist_0 = {post.run_length: post.logp - evidence for post in self.posterior_beam}

            # Remove posterior beam candidates whose run length probability is too low
            run_length_dist, to_remove = {}, {}
            for r, logp in run_length_dist_0.items():
                if logp < min_ll:
                    to_remove[r] = logp
                else:
                    run_length_dist[r] = logp

            # Re-normalize all remaining probabilities to sum to 1
            self.posterior_beam = [post for post in self.posterior_beam if post.run_length not in to_remove]
            if len(to_remove) > 0:
                excess_p = np.exp(logsumexp(list(to_remove.values())))  # log P[to_remove]
                for post in self.posterior_beam:
                    post.logp -= np.log1p(-excess_p)
                    run_length_dist[post.run_length] -= np.log1p(-excess_p)

            # Update the full posterior distribution of run-length at each time, up to the desired lag
            timestamps.append(t)
            run_length_dist = [(r, logp) for r, logp in run_length_dist.items() if self.lag is None or r <= self.lag]
            if len(run_length_dist) > 0:
                all_r, all_logp_r = zip(*run_length_dist)
                full_run_length_posterior[i + n_seen, all_r] = np.exp(all_logp_r)

        # Compute the MAP probability that each point is a change point.
        # full_run_length_posterior[i, r] = P[run length = r at time t_i]
        probs = np.zeros(T)
        self.full_run_length_posterior = full_run_length_posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_run_length_posterior = scipy.sparse.dia_matrix(full_run_length_posterior)
        for i in range(n_seen + int(train), n_seen + T):
            probs[i - n_seen] = full_run_length_posterior.diagonal(-i).max()

        # Convert P[changepoint] to z-score units
        scores = norm.ppf((1 + probs) / 2)
        self.last_train_time = timestamps[-1]
        return UnivariateTimeSeries(time_stamps=timestamps, values=scores, name="anom_score").to_ts()

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:

        # If automatically detecting the change kind, train candidate models with each change kind
        if self.change_kind is ChangeKind.Auto:
            candidates = []
            for change_kind in ChangeKind:
                if change_kind is ChangeKind.Auto:
                    continue
                candidate = copy.deepcopy(self)
                candidate.config.change_kind = change_kind
                train_scores = candidate.train(train_data, anomaly_labels, train_config, post_rule_train_config)
                log_likelihood = logsumexp([p.logp for p in candidate.posterior_beam])
                candidates.append((candidate, train_scores, log_likelihood))
                logger.info(f"Change kind {change_kind.name} has log likelihood {log_likelihood:.3f}.")

            # Choose the model with the best log likelihood
            i_best = np.argmax([candidate[2] for candidate in candidates])
            best, train_scores, _ = candidates[i_best]
            self.__setstate__(best.__getstate__())
            logger.info(f"Using change kind {self.change_kind.name} because it has the best log likelihood.")

        # Otherwise, just train as normal
        else:
            train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)
            train_scores = self.update(time_series=train_data, train=True)
            self.train_post_rule(train_scores, anomaly_labels, post_rule_train_config)

        # Return the anomaly scores on the training data
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
        if time_series_prev is not None:
            self.update(time_series_prev)
        return self.update(time_series)
