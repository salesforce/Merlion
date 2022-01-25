#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Bayesian online change point detection algorithm.
"""
import bisect
import copy
from enum import Enum
import logging
from typing import List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.special import logsumexp
from scipy.stats import norm
from tqdm import tqdm

from merlion.models.anomaly.base import NoCalibrationDetectorConfig
from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.forecast.base import ForecasterConfig
from merlion.plot import Figure
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.conj_priors import ConjPrior, MVNormInvWishart, BayesianMVLinReg
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries, to_pd_datetime

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


class BOCPDConfig(ForecasterConfig, NoCalibrationDetectorConfig):
    """
    Config class for `BOCPD` (Bayesian Online Change Point Detection).
    """

    _default_threshold = AggregateAlarms(alm_threshold=norm.ppf((1 + 0.5) / 2), min_alm_in_window=1)
    """
    Default threshold is for a >=50% probability that a point is a change point.
    """

    def __init__(
        self,
        change_kind: Union[str, ChangeKind] = ChangeKind.Auto,
        cp_prior=1e-2,
        lag=None,
        min_likelihood=1e-12,
        max_forecast_steps=None,
        **kwargs,
    ):
        """
        :param change_kind: the kind of change points we would like to detect
        :param cp_prior: prior belief probability of how frequently changepoints occur
        :param lag: the maximum amount of delay/lookback (in number of steps) allowed for detecting change points.
            If ``lag`` is ``None``, we will consider the entire history. Note: we do not recommend ``lag = 0``.
        :param min_likelihood: we will discard any hypotheses whose probability of being a change point is
            lower than this threshold. Lower values improve accuracy at the cost of time and space complexity.
        :param max_forecast_steps: the maximum number of steps the model is allowed to forecast. Ignored.
        """
        self.change_kind = change_kind
        self.min_likelihood = min_likelihood
        self.cp_prior = cp_prior  # Kats checks [0.001, 0.002, 0.005, 0.01, 0.02]
        self.lag = lag
        super().__init__(max_forecast_steps=max_forecast_steps, **kwargs)

    def to_dict(self, _skipped_keys=None):
        _skipped_keys = _skipped_keys if _skipped_keys is not None else set()
        config_dict = super().to_dict(_skipped_keys.union({"change_kind"}))
        if "change_kind" not in _skipped_keys:
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


class BOCPD(ForecastingDetectorBase):
    """
    Bayesian online change point detection algorithm described by
    `Adams & MacKay (2007) <https://arxiv.org/abs/0710.3742>`__.
    At a high level, this algorithm models the observed data using Bayesian conjugate priors. If an observed value
    deviates too much from the current posterior distribution, it is likely a change point, and we should start
    modeling the time series from that point forwards with a freshly initialized Bayesian conjugate prior.

    The ``get_anomaly_score()`` method returns a z-score corresponding to the probability of each point being
    a change point. The ``forecast()`` method returns the predicted values (and standard error) of the underlying
    piecewise model on the relevant data.
    """

    config_class = BOCPDConfig

    def __init__(self, config: BOCPDConfig = None):
        config = BOCPDConfig() if config is None else config
        super().__init__(config)
        self.posterior_beam: List[_PosteriorBeam] = []
        self.train_timestamps: List[float] = []
        self.full_run_length_posterior = scipy.sparse.dok_matrix((0, 0), dtype=float)
        self.pw_model: List[Tuple[pd.Timestamp, ConjPrior]] = []

    @property
    def last_train_time(self):
        return None if len(self.train_timestamps) == 0 else to_pd_datetime(self.train_timestamps[-1])

    @last_train_time.setter
    def last_train_time(self, t):
        pass

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

    def _create_posterior(self, logp: float) -> _PosteriorBeam:
        posterior = self.change_kind.value()
        return _PosteriorBeam(run_length=0, posterior=posterior, cp_prior=self.cp_prior, logp=logp)

    def _get_anom_scores(self, time_stamps: List[Union[int, float]]) -> TimeSeries:
        # Convert sparse posterior matrix to a form where it's fast to access its diagonals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posterior = scipy.sparse.dia_matrix(self.full_run_length_posterior)

        # Compute the MAP probability that each point is a change point.
        # full_run_length_posterior[i, r] = P[run length = r at time t_i]
        i_0 = bisect.bisect_left(self.train_timestamps, time_stamps[0])
        i_f = bisect.bisect_right(self.train_timestamps, time_stamps[-1])
        probs = np.zeros(i_f - i_0)
        n_lag = None if self.lag is None else self.lag + 1
        for i_prob, i_posterior in enumerate(range(max(i_0, 1), i_f)):
            probs[i_prob] = posterior.diagonal(-i_posterior)[:n_lag].max()

        # Convert P[changepoint] to z-score units, and align it to the right time stamps
        scores = norm.ppf((1 + probs) / 2)
        ts = UnivariateTimeSeries(time_stamps=self.train_timestamps[i_0:i_f], values=scores, name="anom_score").to_ts()
        return ts.align(reference=time_stamps)

    def _update_model(self, timestamps):
        # Figure out where the changepoints are in the data
        changepoints = self.threshold.to_simple_threshold()(self._get_anom_scores(timestamps))
        changepoints = changepoints.to_pd().iloc[:, 0]
        cp_times = changepoints[changepoints != 0].index

        # Remove every sub-model that takes effect after the first timestamp provided.
        self.pw_model = [(t0, model) for t0, model in self.pw_model if t0 < changepoints.index[0]]

        # Update the final piece of the existing model (if there is one)
        t0 = changepoints.index[0] if len(self.pw_model) == 0 else self.pw_model[-1][0]
        tf = changepoints.index[-1] if len(cp_times) == 0 else cp_times[0]
        train_data = self.transform(self.train_data)
        data = train_data.window(t0, tf, include_tf=len(cp_times) == 0)
        if len(data) > 0:
            if len(self.pw_model) == 0:
                self.pw_model.append((t0, self.change_kind.value(data)))
            else:
                self.pw_model[-1] = (t0, self.change_kind.value(data))

        # Build a piecewise model by using the data between each subsequent change point
        t0 = tf
        for tf in cp_times[1:]:
            data = train_data.window(t0, tf)
            if len(data) > 0:
                self.pw_model.append((t0, self.change_kind.value(data)))
                t0 = tf
        if t0 < changepoints.index[-1]:
            _, data = train_data.bisect(t0, t_in_left=False)
            self.pw_model.append((t0, self.change_kind.value(data)))

    def train_pre_process(
        self, train_data: TimeSeries, require_even_sampling: bool, require_univariate: bool
    ) -> TimeSeries:
        # BOCPD doesn't _require_ target_seq_index to be specified, but train_pre_process() does.
        if self.target_seq_index is None and train_data.dim > 1:
            self.config.target_seq_index = 0
            logger.warning(
                f"Received a {train_data.dim}-variate time series, but `target_seq_index` was not "
                f"specified. Setting `target_seq_index = 0` so the `forecast()` method will work."
            )
        train_data = super().train_pre_process(train_data, require_even_sampling, require_univariate)
        # We manually update self.train_data in update(), so do nothing here
        self.train_data = None
        return train_data

    def forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ) -> Union[Tuple[TimeSeries, TimeSeries], Tuple[TimeSeries, TimeSeries, TimeSeries]]:
        if time_series_prev is not None:
            self.update(time_series_prev)
        if isinstance(time_stamps, (int, float)):
            time_stamps = pd.date_range(start=self.last_train_time, freq=self.timedelta, periods=int(time_stamps))[1:]
        else:
            time_stamps = to_pd_datetime(time_stamps)
        if return_prev and time_series_prev is not None:
            time_stamps = to_pd_datetime(time_series_prev.time_stamps).union(time_stamps)

        # Initialize output accumulators
        pred_full, err_full = None, None

        # Split the time stamps based on which model piece should be used
        j = 0
        i = bisect.bisect_left([t0 for t0, model in self.pw_model], time_stamps[j], hi=len(self.pw_model) - 1)
        for i, (t0, posterior) in enumerate(self.pw_model[i:], i):
            # Stop forecasting if we've finished with all the input timestamps
            if j >= len(time_stamps):
                break

            # If this is the last piece, use it to forecast the rest of the timestamps
            if i == len(self.pw_model) - 1:
                pred, err = posterior.forecast(time_stamps[j:])

            # Otherwise, predict until the next piece takes over
            else:
                t_next = self.pw_model[i + 1][0]
                j_next = bisect.bisect_left(time_stamps, t_next)
                pred, err = posterior.forecast(time_stamps[j:j_next])
                j = j_next

            # Accumulate results
            pred_full = pred if pred_full is None else pred_full + pred
            err_full = err if err_full is None else err_full + err

        pred = pred_full.univariates[pred_full.names[self.target_seq_index]].to_pd()
        err = err_full.univariates[err_full.names[self.target_seq_index]].to_pd()
        pred[pred.isna() | np.isinf(pred)] = 0
        err[err.isna() | np.isinf(err)] = 0

        if return_iqr:
            name = pred.name
            lb = UnivariateTimeSeries.from_pd(pred + norm.ppf(0.25) * err, name=f"{name}_lower")
            ub = UnivariateTimeSeries.from_pd(pred + norm.ppf(0.75) * err, name=f"{name}_upper")
            return TimeSeries.from_pd(pred), lb.to_ts(), ub.to_ts()
        return TimeSeries.from_pd(pred), TimeSeries.from_pd(err)

    def get_figure(
        self,
        *,
        time_series: TimeSeries = None,
        time_stamps: List[int] = None,
        time_series_prev: TimeSeries = None,
        plot_anomaly=True,
        filter_scores=True,
        plot_forecast=False,
        plot_forecast_uncertainty=False,
        plot_time_series_prev=False,
    ) -> Figure:
        if time_series is not None:
            self.update(time_series)
        return super().get_figure(
            time_series=time_series,
            time_stamps=time_stamps,
            time_series_prev=time_series_prev,
            plot_anomaly=plot_anomaly,
            filter_scores=filter_scores,
            plot_forecast=plot_forecast,
            plot_forecast_uncertainty=plot_forecast_uncertainty,
            plot_time_series_prev=plot_time_series_prev,
        )

    def update(self, time_series: TimeSeries):
        """
        Updates the BOCPD model's internal state using the time series values provided.

        :param time_series: time series whose values we are using to update the internal state of the model
        :return: anomaly score associated with each point (based on the probability of it being a change point)
        """
        # Only update on the portion of the time series after the last training timestamp
        time_stamps = time_series.time_stamps
        if self.last_train_time is not None:
            time_series_prev, time_series = time_series.bisect(self.last_train_time, t_in_left=True)
        else:
            time_series_prev = None

        # Update the training data accumulated so far
        if self.train_data is None:
            self.train_data = time_series
        else:
            self.train_data = self.train_data + time_series

        # Apply any pre-processing transforms to the time series
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)

        # Align the time series & expand the array storing the full posterior distribution of run lengths
        time_series = time_series.align()
        n_seen, T = self.n_seen, len(time_series)
        self.full_run_length_posterior = scipy.sparse.block_diag(
            (self.full_run_length_posterior, scipy.sparse.dok_matrix((T, T), dtype=float)), format="dok"
        )

        # Compute the minimum log likelihood threshold that we consider.
        min_ll = -np.inf if self.min_likelihood is None or self.min_likelihood <= 0 else np.log(self.min_likelihood)
        if self.change_kind is ChangeKind.TrendChange:
            min_ll = min_ll * time_series.dim
        min_ll = min_ll + np.log(self.cp_prior)

        # Iterate over the time series
        for i, (t, x) in enumerate(tqdm(time_series, desc="BOCPD Update", disable=(T == 0))):
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
            self.posterior_beam.append(self._create_posterior(logp=cp_logp))

            # P(x_{1:t}) = \sum_{r_t} P(r_t, x_{1:t})
            evidence = logsumexp([post.logp for post in self.posterior_beam])

            # P(r_t) = P(r_t, x_{1:t}) / P(x_{1:t})
            run_length_dist_0 = {post.run_length: post.logp - evidence for post in self.posterior_beam}

            # Remove posterior beam candidates whose run length probability is too low
            run_length_dist, to_remove = {}, {}
            for r, logp in run_length_dist_0.items():
                if logp < min_ll and r > 2:  # allow at least 2 updates for each change point hypothesis
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
            run_length_dist = [(r, logp) for r, logp in run_length_dist.items()]
            if len(run_length_dist) > 0:
                all_r, all_logp_r = zip(*run_length_dist)
                self.full_run_length_posterior[n_seen + i, all_r] = np.exp(all_logp_r)

            # Add this timestamp to the list of timestamps we've trained on
            self.train_timestamps.append(t)

        # Update the predictive model if there is any new data
        if len(time_series) > 0:
            if self.lag is None:
                n = len(self.train_timestamps)
            else:
                n = T + self.lag
            self._update_model(self.train_timestamps[-n:])

        # Return the anomaly scores
        return self._get_anom_scores(time_stamps)

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:

        # If automatically detecting the change kind, train candidate models with each change kind
        # TODO: abstract this logic into a merlion.models.automl.GridSearch object?
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
            self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)
            train_scores = self.update(time_series=train_data)
            self.train_post_rule(train_scores, anomaly_labels, post_rule_train_config)

        # Return the anomaly scores on the training data
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        if time_series_prev is not None:
            self.update(time_series_prev)
        return self.update(time_series)
