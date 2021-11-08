#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from packaging import version
import sys
import unittest

import numpy as np
import scipy

from merlion.utils.conj_priors import BetaBernoulli, NormInvGamma, MVNormInvWishart, BayesianLinReg, BayesianMVLinReg
from merlion.utils.time_series import TimeSeries, UnivariateTimeSeries

logger = logging.getLogger(__name__)


class TestConjugatePriors(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(12345)

    def test_beta_bernoulli(self):
        print()
        logger.info("test_beta_bernoulli\n" + "-" * 80 + "\n")
        for theta in [0.21, 0.5, 0.93]:
            data = np.random.rand(1000) < theta
            theta_hat = (1 + sum(data)) / (len(data) + 2)
            dist_np = BetaBernoulli()
            dist_np.update(data)
            self.assertEqual(dist_np.alpha, 1 + sum(data))
            self.assertEqual(dist_np.beta, 1 + sum(1 - data))
            pred = dist_np.posterior([0, 1], log=False)
            expected = np.asarray([1 - theta_hat, theta_hat])
            self.assertAlmostEqual(np.max(np.abs(pred - expected)), 0, places=6)

            ts = TimeSeries.from_pd(data, freq="MS")
            dist_ts = BetaBernoulli(ts[:30])
            dist_ts.update(ts[30:])
            self.assertEqual(dist_ts.alpha, 1 + sum(data))
            self.assertEqual(dist_ts.beta, 1 + sum(1 - data))
            pred = dist_ts.posterior(TimeSeries.from_pd([0, 1]), log=False)
            self.assertAlmostEqual(np.max(np.abs(pred - expected)), 0, places=6)

    def test_normal(self):
        print()
        logger.info("test_normal\n" + "-" * 80 + "\n")
        mu, sigma = 5, 2
        for n in [10, 100, 1000, 100000]:
            # Generate data
            data = np.random.randn(n) * sigma + mu

            # Univariate model
            dist_uni = NormInvGamma()
            pred_uni, dist_uni = dist_uni.posterior(data[: n // 2], return_updated=True)

            # Multivariate model
            dist_multi = MVNormInvWishart()
            pred_multi, dist_multi = dist_multi.posterior(data[: n // 2], return_updated=True)

            # Make sure univariate & multivariate posteriors agree
            self.assertAlmostEqual(np.max(np.abs(pred_uni - pred_multi)), 0, places=6)

            # Make sure univariate & multivariate posteriors agree after additional udpate
            pred_uni = dist_uni.posterior(data[n // 2 :], log=False)
            pred_multi = dist_multi.posterior(data[n // 2 :], log=False)
            self.assertAlmostEqual(np.max(np.abs(pred_uni - pred_multi)), 0, places=6)

            # Make sure we converge to the right model after enough data
            if n > 5000:
                t = [0, 1, 2, 3, 4, 5]
                xhat_u, sigma_u = dist_uni.forecast(t)
                self.assertAlmostEqual(np.max([np.abs(np.array(x) - mu) for t, x in xhat_u]), 0, delta=0.05)
                self.assertAlmostEqual(np.max([np.abs(np.array(s) - sigma) for t, s in sigma_u]), 0, delta=0.05)
                xhat_m, sigma_m = dist_multi.forecast(t)
                self.assertAlmostEqual(np.max([np.abs(np.array(x) - mu) for t, x in xhat_m]), 0, delta=0.05)
                self.assertAlmostEqual(np.max([np.abs(np.array(s) - sigma) for t, s in sigma_m]), 0, delta=0.05)

    def test_mv_normal(self):
        print()
        logger.info("test_mv_normal\n" + "-" * 80 + "\n")
        n, d = 300000, 20
        mu = np.random.randn(d)
        u = np.random.randn(d, d)
        cov = u.T @ u
        data = TimeSeries.from_pd(np.random.randn(n, d) @ u + mu, freq="1h")
        dist = MVNormInvWishart(data[:5])
        dist.update(data[5:-5])
        dist.posterior(data[-5:])  # make sure we can compute a posterior

        # require low L1 distance between expected mean/cov and true mean/cov
        if version.parse(scipy.__version__) >= version.parse("1.6.0"):
            self.assertAlmostEqual(np.abs(mu - dist.mu_posterior(None).loc).mean(), 0, delta=0.05)
        self.assertAlmostEqual(np.abs(cov - dist.Sigma_posterior(None).mean()).mean(), 0, delta=0.05)

        # Make sure the forecast is also accurate, i.e. the stderr-normalized MSE is close to 1
        xhat, stderr = dist.forecast(data.time_stamps[-50000:])
        zscores = (xhat.to_pd() - data[-50000:].to_pd()) / stderr.to_pd().values
        self.assertAlmostEqual(zscores.pow(2).mean().max(), 1, delta=0.02)

    def test_bayesian_linreg(self):
        print()
        logger.info("test_bayesian_linreg\n" + "-" * 80 + "\n")
        n, sigma = 100000, 1.5
        m, b = np.random.randn(2)
        t = np.linspace(0, 2, 2 * n + 1)
        x = UnivariateTimeSeries.from_pd(m * t + b + np.random.randn(len(t)) * sigma, name="test").to_ts()
        x_train = x[: n + 1]
        x_test = x[n + 1 :]

        # Make sure univariate & multivariate agree when initialized from nothing
        uni = BayesianLinReg()
        uni_posterior, uni = uni.posterior(x_train, return_updated=True)
        multi = BayesianMVLinReg()
        multi_posterior, multi = multi.posterior(x_train, return_updated=True)
        self.assertAlmostEqual(np.abs(uni_posterior - multi_posterior).max(), 0, places=6)

        # Get forecasts for the test split. Make sure that the stderr-normalized MSE is close to 1.
        xhat_u, sigma_u = uni.forecast(x_test.time_stamps)
        zscore_u = (xhat_u.to_pd() - x_test.to_pd()) / sigma_u.to_pd().values
        self.assertAlmostEqual(zscore_u.pow(2).mean().item(), 1, delta=0.01)

        # Validate the multivariate forecasting capability as well.
        xhat_m, sigma_m = multi.forecast(x_test.time_stamps)
        zscore_m = (xhat_m.to_pd() - x_test.to_pd()) / sigma_m.to_pd().values
        self.assertAlmostEqual(zscore_m.pow(2).mean().item(), 1, delta=0.01)

        # Make sure univariate & multivariate agree after an additional update
        uni_posterior = uni.posterior(x_test)
        multi_posterior = multi.posterior(x_test)
        self.assertAlmostEqual(np.abs(uni_posterior - multi_posterior).max(), 0, places=6)

        # Make sure explicit version agrees with naive version (univariate)
        naive_uni = np.concatenate([uni.posterior(x_test[i : i + 1]) for i in range(100)])
        explicit_uni = np.concatenate([uni.posterior_explicit(x_test[i : i + 1]) for i in range(100)])
        self.assertAlmostEqual(np.abs(naive_uni - explicit_uni).max(), 0, places=6)

        # Make sure explicit version agrees with naive version (multivariate)
        naive_multi = np.concatenate([multi.posterior(x_test[i : i + 1]) for i in range(100)])
        explicit_multi = np.concatenate([multi.posterior_explicit(x_test[i : i + 1]) for i in range(100)])
        self.assertAlmostEqual(np.abs(naive_multi - explicit_multi).max(), 0, places=6)

        # Make sure we're accurately estimating the slope & intercept
        mhat, bhat = uni.w_0
        self.assertAlmostEqual(mhat, m, delta=0.02)
        self.assertAlmostEqual(bhat, b, delta=0.01)

    def test_mv_bayesian_linreg(self):
        print()
        logger.info("test_mv_bayesian_linreg\n" + "-" * 80 + "\n")
        n, sigma = 200000, 2
        for d in [2, 3, 4, 5, 10, 20]:
            m, b = np.random.randn(2, d)
            t = np.linspace(0, 2, 2 * n + 1)
            x = m.reshape(1, d) * t.reshape(-1, 1) + b.reshape(1, d) + np.random.randn(len(t), d) * sigma
            x_train = x[: n + 1]
            x_test = x[n + 1 :]

            dist = BayesianMVLinReg()
            dist.update(x_train)
            post = dist.posterior(x_test)  # make sure we can compute a multivariate posterior PDF
            self.assertEqual(post.shape, (n,))

            naive = np.concatenate([dist.posterior(x_test[i : i + 1]) for i in range(100)])
            explicit = np.concatenate([dist.posterior_explicit(x_test[i : i + 1]) for i in range(100)])
            self.assertAlmostEqual(np.abs(naive - explicit).max(), 0, delta=0.01)

            # Make sure we're accurately estimating the slope & intercept after all this data
            mhat, bhat = dist.w_0
            self.assertAlmostEqual(np.abs(mhat - m).max(), 0, delta=0.05)
            self.assertAlmostEqual(np.abs(bhat - b).max(), 0, delta=0.05)

            # Make sure the forecast is also accurate, i.e. the stderr-normalized MSE is close to 1
            xhat, stderr = dist.forecast(np.arange(n + 1, 2 * n + 1))
            zscores = (xhat.to_pd() - x_test) / stderr.to_pd().values
            self.assertAlmostEqual(zscores.pow(2).mean().max(), 1, delta=0.02)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.INFO
    )
    unittest.main()
