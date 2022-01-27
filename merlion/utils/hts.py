#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Aggregation for hierarchical time series.
"""
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd

from merlion.utils.time_series import TimeSeries, to_pd_datetime


def minT_reconciliation(
    forecasts: List[TimeSeries], errs: List[TimeSeries], sum_matrix: np.ndarray, n_leaves: int
) -> List[TimeSeries]:
    """
    Computes the minimum trace reconciliation for hierarchical time series, as described by
    `Wickramasuriya et al. 2018 <https://robjhyndman.com/papers/mint.pdf>`__. This algorithm assumes that
    we have a number of time series aggregated at various levels (the aggregation tree is described by ``sum_matrix``),
    and we obtain independent forecasts at each level of the hierarchy. Minimum trace reconciliation finds the optimal
    way to adjust (reconcile) the forecasts to reduce the variance of the estimation.

    :param forecasts: forecast for each aggregation level of the hierarchy
    :param errs: standard errors of forecasts for each level of the hierarchy. While not strictly necessary,
        reconciliation performs better if all forecasts are accompanied by uncertainty estimates.
    :param sum_matrix: matrix describing how the hierarchy is aggregated
    :param n_leaves: the number of leaf forecasts (i.e. the number of forecasts at the most dis-aggregated level
        of the hierarchy). We assume that the leaf forecasts are last in the lists ``forecasts`` & ``errs``,
        and that ``sum_matrix`` reflects this fact.

    :return: reconciled forecasts for each aggregation level of the hierarchy
    """
    m = len(forecasts)
    n = n_leaves
    assert len(errs) == m > n
    assert all(yhat.dim == 1 for yhat in forecasts)
    assert sum_matrix.shape == (m, n), f"Expected sum_matrix to have shape ({m}, {n}) got {sum_matrix.shape}"
    assert (sum_matrix[-n:] == np.eye(n)).all()

    # Convert forecasts to a single aligned multivariate time series
    names = [yhat.names[0] for yhat in forecasts]
    forecasts = OrderedDict((i, yhat.univariates[yhat.names[0]]) for i, yhat in enumerate(forecasts))
    forecasts = TimeSeries(univariates=forecasts).align()
    t_ref = forecasts.time_stamps
    H = len(forecasts)

    # Matrix of stderrs (if any) at each prediction horizon. shape is [m, H].
    # If no stderrs are given, we the estimation error is proportional to the number of leaf nodes being combined.
    coefs = sum_matrix.sum(axis=1)
    if all(e is None for e in errs):
        # FIXME: This heuristic can be improved if training errors are given.
        #        However, the model code should probably be responsible for this, not the reconciliation code.
        Wh = [np.diag(coefs) for _ in range(H)]
    else:
        coefs = coefs.reshape(-1, 1)
        errs = np.asarray(
            [np.full(H, np.nan) if e is None else e.align(reference=t_ref).to_pd().values.flatten() ** 2 for e in errs]
        )  # [m, H]
        # Replace NaN's w/ the mean of non-NaN stderrs & create diagonal error matrices
        nan_errs = np.isnan(errs[:, 0])
        if nan_errs.any():
            errs[nan_errs] = np.nanmean(errs / coefs, axis=0) * coefs[nan_errs]
        Wh = [np.diag(errs[:, h]) for h in range(H)]

    # Create other supplementary matrices
    J = np.zeros((n, m))
    J[:, -n:] = np.eye(n)
    U = np.zeros((m - n, m))
    U[:, : m - n] = np.eye(m - n)
    U[:, m - n :] = -sum_matrix[:-n]

    # Compute projection matrices to compute coherent leaf forecasts
    Ph = []
    for W in Wh:
        inv = np.linalg.inv(U @ W @ U.T)
        P = J - ((J @ W) @ U.T) @ (inv @ U)
        Ph.append(P)

    # Compute reconciled forecasts
    reconciled = []
    for (t, yhat_h), P in zip(forecasts, Ph):
        reconciled.append(sum_matrix @ (P @ yhat_h))
    reconciled = pd.DataFrame(np.asarray(reconciled), index=to_pd_datetime(t_ref))

    return [u.to_ts(name=name) for u, name in zip(TimeSeries.from_pd(reconciled).univariates, names)]
