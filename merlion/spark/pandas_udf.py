#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Pyspark pandas UDFs for Merlion functions.
This module contains pandas UDFs that can be called via ``pyspark.sql.DataFrame.applyInPandas`` to run Merlion
forecasting, anomaly detection, and time series reconciliation in parallel.
"""
from typing import List, Union

import numpy as np
import pandas as pd

from merlion.models.factory import instantiate_or_copy_model
from merlion.models.anomaly.base import DetectorBase
from merlion.models.forecast.base import ForecasterBase
from merlion.spark.dataset import TSID_COL_NAME
from merlion.utils.data_io import TimeSeries


def forecast(
    pdf: pd.DataFrame,
    index_cols: List[str],
    time_col: str,
    target_col: str,
    time_stamps: Union[List[int], List[str]],
    model: Union[ForecasterBase, dict],
) -> pd.DataFrame:
    """
    Pyspark pandas UDF for performing forecasting.
    Should be called on a pyspark dataframe grouped by time series ID, i.e. by ``index_cols``.

    :param pdf: The ``pandas.DataFrame`` containing the training data. Should be a single time series.
    :param index_cols: The list of column names used to index all the time series in the dataset. Not used for modeling.
    :param time_col: The name of the column containing the timestamps.
    :param target_col: The name of the column whose value we wish to forecast.
    :param time_stamps: The timestamps at which we would like to obtain a forecast.
    :param model: The model (or model ``dict``) we are using to obtain a forecast.

    :return: A ``pandas.DataFrame`` with the forecast & its standard error (NaN if the model doesn't have error bars).
        Columns are ``[*index_cols, time_col, target_col, target_col + \"_err\"]``.
    """
    # Sort the dataframe by time & turn it into a Merlion time series
    if TSID_COL_NAME not in index_cols and TSID_COL_NAME in pdf.columns:
        index_cols = index_cols + [TSID_COL_NAME]
    pdf = pdf.sort_values(by=time_col)
    ts = TimeSeries.from_pd(pdf.drop(columns=index_cols).set_index(time_col))

    # Create model
    model = instantiate_or_copy_model(model or {"name": "DefaultForecaster"})
    if not isinstance(model, ForecasterBase):
        raise TypeError(f"Expected `model` to be an instance of ForecasterBase, but got {model}.")

    # Train model
    model.train(ts)

    # Run inference and combine prediction & stderr as a single dataframe.
    pred, err = model.forecast(time_stamps=time_stamps)
    pred = pred.to_pd()
    err = pd.DataFrame(np.full(len(pred), np.nan), index=pred.index) if err is None else err.to_pd()
    pred = pd.DataFrame(pred.iloc[:, 0].rename(target_col))
    err = pd.DataFrame(err.iloc[:, 0].rename(f"{target_col}_err"))
    pred_pdf = pd.concat([pred, err], axis=1)

    # Turn the time index into a regular column, and add the index columns back to the prediction
    pred_pdf.index.name = time_col
    pred_pdf.reset_index(inplace=True)
    index_pdf = pd.concat([pdf[index_cols].iloc[:1]] * len(pred_pdf), ignore_index=True)
    return pd.concat((index_pdf, pred_pdf), axis=1)


def anomaly(
    pdf: pd.DataFrame,
    index_cols: List[str],
    time_col: str,
    train_test_split: Union[int, str],
    model: Union[DetectorBase, dict],
) -> pd.DataFrame:
    """
    Pyspark pandas UDF for performing anomaly detection.
    Should be called on a pyspark dataframe grouped by time series ID, i.e. by ``index_cols``.

    :param pdf: The ``pandas.DataFrame`` containing the training and testing data. Should be a single time series.
    :param index_cols: The list of column names used to index all the time series in the dataset. Not used for modeling.
    :param time_col: The name of the column containing the timestamps.
    :param train_test_split: The time at which the testing data starts.
    :param model: The model (or model ``dict``) we are using to predict anomaly scores.

    :return: A ``pandas.DataFrame`` with the anomaly scores on the test data.
        Columns are ``[*index_cols, time_col, \"anom_score\"]``.
    """
    # Sort the dataframe by time & turn it into a Merlion time series
    if TSID_COL_NAME not in index_cols and TSID_COL_NAME in pdf.columns:
        index_cols = index_cols + [TSID_COL_NAME]
    pdf = pdf.sort_values(by=time_col)
    ts = TimeSeries.from_pd(pdf.drop(columns=index_cols).set_index(time_col))

    # Create model
    model = instantiate_or_copy_model(model or {"name": "DefaultDetector"})
    if not isinstance(model, DetectorBase):
        raise TypeError(f"Expected `model` to be an instance of DetectorBase, but got {model}.")

    # Train model & run inference
    train, test = ts.bisect(train_test_split, t_in_left=False)
    model.train(train)
    pred_pdf = model.get_anomaly_label(test).to_pd()

    # Turn the time index into a regular column, and add the index columns back to the prediction
    pred_pdf.index.name = time_col
    pred_pdf.reset_index(inplace=True)
    index_pdf = pd.concat([pdf[index_cols].iloc[:1]] * len(pred_pdf), ignore_index=True)
    return pd.concat((index_pdf, pred_pdf), axis=1)


def reconciliation(pdf: pd.DataFrame, hier_matrix: np.ndarray, target_col: str):
    """
    Pyspark pandas UDF for computing the minimum-trace hierarchical time series reconciliation, as described by
    `Wickramasuriya et al. 2018 <https://robjhyndman.com/papers/mint.pdf>`__.
    Should be called on a pyspark dataframe grouped by timestamp. Pyspark implementation of
    `merlion.utils.hts.minT_reconciliation`.

    :param pdf: A ``pandas.DataFrame`` containing forecasted values & standard errors from ``m`` time series at a single
        timestamp. Each time series should be indexed by `TSID_COL_NAME`.
        The first ``n`` time series (in order of ID) orrespond to leaves of the hierarchy, while the remaining ``m - n``
        are weighted sums of the first ``n``.
        This dataframe can be produced by calling `forecast` on the dataframe produced by
        `merlion.spark.dataset.create_hier_dataset`.
    :param hier_matrix: A ``m``-by-``n`` matrix describing how the hierarchy is aggregated. The value of the ``k``-th
        time series is ``np.dot(hier_matrix[k], pdf[:n])``. This matrix can be produced by
        `merlion.spark.dataset.create_hier_dataset`.
    :param target_col: The name of the column whose value we wish to forecast.

    :return: A ``pandas.DataFrame`` which replaces the original forecasts & errors with reconciled forecasts & errors.
    """
    # Get shape params & sort the data (for this timestamp) by time series ID
    m, n = hier_matrix.shape
    assert len(pdf) == m >= n
    assert (hier_matrix[:n] == np.eye(n)).all()
    pdf = pdf.sort_values(by=TSID_COL_NAME)

    # Compute the error weight matrix W (m by m)
    errname = f"{target_col}_err"
    coefs = hier_matrix.sum(axis=1)
    errs = pdf[errname].values if errname in pdf.columns else np.full(m, np.nan)
    nan_errs = np.isnan(errs)
    if nan_errs.all():
        W = np.diag(coefs)
    else:
        errs = np.nanmean(errs / coefs) * coefs[nan_errs] if nan_errs.any() else errs
        W = np.diag(errs)

    # Create other supplementary matrices
    J = np.concatenate((np.eye(n), np.zeros((n, m - n))), axis=1)
    U = np.concatenate((-hier_matrix[n:], np.eye(m - n)), axis=1)  # U.T from the paper

    # Compute projection matrix to compute coherent leaf forecasts
    inv = np.linalg.inv(U @ W @ U.T)  # (m-n) by (m-n)
    P = J - ((J @ W) @ U.T) @ (inv @ U)  # n by m

    # Compute reconciled forecasts & errors
    rec = hier_matrix @ (P @ pdf[target_col].values)
    if nan_errs.all():
        rec_errs = errs
    else:
        # P * W.diagonal() is a faster way to compute P @ W, since W is diagonal
        rec_errs = hier_matrix @ (P * W.diagonal())  # m by m
        # np.sum(rec_errs ** 2, axis=1) is a faster way to compute (rec_errs @ rec_errs.T).diagonal()
        rec_errs = np.sqrt(np.sum(rec_errs ** 2, axis=1))

    # Replace original forecasts & errors with reconciled ones
    reconciled = pd.DataFrame(np.stack([rec, rec_errs], axis=1), index=pdf.index, columns=[target_col, errname])
    df = pd.concat((pdf.drop(columns=[target_col, errname]), reconciled), axis=1)
    return df
