#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Pyspark pandas UDFs for Merlion functions.
This module contains pandas UDFs that can be called via ``pyspark.sql.DataFrame.applyInPandas`` to run Merlion
forecasting, anomaly detection, and time series reconciliation in parallel.
"""
import logging
import traceback
from typing import List, Union

import numpy as np
import pandas as pd

from merlion.models.factory import instantiate_or_copy_model, ModelFactory
from merlion.models.anomaly.base import DetectorBase
from merlion.models.forecast.base import ForecasterBase
from merlion.spark.dataset import TSID_COL_NAME
from merlion.utils import TimeSeries, to_pd_datetime

logger = logging.getLogger(__name__)


def forecast(
    pdf: pd.DataFrame,
    index_cols: List[str],
    time_col: str,
    target_col: str,
    time_stamps: Union[List[int], List[str]],
    model: Union[ForecasterBase, dict],
    predict_on_train: bool = False,
    agg_dict: dict = None,
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
    :param predict_on_train: Whether to return the model's prediction on the training data.
    :param agg_dict: A dictionary used to specify how different data columns should be aggregated. If a non-target
        data column is not in agg_dict, we do not model it for aggregated time series.

    :return: A ``pandas.DataFrame`` with the forecast & its standard error (NaN if the model doesn't have error bars).
        Columns are ``[*index_cols, time_col, target_col, target_col + \"_err\"]``.
    """
    # If the time series has been aggregated, drop non-target columns which are not explicitly specified in agg_dict.
    if TSID_COL_NAME not in index_cols and TSID_COL_NAME in pdf.columns:
        index_cols = index_cols + [TSID_COL_NAME]
    if (pdf.loc[:, index_cols] == "__aggregated__").any().any():
        data_cols = [c for c in pdf.columns if c not in index_cols + [time_col]]
        pdf = pdf.drop(columns=[c for c in data_cols if c != target_col and c not in agg_dict])

    # Sort the dataframe by time & turn it into a Merlion time series
    pdf = pdf.sort_values(by=time_col)
    ts = TimeSeries.from_pd(pdf.drop(columns=index_cols).set_index(time_col))

    # Create model
    model = instantiate_or_copy_model(model or {"name": "DefaultForecaster"})
    if not isinstance(model, ForecasterBase):
        raise TypeError(f"Expected `model` to be an instance of ForecasterBase, but got {model}.")

    # Train model & run forecast
    try:
        train_pred, train_err = model.train(ts)
        pred, err = model.forecast(time_stamps=time_stamps)
    except Exception:
        row0 = pdf.iloc[0]
        idx = ", ".join(f"{k} = {row0[k]}" for k in index_cols)
        logger.warning(
            f"Model {type(model).__name__} threw an exception on ({idx}). Returning the mean training value as a "
            f"placeholder forecast. {traceback.format_exc()}"
        )
        meanval = pdf.loc[:, target_col].mean().item()
        train_err, err = None, None
        train_pred = TimeSeries.from_pd(pd.DataFrame(meanval, index=pdf[time_col], columns=[target_col]))
        pred = TimeSeries.from_pd(pd.DataFrame(meanval, index=to_pd_datetime(time_stamps), columns=[target_col]))

    # Concatenate train & test results if predict_on_train is True
    if predict_on_train:
        if train_pred is not None and pred is not None:
            pred = train_pred + pred
        if train_err is not None and err is not None:
            err = train_err + err

    # Combine forecast & stderr into a single dataframe
    pred = pred.to_pd()
    dtype = pred.dtypes[0]
    err = pd.DataFrame(np.full(len(pred), np.nan), index=pred.index, dtype=dtype) if err is None else err.to_pd()
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
    predict_on_train: bool = False,
) -> pd.DataFrame:
    """
    Pyspark pandas UDF for performing anomaly detection.
    Should be called on a pyspark dataframe grouped by time series ID, i.e. by ``index_cols``.

    :param pdf: The ``pandas.DataFrame`` containing the training and testing data. Should be a single time series.
    :param index_cols: The list of column names used to index all the time series in the dataset. Not used for modeling.
    :param time_col: The name of the column containing the timestamps.
    :param train_test_split: The time at which the testing data starts.
    :param model: The model (or model ``dict``) we are using to predict anomaly scores.
    :param predict_on_train: Whether to return the model's prediction on the training data.

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
    exception = False
    train, test = ts.bisect(train_test_split, t_in_left=False)
    try:
        train_pred = model.train(train)
        train_pred = model.post_rule(train_pred).to_pd()
        pred = model.get_anomaly_label(test).to_pd()
    except Exception:
        exception = True
        row0 = pdf.iloc[0]
        idx = ", ".join(f"{k} = {row0[k]}" for k in index_cols)
        logger.warning(
            f"Model {type(model).__name__} threw an exception on ({idx}). {traceback.format_exc()}"
            f"Trying StatThreshold model instead.\n"
        )
    if exception:
        try:
            model = ModelFactory.create(name="StatThreshold", target_seq_index=0, threshold=model.threshold)
            train_pred = model.train(train)
            train_pred = model.post_rule(train_pred).to_pd()
            pred = model.get_anomaly_label(test).to_pd()
        except Exception:
            logger.warning(
                f"Model StatThreshold threw an exception on ({idx}).{traceback.format_exc()}"
                f"Returning anomaly score = 0 as a placeholder.\n"
            )
            train_pred = pd.DataFrame(0, index=to_pd_datetime(train.time_stamps), columns=["anom_score"])
            pred = pd.DataFrame(0, index=to_pd_datetime(test.time_stamps), columns=["anom_score"])

    if predict_on_train and train_pred is not None:
        pred = pd.concat((train_pred, pred))

    # Turn the time index into a regular column, and add the index columns back to the prediction
    pred.index.name = time_col
    pred.reset_index(inplace=True)
    index_pdf = pd.concat([pdf[index_cols].iloc[:1]] * len(pred), ignore_index=True)
    return pd.concat((index_pdf, pred), axis=1)


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

    .. note::
        Time series series reconciliation is skipped if the given timestamp has missing values for any of the
        time series. This can happen for training timestamps if the training time series has missing data and
        `forecast` is called with ``predict_on_train=true``.
    """
    # Get shape params & sort the data (for this timestamp) by time series ID.
    m, n = hier_matrix.shape
    assert len(pdf) <= m >= n
    if len(pdf) < m:
        return pdf
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
        if nan_errs.any():
            errs[nan_errs] = np.nanmean(errs / coefs) * coefs[nan_errs]
        W = np.diag(errs)

    # Create other supplementary matrices
    J = np.concatenate((np.eye(n), np.zeros((n, m - n))), axis=1)
    U = np.concatenate((-hier_matrix[n:], np.eye(m - n)), axis=1)  # U.T from the paper

    # Compute projection matrix to compute coherent leaf forecasts
    inv = np.linalg.pinv(U @ W @ U.T)  # (m-n) by (m-n)
    P = J - ((J @ W) @ U.T) @ (inv @ U)  # n by m

    # Compute reconciled forecasts & errors
    rec = hier_matrix @ (P @ pdf[target_col].values)
    if nan_errs.all():
        rec_errs = errs
    else:
        # P * W.diagonal() is a faster way to compute P @ W, since W is diagonal
        rec_errs = hier_matrix @ (P * W.diagonal())  # m by m
        # np.sum(rec_errs ** 2, axis=1) is a faster way to compute (rec_errs @ rec_errs.T).diagonal()
        rec_errs = np.sqrt(np.sum(rec_errs**2, axis=1))

    # Replace original forecasts & errors with reconciled ones
    reconciled = pd.DataFrame(np.stack([rec, rec_errs], axis=1), index=pdf.index, columns=[target_col, errname])
    df = pd.concat((pdf.drop(columns=[target_col, errname]), reconciled), axis=1)
    return df
