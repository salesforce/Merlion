#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
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
    # Sort the dataframe by time & turn it into a Merlion time series
    if TSID_COL_NAME not in index_cols:
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
    # Sort the dataframe by time & turn it into a Merlion time series
    if TSID_COL_NAME not in index_cols:
        index_cols = index_cols + [TSID_COL_NAME]
    pdf = pdf.sort_values(by=time_col)
    ts = TimeSeries.from_pd(pdf.drop(columns=index_cols).set_index(time_col))

    # Create model
    model = instantiate_or_copy_model(model or {"name": "DefaultDetector"})
    if not isinstance(model, DetectorBase):
        raise TypeError(f"Expected `model` to be an instance of DetectorBase, but got {model}.")

    # Train model & run inference
    train, test = ts.bisect(train_test_split)
    model.train(train)
    pred_pdf = model.get_anomaly_label(test).to_pd()

    # Turn the time index into a regular column, and add the index columns back to the prediction
    pred_pdf.index.name = time_col
    pred_pdf.reset_index(inplace=True)
    index_pdf = pd.concat([pdf[index_cols].iloc[:1]] * len(pred_pdf), ignore_index=True)
    return pd.concat((index_pdf, pred_pdf), axis=1)


def reconciliation(pdf: pd.DataFrame, hier_matrix: np.ndarray, target_col: str):
    # Get shape params & sort the data (for this timestamp) by time series ID
    m, n = hier_matrix.shape
    assert len(pdf) == m
    pdf = pdf.sort_values(by=TSID_COL_NAME)

    # Compute the error weight matrix W (m by m)
    errname = f"{target_col}_err"
    coefs = hier_matrix.sum(axis=1)
    errs = pdf[errname].values
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
        rec_errs = np.full(m, np.nan)
    else:
        # P * W.diagonal() is a faster way to compute P @ W, since W is diagonal
        rec_errs = hier_matrix @ (P * W.diagonal())  # m by m
        # np.sum(rec_errs ** 2, axis=1) is a faster way to compute (rec_errs @ rec_errs.T).diagonal()
        rec_errs = np.sqrt(np.sum(rec_errs ** 2, axis=1))

    # Replace original forecasts & errors with reconciled ones
    reconciled = pd.DataFrame(np.stack([rec, rec_errs], axis=1), index=pdf.index, columns=[target_col, errname])
    df = pd.concat((pdf.drop(columns=[target_col, errname]), reconciled), axis=1)
    return df
