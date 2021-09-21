#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import sys
import csv
import ast
import logging
import pickle
import numpy as np
import pandas as pd
from ts_datasets.anomaly.base import TSADBaseDataset
from ts_datasets.anomaly.smd import download, combine_train_test_datasets

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class SMAP(TSADBaseDataset):
    """
    Soil Moisture Active Passive (SMAP) satellite and Mars Science Laboratory (MSL) rover Datasets.
    SMAP and MSL are two realworld public datasets, which are two real-world datasets expert-labeled by NASA.

    - source: https://github.com/khundman/telemanom
    """

    url = "https://www.dropbox.com/s/uv9ojw353qwzqht/SMAP.tar.gz?dl=1"

    def __init__(self, subset=None, rootdir=None):
        super().__init__()

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "smap")

        # Download the SMAP dataset if it doesn't exist
        download(_logger, rootdir, SMAP.url, "SMAP")
        preprocess(_logger, os.path.join(rootdir, "SMAP"), dataset="SMAP")
        # Load training/test datasets
        df, metadata = combine_train_test_datasets(*load_data(os.path.join(rootdir, "SMAP"), "SMAP"))
        self.time_series.append(df)
        self.metadata.append(metadata)


def preprocess(logger, data_folder, dataset):
    if (
        os.path.exists(os.path.join(data_folder, f"{dataset}_test_label.pkl"))
        and os.path.exists(os.path.join(data_folder, f"{dataset}_train.pkl"))
        and os.path.exists(os.path.join(data_folder, f"{dataset}_test.pkl"))
    ):
        return

    logger.info(f"Preprocessing {dataset}")
    with open(os.path.join(data_folder, "labeled_anomalies.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])

    labels = []
    data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
    for row in data_info:
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=bool)
        for anomaly in anomalies:
            label[anomaly[0] : anomaly[1] + 1] = True
        labels.extend(label)
    labels = np.asarray(labels)
    with open(os.path.join(data_folder, f"{dataset}_test_label.pkl"), "wb") as f:
        pickle.dump(labels, f)

    for category in ["train", "test"]:
        data = []
        for row in data_info:
            data.extend(np.load(os.path.join(data_folder, category, row[0] + ".npy")))
        data = np.asarray(data)
        with open(os.path.join(data_folder, f"{dataset}_{category}.pkl"), "wb") as f:
            pickle.dump(data, f)


def load_data(directory, dataset):
    with open(os.path.join(directory, f"{dataset}_test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(directory, f"{dataset}_test_label.pkl"), "rb") as f:
        test_labels = pickle.load(f)
    with open(os.path.join(directory, f"{dataset}_train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    train_df, test_df = pd.DataFrame(train_data), pd.DataFrame(test_data)
    return train_df, test_df, test_labels.astype(int)
