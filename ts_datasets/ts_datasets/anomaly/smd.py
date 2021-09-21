#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import sys
import logging
import requests
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from ts_datasets.anomaly.base import TSADBaseDataset

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class SMD(TSADBaseDataset):
    """
    The Server Machine Dataset (SMD) is a new 5-week-long dataset from
    a large Internet company collected and made publicly available.
    It contains data from 28 server machines and each machine is monitored by 33 metrics.
    SMD is divided into training set and testing set of equal size.

    - source: https://github.com/NetManAIOps/OmniAnomaly
    """

    filename = "ServerMachineDataset"
    url = "https://www.dropbox.com/s/x53ph5cru62kv0f/ServerMachineDataset.tar.gz?dl=1"
    valid_subsets = (
        [f"machine-1-{i}" for i in range(1, 9)]
        + [f"machine-2-{i}" for i in range(1, 10)]
        + [f"machine-3-{i}" for i in range(1, 12)]
    )

    def __init__(self, subset="all", rootdir=None):
        super().__init__()
        if subset == "all":
            subset = self.valid_subsets
        elif type(subset) == str:
            assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"
            subset = [subset]

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "smd")

        # Download the SMD dataset if it doesn't exist
        download(_logger, rootdir, SMD.url, SMD.filename)
        for s in subset:
            # Load training/test datasets
            df, metadata = combine_train_test_datasets(
                *SMD._load_data(directory=os.path.join(rootdir, SMD.filename), sequence_name=s)
            )
            self.time_series.append(df)
            self.metadata.append(metadata)

    @staticmethod
    def _load_data(directory, sequence_name):
        with open(os.path.join(directory, f"test/{sequence_name}.txt"), "r") as f:
            test_data = np.genfromtxt(f, dtype=np.float32, delimiter=",")
        with open(os.path.join(directory, f"test_label/{sequence_name}.txt"), "r") as f:
            test_labels = np.genfromtxt(f, dtype=np.float32, delimiter=",")
        with open(os.path.join(directory, f"train/{sequence_name}.txt"), "r") as f:
            train_data = np.genfromtxt(f, dtype=np.float32, delimiter=",")
        return (pd.DataFrame(train_data), pd.DataFrame(test_data), test_labels.astype(int))


def combine_train_test_datasets(train_df, test_df, test_labels):
    train_df.columns = [str(c) for c in train_df.columns]
    test_df.columns = [str(c) for c in test_df.columns]
    df = pd.concat([train_df, test_df]).reset_index()
    if "index" in df:
        df.drop(columns=["index"], inplace=True)
    df.index = pd.to_datetime(df.index * 60, unit="s")
    df.index.rename("timestamp", inplace=True)
    # There are no labels for training examples, so the training labels are set to 0 by default
    # The dataset is only for unsupervised time series anomaly detection
    metadata = pd.DataFrame(
        {
            "trainval": df.index < df.index[train_df.shape[0]],
            "anomaly": np.concatenate([np.zeros(train_df.shape[0], dtype=int), test_labels]),
        },
        index=df.index,
    )
    return df, metadata


def download(logger, datapath, url, filename):
    os.makedirs(datapath, exist_ok=True)
    compressed_file = os.path.join(datapath, f"{filename}.tar.gz")

    # Download the compressed dataset
    if not os.path.exists(compressed_file):
        logger.info("Downloading " + url)
        with requests.get(url, stream=True) as r:
            with open(compressed_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()

    # Uncompress the downloaded tar file
    if not os.path.exists(os.path.join(datapath, "_SUCCESS")):
        logger.info(f"Uncompressing {compressed_file}")
        tar = tarfile.open(compressed_file, "r:gz")
        tar.extractall(path=datapath)
        tar.close()
        Path(os.path.join(datapath, "_SUCCESS")).touch()
