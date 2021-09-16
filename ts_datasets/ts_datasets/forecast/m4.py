#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import os
import requests
from tqdm import tqdm
import pandas as pd

from ts_datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class M4(BaseDataset):
    """
    The M4 Competition data is an extended and diverse set of time series to
    identify the most accurate forecasting method(s) for different types
    of domains, including Business, financial and economic forecasting,
    and different type of granularity, including Yearly (23,000 sequences),
    Quarterly (24,000 sequences), Monthly (48,000 sequences),
    Weekly(359 sequences), Daily (4,227 sequences) and Hourly (414 sequences)
    data.

    - source: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
    - timeseries sequences: 100,000
    """

    valid_subsets = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}.csv"

    def __init__(self, subset="Hourly", rootdir=None):
        super().__init__()
        self.subset = subset
        assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "M4")

        # download dataset if it is not found in root dir
        if not os.path.isdir(rootdir):
            logger.info(
                f"M4 {subset} dataset cannot be found from {rootdir}.\n"
                f"M4 {subset} dataset will be downloaded from {self.url}.\n"
            )
            download(rootdir, self.url, "M4-info")

        # extract starting date from meta-information of dataset
        info_dataset = pd.read_csv(os.path.join(rootdir, "M4-info.csv"), delimiter=",").set_index("M4id")

        if subset == "Yearly":
            logger.warning(
                "the max length of yearly data is 841 which is too big to convert to "
                "timestamps, we fallback to quarterly frequency"
            )
            freq = "13W"
        elif subset == "Quarterly":
            freq = "13W"
        elif subset == "Monthly":
            freq = "30D"
        else:
            freq = subset[0]

        train_csv = os.path.join(rootdir, f"train/{subset}-train.csv")
        if not os.path.isfile(train_csv):
            download(os.path.join(rootdir, "train"), self.url, f"{subset}-train", "Train")
        test_csv = os.path.join(rootdir, f"test/{subset}-test.csv")
        if not os.path.isfile(test_csv):
            download(os.path.join(rootdir, "test"), self.url, f"{subset}-test", "Test")

        train_set = pd.read_csv(train_csv).set_index("V1")
        test_set = pd.read_csv(test_csv).set_index("V1")
        for i in tqdm(range(train_set.shape[0])):
            ntrain = train_set.iloc[i, :].dropna().shape[0]
            sequence = pd.concat((train_set.iloc[i, :].dropna(), test_set.iloc[i, :].dropna()))
            # raw data do not follow consistent timestamp format
            sequence.index = pd.date_range(start=0, periods=sequence.shape[0], freq=freq)
            sequence = sequence.to_frame()

            metadata = pd.DataFrame({"trainval": sequence.index < sequence.index[ntrain]}, index=sequence.index)

            self.metadata.append(metadata)
            self.time_series.append(sequence)


def download(datapath, url, name, split=None):
    os.makedirs(datapath, exist_ok=True)
    if split is not None:
        namesplit = split + "/" + name
    else:
        namesplit = name
    url = url.format(namesplit)
    file_path = os.path.join(datapath, name) + ".csv"
    if os.path.isfile(file_path):
        logger.info(name + " already exists")
        return
    logger.info("Downloading " + url)
    r = requests.get(url, stream=True)
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
