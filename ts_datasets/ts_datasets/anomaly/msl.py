#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import sys
import logging
from ts_datasets.anomaly.base import TSADBaseDataset
from ts_datasets.anomaly.smd import download, combine_train_test_datasets
from ts_datasets.anomaly.smap import preprocess, load_data

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class MSL(TSADBaseDataset):
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
        download(_logger, rootdir, MSL.url, "SMAP")
        preprocess(_logger, os.path.join(rootdir, "SMAP"), dataset="MSL")
        # Load training/test datasets
        df, metadata = combine_train_test_datasets(*load_data(os.path.join(rootdir, "SMAP"), "MSL"))
        self.time_series.append(df)
        self.metadata.append(metadata)
