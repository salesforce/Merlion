#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import os
import logging
import requests
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from ts_datasets.anomaly.base import TSADBaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class UCR(TSADBaseDataset):
    """
    Data loader for the Hexagon ML/UC Riverside Time Series Anomaly Archive.

    See `here <https://compete.hexagon-ml.com/practice/competition/39/>`_ for details.

    Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu,
    Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu,
    Nurjahan Begum, Anthony Bagnall , Abdullah Mueen, Gustavo Batista, & Hexagon-ML (2019).
    The UCR Time Series Classification Archive. URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    """

    def __init__(self, rootdir=None):
        super().__init__()
        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "ucr")

        self.download(rootdir)
        self.time_series = sorted(
            glob.glob(
                os.path.join(
                    rootdir, "UCR_TimeSeriesAnomalyDatasets2021", "FilesAreInHere", "UCR_Anomaly_FullData", "*.txt"
                )
            )
        )

    def __getitem__(self, i):
        fname = self.time_series[i]
        split, anom_start, anom_end = [int(x) for x in fname[: -len(".txt")].split("_")[-3:]]
        name = fname.split("_")[-4]
        arr = np.loadtxt(fname)
        trainval = [i < split for i in range(len(arr))]
        anomaly = [anom_start <= i <= anom_end for i in range(len(arr))]
        index = pd.date_range(start=0, periods=len(arr), freq="1min")
        df = pd.DataFrame({name: arr}, index=index)
        return (
            df,
            pd.DataFrame(
                {
                    "anomaly": [anom_start - 100 <= i <= anom_end + 100 for i in range(len(arr))],
                    "trainval": [i < split for i in range(len(arr))],
                },
                index=index,
            ),
        )

    def download(self, rootdir):
        filename = "UCR_TimeSeriesAnomalyDatasets2021.zip"
        url = f"https://www.cs.ucr.edu/~eamonn/time_series_data_2018/{filename}"

        os.makedirs(rootdir, exist_ok=True)
        compressed_file = os.path.join(rootdir, filename)

        # Download the compressed dataset
        if not os.path.exists(compressed_file):
            logger.info("Downloading " + url)
            with requests.get(url, stream=True) as r:
                with open(compressed_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            f.flush()

        # Uncompress the downloaded zip file
        if not os.path.isfile(os.path.join(rootdir, "_SUCCESS")):
            logger.info(f"Uncompressing {compressed_file}")
            with zipfile.ZipFile(compressed_file, "r") as zip_ref:
                zip_ref.extractall(rootdir)
            Path(os.path.join(rootdir, "_SUCCESS")).touch()
