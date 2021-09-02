#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import pandas as pd
from typing import Tuple

_intro_docstr = "Base dataset class for storing time series as ``pd.DataFrame`` s."

_main_fns_docstr = """
Each dataset supports the following features:

1.  ``__getitem__``: you may call ``ts, metadata = dataset[i]``. ``ts`` is a time-indexed ``pandas`` DataFrame, with
    each column representing a different variable (in the case of multivariate time series). ``metadata`` is a dict or
    ``pd.DataFrame`` with the same index as ``ts``, with different keys indicating different dataset-specific
    metadata (train/test split, anomaly labels, etc.) for each timestamp.
2.  ``__len__``:  Calling ``len(dataset)`` will return the number of time series in the dataset.
3.  ``__iter__``: You may iterate over the ``pandas`` representations of the time series in the dataset with
    ``for ts, metadata in dataset: ...``

.. note::

    For each time series, the ``metadata`` will always have the key ``trainval``, which is a 
    ``pd.Series`` of ``bool`` indicating whether each timestamp of the time series should be
    training/validation (if ``True``) or testing (if ``False``). 
"""


class BaseDataset:
    __doc__ = _intro_docstr + _main_fns_docstr

    time_series: list
    """
    A list of all individual time series contained in the dataset. Iterating over
    the dataset will iterate over this list. Note that for some large datasets, 
    ``time_series`` may be a list of filenames, which are read lazily either during
    iteration, or whenever ``__getitem__`` is invoked.
    """

    metadata: list
    """
    A list containing the metadata for all individual time series in the dataset.
    """

    def __init__(self):
        self.subset = None
        self.time_series = []
        self.metadata = []

    def __getitem__(self, i) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.time_series[i], self.metadata[i]

    def __len__(self):
        return len(self.time_series)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def describe(self):
        for ts_df in self.time_series:
            print(f"length of the data: {len(ts_df)}")
            print(f"timestamp index name: {ts_df.index.name}")
            print(f"number of data columns: {len(ts_df.columns)}")
            print("data columns names (the first 20): ")
            print(ts_df.columns[:20])
            print(f"number of null entries: {ts_df.isnull().sum()}")
