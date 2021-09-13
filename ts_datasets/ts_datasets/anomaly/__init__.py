#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Datasets for time series anomaly detection (TSAD). All the time series in these
datasets have anomaly labels.
"""
from ts_datasets.anomaly.base import TSADBaseDataset
from ts_datasets.anomaly.iops_competition import IOpsCompetition
from ts_datasets.anomaly.nab import NAB
from ts_datasets.anomaly.synthetic import Synthetic
from ts_datasets.anomaly.ucr import UCR

from ts_datasets.anomaly.smd import SMD
from ts_datasets.anomaly.smap import SMAP
from ts_datasets.anomaly.msl import MSL

__all__ = ["get_dataset", "TSADBaseDataset", "IOpsCompetition", "NAB", "Synthetic", "UCR", "SMD", "SMAP", "MSL"]


def get_dataset(dataset_name: str, rootdir: str = None) -> TSADBaseDataset:
    """
    :param dataset_name: the name of the dataset to load, formatted as
        ``<name>`` or ``<name>_<subset>``, e.g. ``IOPsCompetition``
        or ``NAB_realAWSCloudwatch``
    :param rootdir: the directory where the desired dataset is stored. Not
        required if the package :py:mod:`ts_datasets` is installed in editable
        mode, i.e. with flag ``-e``.
    :return: the data loader for the desired dataset (and subset) desired
    """
    name_subset = dataset_name.split("_", maxsplit=1)
    valid_datasets = set(__all__).difference({"TSADBaseDataset", "get_dataset"})
    if name_subset[0] in valid_datasets:
        cls = globals()[name_subset[0]]
    else:
        raise KeyError(
            "Dataset should be formatted as <name> or "
            "<name>_<subset>, where <name> is one of "
            f"{valid_datasets}. Got {dataset_name} instead."
        )
    if not hasattr(cls, "valid_subsets") and len(name_subset) == 2:
        raise ValueError(
            f"Dataset {name_subset[0]} does not have any subsets, "
            f"but attempted to load subset {name_subset[1]} by "
            f"specifying dataset name {dataset_name}."
        )

    kwargs = dict() if len(name_subset) == 1 else dict(subset=name_subset[1])
    return cls(rootdir=rootdir, **kwargs)
