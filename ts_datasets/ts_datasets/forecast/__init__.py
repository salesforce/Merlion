#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Datasets for time series forecasting. Really, these are just time series with
no labels of any sort.
"""
from ts_datasets.base import BaseDataset
from ts_datasets.forecast.m4 import M4
from ts_datasets.forecast.energy_power import EnergyPower
from ts_datasets.forecast.seattle_trail import SeattleTrail
from ts_datasets.forecast.solar_plant import SolarPlant

__all__ = ["get_dataset", "M4", "EnergyPower", "SeattleTrail", "SolarPlant"]


def get_dataset(dataset_name: str, rootdir: str = None) -> BaseDataset:
    """
    :param dataset_name: the name of the dataset to load, formatted as
        ``<name>`` or ``<name>_<subset>``, e.g. ``EnergyPower`` or ``M4_Hourly``
    :param rootdir: the directory where the desired dataset is stored. Not
        required if the package :py:mod:`ts_datasets` is installed in editable
        mode, i.e. with flag ``-e``.
    :return: the data loader for the desired dataset (and subset) desired
    """
    name_subset = dataset_name.split("_", maxsplit=1)
    valid_datasets = set(__all__).difference({"get_dataset"})
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
