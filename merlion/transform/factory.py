#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains the `TransformFactory` for instantiating transforms.
"""

from typing import Type
from merlion.transform.base import TransformBase
from merlion.utils import dynamic_import


import_alias = dict(
    Identity="merlion.transform.base:Identity",
    MovingAverage="merlion.transform.moving_average:MovingAverage",
    ExponentialMovingAverage="merlion.transform.moving_average:ExponentialMovingAverage",
    DifferenceTransform="merlion.transform.moving_average:DifferenceTransform",
    LagTransform="merlion.transform.moving_average:LagTransform",
    Rescale="merlion.transform.normalize:Rescale",
    AbsVal="merlion.transform.normalize:AbsVal",
    MeanVarNormalize="merlion.transform.normalize:MeanVarNormalize",
    MinMaxNormalize="merlion.transform.normalize:MinMaxNormalize",
    PowerTransform="merlion.transform.normalize:PowerTransform",
    TemporalResample="merlion.transform.resample:TemporalResample",
    Shingle="merlion.transform.resample:Shingle",
    TransformSequence="merlion.transform.sequence:TransformSequence",
    TransformStack="merlion.transform.sequence:TransformStack",
    InvertibleTransformSequence="merlion.transform.sequence:InvertibleTransformSequence",
)


class TransformFactory(object):
    @classmethod
    def get_transform_class(cls, name: str) -> Type[TransformBase]:
        return dynamic_import(name, import_alias)

    @classmethod
    def create(cls, name: str, **kwargs) -> TransformBase:
        transform_class = cls.get_transform_class(name)
        return transform_class.from_dict(kwargs)
