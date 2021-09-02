#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains the `PostRuleFactory`.
"""
from typing import Type
from merlion.post_process.base import PostRuleBase
from merlion.utils import dynamic_import

import_alias = dict(
    Threshold="merlion.post_process.threshold:Threshold",
    AggregateAlarms="merlion.post_process.threshold:AggregateAlarms",
    AdaptiveThreshold="merlion.post_process.threshold:AdaptiveThreshold",
    AdaptiveAggregateAlarms="merlion.post_process.threshold:AdaptiveAggregateAlarms",
    AnomScoreCalibrator="merlion.post_process.calibrate:AnomScoreCalibrator",
    PostRuleSequence="merlion.post_process.sequence:PostRuleSequence",
)


class PostRuleFactory(object):
    @classmethod
    def get_post_rule_class(cls, name: str) -> Type[PostRuleBase]:
        return dynamic_import(name, import_alias)

    @classmethod
    def create(cls, name: str, **kwargs) -> PostRuleBase:
        """
        Uses the given ``kwargs`` to create a post-rule of the given name
        """
        post_rule_class = cls.get_post_rule_class(name)
        return post_rule_class.from_dict(kwargs)
