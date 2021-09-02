#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Class to compose a sequence of post-rules into a single post-rule.
"""
import inspect
from typing import Iterable

from merlion.post_process.base import PostRuleBase
from merlion.post_process.factory import PostRuleFactory
from merlion.utils import TimeSeries


class PostRuleSequence(PostRuleBase):
    def __init__(self, post_rules: Iterable):
        self.post_rules = list(post_rules)

    def train(self, anomaly_scores: TimeSeries, **kwargs) -> TimeSeries:
        for post_rule in self.post_rules:
            params = inspect.signature(post_rule.train).parameters
            if not any(v.kind.name == "VAR_KEYWORD" for v in params.values()):
                local_kwargs = {k: v for k, v in kwargs.items() if k in params}
            anomaly_scores = post_rule.train(anomaly_scores, **local_kwargs)
        return anomaly_scores

    def __call__(self, anomaly_scores: TimeSeries) -> TimeSeries:
        for post_rule in self.post_rules:
            anomaly_scores = post_rule(anomaly_scores)
        return anomaly_scores

    def to_dict(self):
        return {"name": type(self).__name__, "post_rules": [p.to_dict() for p in self.post_rules]}

    @classmethod
    def from_dict(cls, state_dict):
        post_rules = [
            d if isinstance(d, PostRuleBase) else PostRuleFactory.create(**d) for d in state_dict["post_rules"]
        ]
        return cls(post_rules)

    def __repr__(self):
        return "PostRuleSequence(\n " + ",\n ".join([repr(f) for f in self.post_rules]) + "\n)"
