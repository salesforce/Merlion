#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Abstractions for hyperparameter search.
"""
from collections import OrderedDict
import itertools
from typing import Any, Dict, List


class GridSearch:
    """
    Iterator over a grid of parameter values, skipping any restricted combinations of values.
    """

    def __init__(self, param_values: Dict[str, List], restrictions: List[Dict[str, Any]] = None):
        """
        :param param_values: a dict mapping a set of parameter names to lists of values they can take on.
        :param restrictions: a list of dicts indicating inadmissible combinations of parameter values.
            For example, an ETS model has parameters error (add/mul), trend (add/mul/none), seasonal (add/mul),
            and damped_trend (True/False). If we are only considering additive models, we would impose the restrictions
            ``[{"error": "mul"}, {"trend": "mul"}, {"seasonal": "mul"}]``. Since a damped trend is only possible if
            the model has a trend, we would add the restriction ``{"trend": None, "damped_trend": True}``.
        """
        self.param_values = param_values
        self.restrictions = [] if restrictions is None else restrictions

    def __iter__(self):
        for val_tuples in itertools.product(*(itertools.product([k], v) for k, v in self.param_values.items())):
            val_dict = OrderedDict(val_tuples)
            if not any(all(k in val_dict and val_dict[k] == v for k, v in r.items()) for r in self.restrictions):
                yield val_dict
