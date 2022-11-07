#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from collections import OrderedDict
from enum import Enum
import inspect
import json
import os
import numpy as np
import pandas as pd
from merlion.models.factory import ModelFactory


class DataMixin:
    def load_data(self, file_path, nrows=None):
        if nrows is None:
            self.logger.info("Loading the time series...")
        df = pd.read_csv(file_path, nrows=nrows, index_col=0)
        df.index = pd.to_datetime(df.index, unit="ms" if df.dtypes[df.columns[0]] in [np.int32, np.int64] else None)
        return df


class ModelMixin:
    @staticmethod
    def get_parameter_info(algorithm):
        model_class = ModelFactory.get_model_class(algorithm)
        param_info = ModelMixin._param_info(model_class.config_class.__init__)
        if "max_forecast_steps" in param_info:
            if param_info["max_forecast_steps"]["default"] == "":
                param_info["max_forecast_steps"]["default"] = 100
        return param_info

    @staticmethod
    def _param_info(function):
        def is_enum(t):
            return isinstance(t, type) and issubclass(t, Enum)

        def is_valid_type(t):
            return t in [int, float, str, bool, list, tuple, dict] or is_enum(t)

        param_info = OrderedDict()
        signature = inspect.signature(function).parameters
        for name, param in signature.items():
            if name in ["self", "target_seq_index"]:
                continue
            value = param.default
            if value == param.empty:
                value = ""
            if is_valid_type(type(param.default)):
                value = value.name if isinstance(value, Enum) else value
                param_info[name] = {"type": type(param.default), "default": value}
            elif is_valid_type(param.annotation):
                value = value.name if isinstance(value, Enum) else value
                param_info[name] = {"type": param.annotation, "default": value}

        return param_info

    @staticmethod
    def parse_parameters(param_info, params):
        for key in params.keys():
            assert key in param_info, f"{key} is not in `param_info`."

        kwargs = {}
        for name, value in params.items():
            info = param_info[name]
            value_type = info["type"]
            if value.lower() in ["none", "null"]:
                kwargs[name] = None
            elif value_type in [int, float, str]:
                kwargs[name] = value_type(value)
            elif issubclass(value_type, Enum):
                valid_enum_values = value_type.__members__.keys()
                assert value in valid_enum_values, f"The value of {name} should be in {valid_enum_values}"
                kwargs[name] = value_type[value]
            elif value_type == bool:
                assert value.lower() in ["true", "false"], f"The value of {name} should be either True or False."
                kwargs[name] = value.lower() == "true"
            elif info["type"] in [list, tuple, dict]:
                value = value.replace(" ", "").replace("\t", "")
                value = value.replace("(", "[").replace(")", "]").replace(",]", "]")
                kwargs[name] = json.loads(value)
        return kwargs

    @staticmethod
    def save_model(directory, model, algorithm):
        if model is None:
            return
        d = os.path.join(directory, algorithm)
        if not os.path.exists(d):
            os.makedirs(d)
        model.save(d)

    @staticmethod
    def load_model(directory, algorithm):
        d = os.path.join(directory, algorithm)
        model_class = ModelFactory.get_model_class(algorithm)
        return model_class.load(d)
