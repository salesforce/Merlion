#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import json
import numpy as np
import pandas as pd
from merlion.models.factory import ModelFactory


class DataMixin:
    def load_data(self, file_path, nrows=None):
        if nrows is None:
            self.logger.info("Loading the time series...")
        df = pd.read_csv(file_path, nrows=nrows)
        if df.dtypes[df.columns[0]] in [np.int32, np.int64]:
            df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index.values, unit="ms")
        else:
            df = df.set_index(df.columns[0])
        return df


class ModelMixin:
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
