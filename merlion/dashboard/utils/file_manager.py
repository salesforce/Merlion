#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import base64
import zipfile
import diskcache
from pathlib import Path
from dash.long_callback import DiskcacheLongCallbackManager


class SingletonClass:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class FileManager(SingletonClass):

    def __init__(self, directory=None):
        self.directory = os.path.join(str(Path.home()), "merlion") \
            if directory is None else directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.data_folder = os.path.join(self.directory, "data")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        self.model_folder = os.path.join(self.directory, "models")
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        self.cache_folder = os.path.join(self.directory, "cache")
        self.long_callback_manager = DiskcacheLongCallbackManager(
            diskcache.Cache(self.cache_folder))

    def save_file(self, name, content):
        data = content.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(self.data_folder, name), "wb") as fp:
            fp.write(base64.decodebytes(data))

    def uploaded_files(self):
        files = []
        for filename in os.listdir(self.data_folder):
            path = os.path.join(self.data_folder, filename)
            if os.path.isfile(path):
                files.append(filename)
        return files

    def get_model_download_path(self, model_name):
        path = os.path.join(self.model_folder, model_name)
        zip_file = os.path.join(path, f"{model_name}.zip")
        with zipfile.ZipFile(zip_file, mode="w") as f:
            for file in Path(path).iterdir():
                if Path(file).name != f"{model_name}.zip":
                    f.write(file, arcname=file.name)
        return zip_file

    def get_model_list(self):
        models = []
        for name in os.listdir(self.model_folder):
            folder = os.path.join(self.model_folder, name)
            if os.path.isdir(folder):
                models.append(name)
        return models

    @property
    def base_directory(self):
        return self.directory

    @property
    def data_directory(self):
        return self.data_folder

    @property
    def model_directory(self):
        return self.model_folder

    def get_long_callback_manager(self):
        return self.long_callback_manager
