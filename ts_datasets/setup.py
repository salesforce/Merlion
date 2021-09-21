#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from setuptools import find_packages, setup

setup(
    name="ts_datasets",
    version="0.1.0",
    author="Aadyot Bhatnagar, Tian Lan, Chenghao Liu, Wenzhuo Yang",
    author_email="abhatnagar@salesforce.com",
    description="A library for easily loading time series anomaly detection & forecasting datasets",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(include=["ts_datasets*"]),
    install_requires=["cython", "numpy", "pandas", "requests", "sklearn", "tqdm", "wheel"],
)
