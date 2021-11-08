#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from setuptools import setup, find_namespace_packages

MERLION_JARS = [
    "resources/gson-2.8.6.jar",
    "resources/randomcutforest-core-1.0.jar",
    "resources/randomcutforest-serialization-json-1.0.jar",
]


def read_file(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="salesforce-merlion",
    version="1.0.2",
    author=", ".join(read_file("AUTHORS.md").split("\n")),
    author_email="abhatnagar@salesforce.com",
    description="Merlion: A Machine Learning Framework for Time Series Intelligence",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords="time series, forecasting, anomaly detection, machine learning, autoML, "
    "ensemble learning, benchmarking, Python, scientific toolkit",
    url="https://github.com/salesforce/Merlion",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="merlion.*"),
    package_dir={"merlion": "merlion"},
    package_data={"merlion": MERLION_JARS},
    install_requires=[
        "cython",
        "dill",
        "prophet",
        "GitPython",
        "JPype1==1.0.2",
        "matplotlib",
        "numpy!=1.18.*",  # 1.18 causes a bug with scipy
        "packaging",
        "pandas>=1.1.0",  # >=1.1.0 for origin kwarg to df.resample()
        'pystan<3.0"',  # >=3.0 fails with prophet
        "scikit-learn>=0.22",  # >=0.22 for changes to isolation forest algorithm
        "scipy>=1.6.0; python_version >= '3.7'",  # 1.6.0 adds multivariate_t density to scipy.stats
        "scipy>=1.5.0; python_version < '3.7'",  # however, scipy 1.6.0 requires python 3.7+
        "statsmodels>=0.12.2",
        "torch>=1.1.0",
        "lightgbm",  # if running at MacOS, need OpenMP: "brew install libomp"
        "tqdm",
        "wheel",
        "pytest",
    ],
    extras_require={"plot": "plotly>=4.13"},
    python_requires=">=3.6.0",
    zip_safe=False,
)
