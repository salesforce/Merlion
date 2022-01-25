#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from setuptools import setup, find_namespace_packages

MERLION_JARS = [
    "resources/gson-2.8.9.jar",
    "resources/randomcutforest-core-1.0.jar",
    "resources/randomcutforest-serialization-json-1.0.jar",
]

# optional dependencies
extra_require = {
    "plot": ["plotly>=4.13"],
    "prophet": ["prophet", "pystan<3.0"],  # pystan >= 3.0 doesn't work with prophet
    "deep-learning": ["torch>=1.1.0"],
}
extra_require["all"] = sum(extra_require.values(), [])


def read_file(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="salesforce-merlion",
    version="1.1.1",
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
        "GitPython",
        "py4j>=0.10.9.2",  # same minimum supported version as pyspark
        "matplotlib",
        "numpy>=1.21; python_version >= '3.7'",  # 1.21 remediates a security risk
        "numpy>=1.19; python_version < '3.7'",  # however, numpy 1.20+ requires python 3.7+
        "packaging",
        "pandas>=1.1.0",  # >=1.1.0 for origin kwarg to df.resample()
        "scikit-learn>=0.22",  # >=0.22 for changes to isolation forest algorithm
        "scipy>=1.6.0; python_version >= '3.7'",  # 1.6.0 adds multivariate_t density to scipy.stats
        "scipy>=1.5.0; python_version < '3.7'",  # however, scipy 1.6.0 requires python 3.7+
        "statsmodels>=0.12.2",
        "lightgbm",  # if running at MacOS, need OpenMP: "brew install libomp"
        "tqdm",
        "wheel",
        "pytest",
    ],
    extras_require=extra_require,
    python_requires=">=3.6.0",
    zip_safe=False,
)
