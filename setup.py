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

setup(
    name="sfdc-merlion",
    version="1.0.0",
    author="Aadyot Bhatnagar, Paul Kassianik, Chenghao Liu, Tian Lan, Wenzhuo Yang, "
    "Rowan Cassius, Doyen Sahoo, Devansh Arpit, Sri Subramanian, Gerald Woo, Amrita Saha, Arun Kumar Jagota, "
    "Gokulakrishnan Gopalakrishnan, Manpreet Singh, K C Krithika, Sukumar Maddineni, Daeki Cho, Bo Zong, "
    "Huan Wang, Yingbo Zhou, Caiming Xiong, Steven Hoi, Silvio Savarese",
    description="Merlion: A Machine Learning Framework for Time Series Intelligence",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Time Series Anomaly Detection Forecast Visualization",
    url="https://github.com/salesforce/Merlion",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="merlion.*"),
    package_dir={"merlion": "merlion"},
    package_data={"merlion": MERLION_JARS},
    install_requires=[
        "cython",
        "dill",
        "fbprophet",
        "GitPython",
        "JPype1==1.0.2",
        "matplotlib",
        "numpy!=1.18.*",  # 1.18 causes a bug with scipy
        "pandas>=1.1.0",  # >=1.1.0 for origin kwarg to df.resample()
        'pystan<3.0"',  # >=3.0 fails with fbprophet
        "scikit-learn>=0.22",  # >=0.22 for changes to isolation forest algorithm
        "scipy>=1.5.0",
        "statsmodels>=0.12.2",
        "torch>=1.1.0",
        "lightgbm",  # if running at MacOS, need OpenMP: "brew install libomp"
        "tqdm",
        "wheel",
        "pytest",
    ],
    extras_require={"plot": "plotly>=4"},
    python_requires=">=3.6.0",
    zip_safe=False,
)
