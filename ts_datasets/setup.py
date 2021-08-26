from setuptools import find_packages, setup

setup(
    name="ts_datasets",
    version="0.1.0",
    author="Aadyot Bhatnagar, Tian Lan, Chenghao Liu, Wenzhuo Yang",
    author_email="abhatnagar@salesforce.com",
    description="A library for easily loading time series anomaly detection & forecasting datasets",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    license="Apache 2.0",
    packages=find_packages(include=["ts_datasets*"]),
    install_requires=[
        "cython",
        "numpy",
        "pandas>=1.1.0",
        "requests",
        "sklearn",
        "tqdm",
        "wheel",
    ],
)
