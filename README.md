# Merlion: A Machine Learning Library for Time Series Intelligence

<div align="center">
  <a href="https://github.com/salesforce/Merlion/actions">
  <img alt="Tests" src="https://github.com/salesforce/Merlion/actions/workflows/tests.yml/badge.svg?branch=main"/>
  </a>
  <a href="https://github.com/salesforce/Merlion/actions">
  <img alt="Coverage" src="https://github.com/salesforce/Merlion/raw/badges/coverage.svg"/>
  </a>
  <a href="https://pypi.python.org/pypi/sfdc-merlion">
  <img alt="PyPI Version" src="https://img.shields.io/pypi/v/sfdc-merlion.svg"/>
  </a>
  <a href="https://salesforce.github.io/Merlion">
  <img alt="docs" src="https://github.com/salesforce/Merlion/actions/workflows/docs.yml/badge.svg"/>
  </a>
</div>


## Table of Contents
1. [Introduction](#introduction)
1. [Installation](#installation)
1. [Documentation](#documentation)
1. [Getting Started](#getting-started)
    1. [Anomaly Detection](#anomaly-detection)
    1. [Forecasting](#forecasting)
1. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
1. [Technical Report and Citing Merlion](#technical-report-and-citing-merlion)

## Introduction
Merlion is a Python library for time series intelligence. It provides an end-to-end machine learning framework that
includes loading and transforming data, building and training models, post-processing model outputs, and evaluating
model performance. It supports various time series learning tasks, including forecasting and anomaly detection for both
univariate and multivariate time series. Merlion's key features are 
-  Standardized and easily extensible data loading & benchmarking for a wide range of forecasting and anomaly
   detection datasets.
-  A library of diverse models and data processing transforms unified under a shared interface. New models are easy
   to add, and we provide "default" models that are both efficient and achieve good performance across a wide range
   of settings.
-  AutoML for automated hyperaparameter tuning and model selection.
-  Practical, industry-inspired post-processing rules for anomaly detectors that make anomaly scores more
   interpretable, while also reducing the number of false positives.
-  Easy-to-use ensembles that combine the outputs of multiple models to achieve more robust performance. 
-  Flexible evaluation pipelines that simulate the live deployment & re-training of a model in production,
   and evaluate performance on both forecasting and anomaly detection.
-  Native support for visualizing model predictions.

The table below provides a visual overview of how Merlion's feature set compares to similar libraries for time series
anomaly detection and/or forecasting.

|              |  Univar Forecast   |  Multivar Forecast |   Univar Anomaly   |  Multivar Anomaly  |     AutoML         |    Ensemble        |  Bench-marking     |   Plots    
|-------------:|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
|  alibi-detect|      --            |      --            | :heavy_check_mark: | :heavy_check_mark: |      --            |      --            |      --            |      --
|          Kats| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |      --            |      --            | :heavy_check_mark:
|   statsmodels| :heavy_check_mark: | :heavy_check_mark: |      --            |      --            |      --            |      --            |      --            |      --
|      gluon-ts| :heavy_check_mark: | :heavy_check_mark: |      --            |      --            |      --            |      --            | :heavy_check_mark: |      --
|          RRCF|      --            |      --            | :heavy_check_mark: | :heavy_check_mark: |      --            | :heavy_check_mark: |      --            |      --
|        STUMPY|      --            |      --            | :heavy_check_mark: | :heavy_check_mark: |      --            |      --            |      --            |      --
|      Greykite| :heavy_check_mark: |      --            | :heavy_check_mark: |      --            | :heavy_check_mark: |      --            |      --            | :heavy_check_mark:
|       Prophet| :heavy_check_mark: |      --            | :heavy_check_mark: |      --            |      --            |      --            |      --            | :heavy_check_mark:
|      pmdarima| :heavy_check_mark: |      --            |      --            |      --            | :heavy_check_mark: |      --            |      --            |      --
|       Merlion| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:

## Installation

Merlion consists of two sub-repos: `merlion` implements the library's core time series intelligence features,
and `ts_datasets` provides standardized data loaders for multiple time series datasets that loads time series as
``pandas.DataFrame`` s with accompanying metadata.

You can install `merlion` from PyPI by calling ``pip install sfdc-merlion``. You may install from source by
cloning this repo, navigating to the root directory, and calling ``pip install .``, or ``pip install -e .`` to
install in editable mode. You may install  additional dependencies for plotting & visualization via
``pip install sfdc-merlion[plot]``, or by calling ``pip install .[plot]`` from the root directory of this repo.

To install the data loading package `ts_datasets`, clone this repo, navigate to its root directory, and call
``pip install -e ts_datasets/``. This package must be installed in editable mode (with the ``-e`` flag)
if you don't want  to manually specify the root directory of every dataset when initializing its data loader.

Note the following external dependencies:

1. For Mac users who want to use the Light GBM Forecaster, please install
   [Homebrew](https://brew.sh/) and call ``brew install libomp`` so that the 
   OpenMP libary is available for the model.

2. `merlion`'s implementation of Random Cut Forest (a popular anomaly detection model from AWS, proposed by
   [Guha et al. 2016](http://proceedings.mlr.press/v48/guha16.pdf)) depends on having the Java Development Kit (JDK)
   installed. For Ubuntu, call ``sudo apt-get install openjdk-11-jdk``. For Mac OS, install
   [Homebrew](https://brew.sh/) and call ``brew tap adoptopenjdk/openjdk && brew install --cask adoptopenjdk11``.

## Documentation

For example code and an introduction to Merlion, see the Jupyter notebooks in
[`examples`](https://github.com/salesforce/Merlion/tree/main/examples), and the guided walkthrough
[here](https://salesforce.github.io/Merlion/tutorials.html). You may find detailed API documentation (including the
example code) [here](https://salesforce.github.io/Merlion/index.html). The
[technical report](TODO) outlines Merlion's overall architecture
and presents experimental results on time series anomaly detection & forecasting for both univariate and multivariate
time series.

## Getting Started
Here, we provide some minimal examples using Merlion default models, 
to help you get started with both anomaly detection and forecasting.

### Anomaly Detection
We begin by importing Merlion’s `TimeSeries` class and the data loader for the Numenta Anomaly Benchmark `NAB`.
We can then divide a specific time series from this dataset into training and testing splits.

```python
from merlion.utils import TimeSeries
from ts_datasets.anomaly import NAB

# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = NAB(subset="realKnownCause")[3]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
test_labels = TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])
```

We can then initialize and train Merlion’s `DefaultDetector`, which is an anomaly detection model that
balances performance with efficiency. We also obtain its predictions on the test split.

```python
from merlion.models.defaults import DefaultDetectorConfig, DefaultDetector
model = DefaultDetector(DefaultDetectorConfig())
model.train(train_data=train_data)
test_pred = model.get_anomaly_label(time_series=test_data)
```

Next, we visualize the model's predictions.

```python
from merlion.plot import plot_anoms
import matplotlib.pyplot as plt
fig, ax = model.plot_anomaly(time_series=test_data)
plot_anoms(ax=ax, anomaly_labels=test_labels)
plt.show()
```
![anomaly figure](https://github.com/salesforce/Merlion/raw/main/figures/anom_example.png)

Finally, we can quantitatively evaluate the model. The precision and recall come from the fact that the model
fired 3 alarms, with 2 true positives, 1 false negative, and 1 false positive. We also evaluate the mean time
the model took to detect each anomaly that it correctly detected.

```python
from merlion.evaluate.anomaly import TSADMetric
p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n"
      f"Mean Time To Detect: {mttd}")
```
```
Precision: 0.6667, Recall: 0.6667, F1: 0.6667
Mean Time To Detect: 1 days 10:30:00
```
### Forecasting
We begin by importing Merlion’s `TimeSeries` class and the data loader for the `M4` dataset. We can then divide a
specific time series from this dataset into training and testing splits.

```python
from merlion.utils import TimeSeries
from ts_datasets.forecast import M4

# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = M4(subset="Hourly")[0]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
```

We can then initialize and train Merlion’s `DefaultForecaster`, which is an forecasting model that balances
performance with efficiency. We also obtain its predictions on the test split.

```python
from merlion.models.defaults import DefaultForecasterConfig, DefaultForecaster
model = DefaultForecaster(DefaultForecasterConfig())
model.train(train_data=train_data)
test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)
```

Next, we visualize the model’s predictions.

```python
import matplotlib.pyplot as plt
fig, ax = model.plot_forecast(time_series=test_data, plot_forecast_uncertainty=True)
plt.show()
```
![forecast figure](https://github.com/salesforce/Merlion/raw/main/figures/forecast_example.png)

Finally, we quantitatively evaluate the model. sMAPE measures the error of the prediction on a scale of 0 to 100
(lower is better), while MSIS evaluates the quality of the 95% confidence band on a scale of 0 to 100 (lower is better).

```python
# Evaluate the model's predictions quantitatively
from scipy.stats import norm
from merlion.evaluate.forecast import ForecastMetric

# Compute the sMAPE of the predictions (0 to 100, smaller is better)
smape = ForecastMetric.sMAPE.value(ground_truth=test_data, predict=test_pred)

# Compute the MSIS of the model's 95% confidence interval (0 to 100, smaller is better)
lb = TimeSeries.from_pd(test_pred.to_pd() + norm.ppf(0.025) * test_err.to_pd().values)
ub = TimeSeries.from_pd(test_pred.to_pd() + norm.ppf(0.975) * test_err.to_pd().values)
msis = ForecastMetric.MSIS.value(ground_truth=test_data, predict=test_pred,
                                 insample=train_data, lb=lb, ub=ub)
print(f"sMAPE: {smape:.4f}, MSIS: {msis:.4f}")
```
```
sMAPE: 6.2855, MSIS: 19.1584
```

## Evaluation and Benchmarking

One of Merlion's key features is an evaluation pipeline that simulates the live deployment
of a model on historical data. This enables you to compare models on the datasets relevant
to them, under the conditions that they may encounter in a production environment. Our
evaluation pipeline proceeds as follows:
1. Train an initial model on recent historical training data (designated as the training split of the time series)
1. At a regular interval (e.g. once per day), retrain the entire model on the most recent data. This can be either the
   entire history of the time series, or a more limited window (e.g. 4 weeks).
1. Obtain the model's predictions (anomaly scores or forecasts) for the time series values that occur between
   re-trainings. You may customize whether this should be done in batch (predicting all values at once),
   streaming (updating the model's internal state after each data point without fully re-training it),
   or some intermediate cadence.
1. Compare the model's predictions against the ground truth (labeled anomalies for anomaly detection, or the actual
   time series values for forecasting), and report quantitative evaluation metrics.

We provide scripts that allow you to use this pipeline to evaluate arbitrary models on arbitrary datasets.
For example, invoking
```shell script
python benchmark_anomaly.py --dataset NAB_realAWSCloudwatch --model IsolationForest --retrain_freq 1d
``` 
will evaluate the anomaly detection performance of the `IsolationForest` (retrained once a day) on the
"realAWSCloudwatch" subset of the NAB dataset.  Similarly, invoking
```shell script
python benchmark_forecast.py --dataset M4_Hourly --model ETS
```
will evaluate the batch forecasting performance (i.e. no retraining) of `ETS` on the "Hourly" subset of the M4 dataset. 
You can find the results produced by running these scripts in the Experiments section of the
[technical report](TODO).

## Technical Report and Citing Merlion
You can find more details in our technical report: TODO

If you're using Merlion in your research or applications, please cite using this BibTeX:
```
@article{bhatnagar2021merlion,
  author    =  {Aadyot Bhatnagar and Paul Kassianik and Chenghao Liu and Tian Lan
                and Wenzhuo Yang and Rowan Cassius and Doyen Sahoo and Devansh Arpit
                and Sri Subramanian and Gerald Woo and Amrita Saha and Arun Kumar Jagota
                and Gokulakrishnan Gopalakrishnan and Manpreet Singh and K C Krithika
                and Sukumar Maddineni and Daeki Cho and Bo Zong and Huan Wang
                and Yingbo Zhou and Caiming Xiong and Steven Hoi and Silvio Savarese},
  title     = {TODO},
  journal   = {CoRR},
  volume    = {TODO},
  year      = {2021},
  url       = {TODO},
  archivePrefix = {arXiv},
  eprint    = {TODO},
}
```
