.. Merlion documentation master file, created by
   sphinx-quickstart on Mon Feb 22 16:50:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Merlion's documentation!
===================================
Merlion is a Python library for time series intelligence. It features a unified interface for many commonly used
:doc:`models <merlion.models>` and :doc:`datasets <ts_datasets>` for anomaly detection and forecasting
on both univariate and multivariate time series, along with standard
:doc:`pre-processing <merlion.transform>` and :doc:`post-processing <merlion.post_process>` layers.
It has several modules to improve ease-of-use,
including :ref:`visualization <merlion.plot>`,
anomaly score :ref:`calibration <merlion.post_process.calibrate>` to improve interpetability,
:doc:`AutoML <merlion.models.automl>` for hyperparameter tuning and model selection,
and :doc:`model ensembling <merlion.models.ensemble>`.
Merlion also provides a unique :doc:`evaluation framework <merlion.evaluate>`
that simulates the live deployment and re-training of a model in production.
This library aims to provide engineers and researchers a one-stop solution to rapidly develop models
for their specific time series needs, and benchmark them across multiple time series datasets.

Installation
------------
Merlion consists of two sub-packages: :doc:`merlion <merlion>` implements the library's core time series intelligence features,
and :doc:`ts_datasets <ts_datasets>` provides standardized data loaders for multiple time series datasets. These loaders load
time series as ``pandas.DataFrame`` s with accompanying metadata.

You can install ``merlion`` from PyPI by calling ``pip install salesforce-merlion``. You may install from source by
cloning the Merlion `repo <https://github.com/salesforce/Merlion>`_, navigating to the root directory, and calling
``pip install .``, or ``pip install -e .`` to install in editable mode. You may install additional dependencies
for plotting & visualization via ``pip install salesforce-merlion[plot]``, or by calling ``pip install ".[plot]"`` from the
root directory of the repo if installing from source.

To install the data loading package ``ts_datasets``, simply clone the Merlion
`repo <https://github.com/salesforce/Merlion>`_ and call ``pip install -e ts_datasets/``
from its root directory. This package must be installed in editable mode (i.e. with the ``-e`` flag)
if you don't want to manually specify the root directory of every dataset when initializing its data loader.

Note the following external dependencies:

1. Some of our forecasting models depend on OpenMP. Some of our forecasting models depend on OpenMP.
   If using ``conda``, please ``conda install -c conda-forge lightgbm``
   before installing our package. This will ensure that OpenMP is configured to work with the ``lightgbm`` package
   (one of our dependencies) in your ``conda`` environment.
   If using Mac, please install `Homebrew <https://brew.sh/>`_ and call ``brew install libomp`` so that the
   OpenMP libary is available for the model.
   This is relevant for the
   :py:class:`LGBMForecaster <merlion.models.forecast.boostingtrees.LGBMForecaster>`,
   which is also used as a part of the :py:class:`DefaultForecaster <merlion.models.defaults.DefaultForecaster>`.

2. Some of our anomaly detection models depend on having the Java Development Kit (JDK) installed. For Ubuntu, call
   ``sudo apt-get install openjdk-11-jdk``. For Mac OS, install `Homebrew <https://brew.sh/>`_ and call
   ``brew tap adoptopenjdk/openjdk && brew install --cask adoptopenjdk11``.
   This is relevant for the :py:class:`RandomCutForest <merlion.models.anomaly.random_cut_forest.RandomCutForest>`
   which is also used as a part of the :py:class:`DefaultDetector <merlion.models.defaults.DefaultDetector>`.

Getting Started
---------------
To get started, we recommend the linked tutorials on `anomaly detection <examples/anomaly/0_AnomalyIntro>`
and `forecasting <examples/forecast/0_ForecastIntro>`. After that, you should read in more detail about Merlion's
main data structure for representing time series `here <examples/TimeSeries>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   merlion
   ts_datasets
   tutorials


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
