.. Merlion documentation master file, created by
   sphinx-quickstart on Mon Feb 22 16:50:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Merlion's documentation!
===================================
Merlion consists of two sub-repos: :py:mod:`merlion`, which is a repo for time series analysis, focusing on
anomaly detection and forecasting, and :py:mod:`ts_datasets`, a repo which implements data loaders for multiple time
series datasets, which loads time series as ``pandas.DataFrame`` s with accompanying metadata.

You can install :py:mod:`merlion` from PyPI by calling ``pip install sfdc-merlion``. You may install from source by
cloning this repo, navigating to the root directory, and calling ``pip install [-e] merlion/``. We recommend
installing in editable mode (i.e. specifying ``-e``) so you can pull new changes & run them without having
to reinstall the package. You may install additional dependencies for plotting & visualization via
``pip install sfdc-merlion[plot]``, or by calling ``pip install [-e] merlion/[plot]`` from the root directory of
this repo.

To install the data loading package :py:mod:`ts_datasets`, simply call ``pip install -e ts_datasets/``
from Merlion's root directory. Note that this package must be installed in editable mode if you don't want to
manually specify the root directory of every dataset when initializing its data loader.

Note the following external dependencies:

1. For Mac users who want to use the :py:class:`Light GBM Forecaster <merlion.models.forecast.boostingtrees.LGBMForecaster>`,
   please install `Homebrew <https://brew.sh/>`_ and call ``brew install libomp`` so that the OpenMP libary is
   available for the model.

2. :py:mod:`merlion`'s implementation of :py:mod:`Random Cut Forest <merlion.models.anomaly.random_cut_forest>`
   (a popular anomaly detection model from AWS, proposed by
   `Guha et al. 2016 <http://proceedings.mlr.press/v48/guha16.pdf>`_) depends on having the Java Development Kit (JDK)
   installed. For Ubuntu, call ``sudo apt-get install openjdk-11-jdk``. For Mac OS, install
   `Homebrew <https://brew.sh/>`_ and call ``brew tap adoptopenjdk/openjdk && brew install --cask adoptopenjdk11``.

To get started, we recommend the linked tutorials on `anomaly detection <examples/anomaly/0_AnomalyIntro>`
and `forecasting <examples/forecast/0_ForecastIntro>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   merlion
   ts_datasets


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
