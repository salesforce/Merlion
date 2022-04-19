ts_datasets: Easy Data Loading
==============================

:py:mod:`ts_datasets` implements Python classes that manipulate numerous time series datasets
into standardized ``pandas.DataFrame`` s. The sub-modules are :py:mod:`ts_datasets.anomaly`
for time series anomaly detection, and :py:mod:`ts_datasets.forecast` for time series forecasting.
Simply install the package by calling ``pip install -e ts_datasets/`` from the root directory of Merlion.
Then, you can load a dataset (e.g. the "realAWSCloudwatch" split of the Numenta Anomaly Benchmark
or the "Hourly" subset of the M4 dataset) by calling

.. code-block:: python

    from ts_datasets.anomaly import NAB
    from ts_datasets.forecast import M4
    anom_dataset = NAB(subset="realAWSCloudwatch", rootdir=path_to_NAB)
    forecast_dataset = M4(subset="Hourly", rootdir=path_to_M4)

If you install this package in editable mode (i.e. specify ``-e`` when calling ``pip install -e ts_datasets/``),
there is no need to specify a ``rootdir`` for any of the data loaders.

The core features of general data loaders (e.g. for forecasting) are outlined in the API doc for
:py:class:`ts_datasets.base.BaseDataset`, and the features for time series anomaly detection data loaders
are outlined in the API doc for :py:class:`ts_datasets.anomaly.TSADBaseDataset`.

.. automodule:: ts_datasets
   :members:
   :undoc-members:
   :show-inheritance:

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   ts_datasets.anomaly
   ts_datasets.forecast

Submodules
----------

datasets.base module
--------------------

.. automodule:: ts_datasets.base
   :members:
   :undoc-members:
   :show-inheritance:
