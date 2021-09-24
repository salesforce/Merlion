merlion: Time Series Intelligence
=================================
:py:mod:`merlion` is a Python library for time series intelligence. We support the following key features,
each associated with its own sub-package:

-   :py:mod:`merlion.models`: A library of models unified under a single shared interface, with specializations
    for anomaly detection and forecasting. More specifically, we have

    -   :py:mod:`merlion.models.anomaly`: Anomaly detection models
    -   :py:mod:`merlion.models.forecast`: Forecasting models
    -   :py:mod:`merlion.models.anomaly.forecast_based`: Forecasting models adapted for anomaly detection. Anomaly
        scores are based on the residual between the predicted and true value at each timestamp.
    -   :py:mod:`merlion.models.ensemble`: Ensembles & automated model selection of models for both anomaly
        detection and forecasting.
    -   :py:mod:`merlion.models.automl`: AutoML layers for various models

-   :py:mod:`merlion.transform`: Data pre-processing layer which implements many standard data transformations used in
    time series analysis. Transforms are callable objects, and each model has its own configurable ``model.transform``
    which it uses to pre-process all input time series for both training and inference.
-   :py:mod:`merlion.post_process`: Post-processing rules to apply on the output of a model. Currently, these are
    specific to anomaly detection, and include

    -   :py:mod:`merlion.post_process.calibrate`: Rules to calibrate the anomaly scores returned by a model, to
        be interpretable as z-scores, i.e. as standard deviations of a standard normal random variable. Each
        anomaly detection model has a ``model.calibrator`` from this module, which can optionally be applied to ensure
        that the model's anomaly scores are calibrated.
    -   :py:mod:`merlion.post_process.threshold`: Rules to reduce the noisiness of an anomaly detection model's outputs.
        Each anomaly detection model has a ``model.threshold`` from this module, which can optionally be applied to
        filter the model's predicted sequence of anomaly scores.
-   :py:mod:`merlion.evaluate`: Evaluation metrics & pipelines to simulate the live deployment of a time series model
    for any task.
-   :py:mod:`merlion.plot`: Automated visualization of model outputs for univariate time series

The key classes for input and output are `merlion.utils.time_series.TimeSeries` and
`merlion.utils.time_series.UnivariateTimeSeries`. Notably, these classes have transparent inter-operability
with ``pandas.DataFrame`` and ``pandas.Series``, respectively. Check this `tutorial <examples/TimeSeries>`
for some examples on how to use these classes, or the API docs linked above for a full list of features.

.. automodule:: merlion
   :members:
   :undoc-members:
   :show-inheritance:

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   merlion.models
   merlion.transform
   merlion.post_process
   merlion.evaluate
   merlion.utils

Submodules
----------

.. _merlion.plot:

merlion.plot module
-------------------

.. automodule:: merlion.plot
   :members:
   :undoc-members:
   :show-inheritance:
