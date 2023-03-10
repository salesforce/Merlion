Merlion Architecture
====================
This document is intended for Merlion developers. It outlines the architecture of Merlion's key components,
and how they interact with each other.

Transforms
----------
:doc:`Transforms <merlion.transform>` in Merlion apply various useful pre-processing to time series data.

Training
^^^^^^^^
Many transforms are *trainable*.
For example, if we want to normalize the data to have zero mean and unit variance, we use training data to learn the
mean and variance of each variable in the time series. If we wish to resample the data to a fixed granularity, we use
the most commonly observed timedelta in the training data.

Inversion
^^^^^^^^^
Many transforms are *invertible*.
For example, one may invert the normalization ``y = (x - mu) / sigma`` via ``x = sigma * y + mu``.
However, other transforms are lossy, and the input cannot be recovered without a *state*. For example, consider the
difference transform ``y[i+1] = x[i+1] - x[i]``. We need to record ``x[0]`` as the ``transform.inversion_state``
in order invert the difference transform and recover ``x`` from ``y``.

For invertible transforms which require an inversion state, we handle the inversion state as follows:

* When the transform is called, the inversion state is set. For example, if ``diff = DifferenceTransform()``,
  ``y = diff(x)`` will record the first observation of each univariate in ``x`` as its inversion state.
* When ``transform.invert(y)`` is called, the inversion state is reset to ``None``, unless the user explicitly
  invokes ``transform.invert(y, retain_inversion_state=True)``. This ensures that the user doesn't inadvertently
  apply a stale inversion state to a new time series.

Some transforms are not invertible at all (e.g. resampling). In these case, ``transform.invert(y)`` simply returns
``y``, and a warning is emitted.

Multivariate Time Series
^^^^^^^^^^^^^^^^^^^^^^^^
For the time being, all transforms are applied identically to all univariates in a time series.
We generally track the variables required for each univariate via a dictionary that maps the name of the univariate to
the variables relevant for it. We explicitly use the names of each univariate to ensure robustness to ensure that
everything behaves as expected even if the individual variables are reordered.

A notable limitation of the current implementation is the fact that we cannot currently apply different transforms to
different univariates. For example, we cannot mean-variance normalize univariate 0 and apply a difference transform
to univariate 1. If there is demand for this sort of behavior in the future, we may consider adding a parameter to
each transform which indicates the names of the univariates it should be applied to. This may be combined with a
:py:class:`TransformStack <merlion.transform.sequence.TransformStack>` to apply different transforms to different
univariates. A new tutorial should be written if this feature is added.

Models
------
:doc:`Models <merlion.models>` are the central object in Merlion.

Pre-Processing
^^^^^^^^^^^^^^
Each ``model`` has a ``model.transform`` which pre-processes the data. Automatically applying this transform at both
training and inference time (and inverting the transform for forecasting) is a key feature of Merlion models. In
reality, it is worth noting that ``model.transform`` is generally a reference to ``model.config.transform``.

When ``model.train()`` is called, the first step is to call ``model.train_pre_process()``. This method

* Records the dimension of the training data as ``model.dim``
* Trains ``model.transform`` and applies it to the training data
* Records the sampling frequency of the transformed training data as ``model.timedelta``
  (as well as the offset ``model.timedelta_offset``)
* For forecasters, we additionally train and apply ``model.exog_transform`` on the exogenous data if any are given.
  We also record the dimension of the exogenous data as ``model.exog_dim``.

For anomaly detection, ``model.get_anomaly_score(time_series, time_series_prev)``
includes the following pre-processing steps:

* Apply ``model.transform`` to the concatenation of ``time_series_prev`` and ``time_series``.
* Ensure that the data's dimension matches the dimension of the training data.

For forecasting, ``model.forecast(time_stamps, time_series_prev, exog_data)``
includes the following pre-processing steps:

* If the model expects time series to be sampled at a fixed frequency, resample ``time_stamps``
  to the frequency specified by ``model.timedelta`` and ``model.timedelta_offset``.
* Save the current inversion state of ``model.transform``, and then apply ``model.transform`` to ``time_series_prev``.
* If ``exog_data`` is given, apply ``model.exog_transform`` to ``exog_data``, and
  resample ``exog_data`` to the same time stamps as ``time_series_prev`` (after the transform) and ``time_stamps``.
* Ensure that the dimensions of ``time_series_prev`` and ``exog_data`` match the training data.

User-Defined Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After pre-processing the input data, we pass it to the user-defined implementations ``model._train()``,
``model._train_with_exog()``, ``model._get_anomaly_score()``, or ``model._forecast()``. These methods do the real work
of training or inference for the underlying model, and these are the methods that must be manually defined for each new model.

Post-Processing
^^^^^^^^^^^^^^^
After training, both anomaly detectors and forecasters apply ``model.train_post_process()`` on the output of
``model._train()``. For anomaly detectors, this involves training their post-rule (calibrator and threshold) and then
returning the anomaly scores returned by ``model._train()``. For forecasters, this involves applying the inverse of
``model.transform`` on the forecast returned by ``model._train()``.

For anomaly detectors, the final step of calling ``model.get_anomaly_label()`` is to apply the post-rule on the
unprocessed anomaly scores. For forecasters, we apply the inverse transform on the forecast and then set the inversion
state of ``model.transform`` to be what it was before ``model.forecast()`` was called.

Multiple Time Series
^^^^^^^^^^^^^^^^^^^^
If we extend Merlion to accommodate training models on multiple time series, we must make some changes to the way that
models handle transforms. In particular,

* ``model.transform`` should be re-trained for each time series individually.
    * At training time, we will probably need to rewrite ``model.train_pre_process()`` to use a different copy of
      ``model.transform`` for each time series.
    * At inference time, ``time_series_prev`` must be a required parameter, and a copy of ``model.transform``
      should be trained on ``time_series_prev``.
* To make training code easier to write, ``model.train()`` probably doesn't need to return anything when trained on
  multiple time series. This also removes the need to invert the transform on the training data.
* For anomaly detection, the :doc:`post-processing transforms <merlion.post_process>` should be updated to accommodate
  multiple time series. This is especially important for calibration.
* For forecasting, ``model.transform`` can be trained and applied on ``time_series_prev``, and then inverted on the
  concatenation of ``time_series_prev`` and ``forecast`` as it is done now, via a call to ``model._process_forecast()``.
  ``model.exog_transform`` should also be handled similarly (minus the inversion).

Model Variants
--------------
There are a number of model variants which either build upon the above model classes or modify them slightly.

Simple Variants
^^^^^^^^^^^^^^^
Below are some simpler model variants that are useful to understand:

* In order to support forecasting with exogenous regressors, we implement the
  :py:class:`ForecasterExogBase <merlion.models.forecast.base.ForecasterExogBase>` base class.
  Most of the functionality to support exogenous regressors is actually implemented in
  :py:class:`ForecasterBase <merlion.models.forecast.base.ForecasterBase>`, which this class inherits from. The only
  real difference is that a few internal fields have been changed to indicate that exogenous regressors are supported.
* We support using basic forecasters as the basis for anomaly detection models. The key piece is the mixin class
  :py:class:`ForecastingDetectorBase <merlion.models.anomaly.forecast_based.base.ForecastingDetectorBase>`.
* Some models don't work unless the input is pre-normalized. To support these models, we implement the
  :py:class:`NormalizingConfig <merlion.models.base.NormalizingConfig>`. This config class applies a
  ``MeanVarNormalize`` after any other pre-processing (specified by the user in ``transform``) has been applied.
  The full transform is accessed via ``config.full_transform``. Models automatically understand how this works because
  the property ``model.transform`` tries to get ``model.config.full_transform`` if possible and defaults to
  ``model.config.transform`` otherwise.

Ensembles
^^^^^^^^^
Merlion supports ensembles of both anomaly detectors and forecasters. The ensemble config has two key components
which make this possible: ``ensemble.config.models`` contains all the models present in the ensemble, while
``ensemble.config.combiner`` contains a :py:mod:`combiner <merlion.models.ensemble.combine>` object which defines
a way of combining the outputs of multiple models. This includes Mean, Median, and ModelSelection based on an evaluation
metric. When doing model selection, the ``ensemble.train()`` method automatically splits the train data into training
and validation splits, and it evaluates the performance of each model on the validation split.
It then re-trains each model on the full training data afterwards.

Layered Models
^^^^^^^^^^^^^^
Layered models are a useful abstraction for models that act as a wrapper around another model. This feature is
especially useful for AutoML. Like ensembles, we store the wrapped model in ``layered_model.config.model``,
and ``layered_model.model`` is a reference to ``layered_model.config.model``. The *base model* is the model at the
lowest level of the hierarchy.

There are a number of dirty tricks used to (1) ensure that layered anomaly detectors and forecasters inherit from the
right base classes, (2) config parameters are not duplicated between different levels of the hierarchy, and (3) users
can call a parameter like ``layered_model.max_forecast_steps`` (which should only be defined for the base model) and
receive ``layered_model.base_model.max_forecast_steps`` directly.

The documentation for :py:mod:`merlion.models.layers` has some more details.

Post-Processing
---------------
Distinct :doc:`post-rules <merlion.post_process>` are only relevant for anomaly detection.
There are two types of post-rules: calibration and thresholding. Similar to transforms, post-rules may be trained by
calling ``post_rule.train(train_anom_scores)`` and applied by calling ``post_rule(anom_scores)``. Extending post-rules
so that they can be trained on multiple time series simultaneously is a worthwhile direction to investigate.

Other Modules
-------------
Most other modules are stand-alone pieces that don't directly interact with each other, except in longer pipelines. We
defer to the main documentation in :doc:`merlion`. 
