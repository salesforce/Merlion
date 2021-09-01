# Contributing to Merlion
Thank you for your interest in contributing to Merlion! This document will help you get started with your contribution.
If your contribution adds new dependencies, please document them
[here](https://docs.google.com/spreadsheets/d/1ycL1cH5_HdWduchHySdbF3BcGwTUo4CSSO3zS8k6BRI/edit?usp=sharing). As a
general rule, dependencies must be installable via `pip` and should have a permissive license like Apache 2.0, MIT,
or BSD.

## Table of Contents
1. [Models](#models)
    1. [Anomaly Detectors](#anomaly-detectors)
    1. [Forecasters](#forecasters)
    1. [Forecaster-Based Anomaly Detectors](#forecaster-based-anomaly-detectors)
1. [Data Pre-Processing Transforms](#data-pre-processing-transforms)
1. [Datasets](#datasets)

## Models
Merlion supports two kinds of models: anomaly detectors (`merlion.models.anomaly`) and forecasters
(`merlion.models.forecast`). Each model should inherit from its respective base class:
`merlion.models.anomaly.base.DetectorBase` for anomaly detectors, and `merlion.models.forecast.base.ForecasterBase`
for forecasters. Make sure to carefully read the [API docs](https://salesforce.github.io/Merlion/merlion.models.html)
for the appropriate base class and module before  proceeding.

After implementing your model, please register it with the [model factory](merlion/models/factory.py), and add it to
the appropriate config file for the benchmarking scripts ([`conf/benchmark_anomaly.py`](conf/benchmark_anomaly.json)
or [`conf/benchmark_forecast.json`](conf/benchmark_forecast.json)). Also implement some [unit tests](tests) to ensure
that the model is behaving as expected.

Finally, update the API docs by
-   Adding your new module to the autosummary block of the appropriate `__init__.py`
    (e.g. [`merlion/models/anomaly/__init__.py`](merlion/models/anomaly/__init__.py) for an anomaly detector)
-   Adding your new module to the appropriate Sphinx ReST file (e.g. 
    [`docs/source/merlion.models.anomaly.rst`](docs/source/merlion.models.anomaly.rst) for an anomaly detector)

### Anomaly Detectors
To implement a new anomaly detector, you need to do the following:
-   Implement an appropriate  `ConfigClass` which inherits from `merlion.models.anomaly.base.DetectorConfig`
-   Implement an initializer that takes an instance of this config class, i.e. you can instantiate
    `model = ModelClass(config)`. `ModelClass` should inherit from `DetectorBase`.
-   Set the model class's class variable `config_class` equal to the `ConfigClass` above
-   Implement the abstract `train()` method, and have it return the model's sequence of anomaly scores on the train
    data. Note that the `train()` method should include
    -   a call to `self.transform.train()`, to train the model's data pre-processing transform
    -   a call to `self.train_post_rule()`, to train the model's post-processing rule for anomaly scores
-   Implement the abstract `get_anomaly_score()` method, which returns the model's predicted anomaly score on an
    input time series. Note that you should transform the input time series using `self.transform(train_data)` before
    proceeding with predicting anomaly scores.

You may optionally override the following class variables of `ConfigClass` or `ModelClass`:
-   `ConfigClass._default_transform`: if the `transform` keyword argument is not given when initializing the model's
    config object, we use this default transform. If not overridden, we just use the `Identity` (i.e. no pre-processing)
-   `ConfigClass._default_post_rule`: if the `post_rule` keyword argument is not given when initializing the model's
    config object, we use this default post-rule. This is the rule used to post-process the output of
    `model.get_anomaly_score()` (e.g. to reduce noise). You can get the post-processed anomaly scores by
    calling `model.get_anomaly_label()`
-   `ModelClass._default_post_rule_train_config`: If `post_rule_train_config` is not passed to `model.train()` use this
    default config to train the post-rule (e.g. select the minimum anomaly score we want to set as a detection
    threshold).

See our implementation of [Isolation Forest](merlion/models/anomaly/isolation_forest.py) for a fairly simple example of
what this looks like in practice, and this [notebook](examples/anomaly/3_AnomalyNewModel.ipynb) for a step-by-step
walkthrough of a minimal example.

### Forecasters
To implement a new forecaster, you need to do the following:
-   Implement an appropriate  `ConfigClass` which inherits from `merlion.models.forecast.base.ForecasterConfig`. Make
    sure that this object takes a parameter `max_forecast_steps` if your model has a maximum horizon it is allowed to
    forecast for.
-   Implement an initializer that takes an instance of this config class, i.e. you can instantiate
    `model = ModelClass(config)`. `ModelClass` should inherit from `ForecasterBase`.
-   Set the model class's class variable `config_class` equal to the `ConfigClass` above
-   Implement the abstract `train()` method, and have it return the model's sequence of predictions on the train
    data. These predictions should have the same format as the output of `model.forecast()` (see below). Note that the
    `train()` method should include a call to `self.transform.train(train_data)`, to train the
    model's data pre-processing transform.
-   Implement the abstract `forecast()` method, which returns the model's forecast (and optionally the standard errors
    of that forecast) on the timestamps of interest. Note that this method should return a forecast of the
    *transformed* data, i.e. after `self.transform()` has been applied.

You may optionally override the class variable `ConfigClass._default_transform`. This is the default data pre-processing
transform used to process the data before giving it to the model, if the `transform` keyword argument is not
given when initializing the config.

See our implementation of [SARIMA](merlion/models/forecast/sarima.py) for a fairly simple example of what this looks
like in practice, and this [notebook](examples/forecast/ForecastNewModel.ipynb) for a step-by-step walkthrough of a
minimal example.

### Forecaster-Based Anomaly Detectors
Forecaster-based anomaly detectors convert a model's forecast into an anomaly score, by comparing the residual between
the true value and the forecaster's predicted value. Their base class is
`merlion.models.anomaly.forecast_based.base.ForecastingDetectorBase`. 

Consider a forecaster class `Forecaster` with config class `ForecasterConfig`. It is fairly straightforward to extend
this class into an `ForecasterDetectorClass`. You need to do the following things:
-   Define a config class which inherits from both `ForecasterConfig` and `DetectorConfig`
    (in that order). You may optionally specify a `_default_post_rule` class variable, but the `_default_transform`
    will be the same as `ForecasterConfig`.
-   Implement a model class which inherits from both `ForecastingDetectorBase` and `ForecasterClass` (in that order).
    You may optionally override the `_default_post_rule_train_config` class variable.

See our implementation of a [Prophet-based anomaly detector](merlion/models/anomaly/forecast_based/prophet.py) for an
example of what this looks like in practice, as well as the forecaster tutorial 
[notebook](examples/forecast/3_ForecastNewModel.ipynb).

## Data Pre-Processing Transforms
To implement a new data pre-processing transform, begin by reading the
[API docs](https://salesforce.github.io/Merlion/merlion.transform.html) for the base classes
`merlion.transform.base.TransformBase` and `merlion.transform.base.InvertibleTransformBase`. Inherit from the
appropriate base class, depending on whether you transform is invertible. However, even non-invertible transforms
should support pseudo-inversion. For example, when you resample an input time series with
`merlion.transform.resample.TemporalResample`, it isn't guaranteed that the original time series can be recovered.
However, we can approximate the desired inversion by keeping track of the original timestamps, and returning
*interpolated* values of the resampled time series at the original time stamps.

To actually implement a transform, override the abstract methods `train()` (this may be a no-op for
transforms with no data-dependent parameters), `__call__()` (actually applying the transform), and `_invert` (inverting
or pseudo-inverting the transform). Then, register it with the [transform factory](merlion/transform/factory.py).

Specify the class property `requires_inversion_state` to indicate whether the inversion (or pseudo-inversion) is
stateless (e.g. `merlion.transform.Rescale` simply rescales an input by a fixed scale and bias) or stateful
(e.g. `merlion.transform.moving_average.DifferenceTransform` requires us to know the first time & value in the time
series, since these are discarded by the forward transform). If the transform requires an inversion state, set
`self.inversion_state` in the `__call__()` method and use it as needed in the `_invert()` method. Note that the
inversion state can have whatever data format is most suitable for your purposes.

Check the implementations of [`DifferenceTransform`](merlion/transform/moving_average.py#L139),
[`PowerTransform`](merlion/transform/normalize.py#L123), and [`LowerUpperClip`](merlion/transform/bound.py)
for a few examples. Add your transforms to whatever module you feel is most appropriate, or create a new file if it
doesn't fit in with any of the existing ones. However, if you do add a new file, make sure to add it to the API docs
by updating the autosummary block of [`merlion/transform/__init__.py`](merlion/transform/__init__.py), and adding the
new module to the Sphinx ReST file [`docs/source/merlion.transform.rst`](docs/source/merlion.transform.rst). 

## Datasets
You can add support for a new dataset of time series by implementing an appropriate data loading class in
[`ts_datasets`](ts_datasets), and uploading the raw data (potentially compressed) to the [`data`](data) directory.
If your dataset has labeled anomalies, it belongs in [`ts_datasets.anomaly`](ts_datasets/ts_datasets/anomaly). If it
does not have labeled anomalies, it belongs in [`ts_datasets.forecast`](ts_datasets/ts_datasets/forecast). See the
[API docs](https://salesforce.github.io/Merlion/ts_datasets.html) for more details.

Once you've implemented your data loader class, add it to the top-level namespace of the module
([`ts_datasets/ts_datasets/anomaly/__init__.py`](ts_datasets/ts_datasets/anomaly/__init__.py) or
[`ts_datasets/ts_datasets/forecast/__init__.py`](ts_datasets/ts_datasets/forecast/__init__.py)) by importing it
and adding it to `__all__`. 
