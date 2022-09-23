# Contributing to Merlion
Thank you for your interest in contributing to Merlion! This document will help you get started with your contribution.
Before you get started, clone this repo, ``pip install pre-commit``, and run ``pre-commit install`` from the root
directory of the repo. This will ensure all files are formatted correctly and contain the appropriate
license headers whenever you make a commit.

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
for forecasters. Make sure to carefully read the [API docs](https://opensource.salesforce.com/Merlion/merlion.models.html)
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
-   Implement the abstract `_train()` method, and have it return the model's sequence of anomaly scores on the train
    data.
-   Implement the abstract `_get_anomaly_score()` method, which returns the model's predicted anomaly score on an
    input time series.

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
-   Implement the abstract `_train()` method, and have it return the model's sequence of predictions on the train
    data. These predictions should have the same format as the output of `model._forecast()` (see below).
-   Implement the abstract `_forecast()` method, which returns the model's forecast (and optionally the standard errors
    of that forecast, or `None` if the model does not have uncertainty quantification) on the timestamps of interest.

You may optionally override the class variable `ConfigClass._default_transform`. This is the default data pre-processing
transform used to process the data before giving it to the model, if the `transform` keyword argument is not
given when initializing the config.

See our implementation of [SARIMA](merlion/models/forecast/sarima.py) for a fairly simple example of what this looks
like in practice, and this [notebook](examples/forecast/4_ForecastNewModel.ipynb) for a step-by-step walkthrough of a
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
[notebook](examples/forecast/4_ForecastNewModel.ipynb).

## Data Pre-Processing Transforms
To implement a new data pre-processing transform, begin by reading the
[API docs](https://opensource.salesforce.com/Merlion/merlion.transform.html) for the base classes
`merlion.transform.base.TransformBase` and `merlion.transform.base.InvertibleTransformBase`. Inherit from the
appropriate base class, depending on whether you transform is invertible. However, even non-invertible transforms
should support pseudo-inversion. For example, when you take the moving average of an input time series with
`merlion.transform.moving_average.MovingAverage`, it isn't guaranteed that the original time series can be recovered.
However, we can approximate the desired inversion by keeping track of the boundary values and performing a
de-convolution.

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
[API docs](https://opensource.salesforce.com/Merlion/ts_datasets.html) for more details.

Once you've implemented your data loader class, add it to the top-level namespace of the module
([`ts_datasets/ts_datasets/anomaly/__init__.py`](ts_datasets/ts_datasets/anomaly/__init__.py) or
[`ts_datasets/ts_datasets/forecast/__init__.py`](ts_datasets/ts_datasets/forecast/__init__.py)) by importing it
and adding it to `__all__`. 
