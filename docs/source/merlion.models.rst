
merlion.models package
======================
Broadly, Merlion contains two types of models: anomaly detection (:py:mod:`merlion.models.anomaly`)
and forecasting (:py:mod:`merlion.models.forecast`). Note that there is a distinct subset of anomaly
detection models that use forecasting models at their core (:py:mod:`merlion.models.anomaly.forecast_based`).

We implement an abstract `ModelBase` class which provides the following functionality for all models:

1.  ``model = ModelClass(config)``

    -   initialization with a model-specific config (which inherits from `Config`)
    -   configs contain:

        -   a (potentially trainable) data pre-processing transform from :py:mod:`merlion.transform`;
            note that ``model.transform`` is a property which refers to ``model.config.transform``
        -   model-specific hyperparameters

2.  ``model.save(dirname, save_config=None)``

    -   saves the model to the specified directory. The model's configuration is saved to
        ``<dirname>/config.json``, while the model's binary data is (by default) saved in binary form to
        ``<dirname>/model.pkl``. Note that if you edit the saved ``<dirname>/config.json`` on disk, the changes
        will be loaded when you call ``ModelClass.load(dirname)``!
    -   this method heavily exploits the fact that many objects in Merlion are JSON-serializable

3. ``ModelClass.load(dirname, **kwargs)``

    -   this class method initializes an instance of ``ModelClass`` from the config file saved in
        ``<dirname>/config.json``, (overriding any parameters of the config with ``kwargs`` where relevant),
        loads the remaining binary data into the model object, and returns the fully initialized model.

For users who aren't familiar with the specific details of various models, we provide default models for anomaly
detection and forecasting in :py:mod:`merlion.models.defaults`.

We also provide a `ModelFactory` which can be used to conveniently instantiate models from their name and a set of
keyword arguments, or to load them directly from disk. For example, we may have the following workflow:

.. code-block:: python

    from merlion.models.factory import ModelFactory
    from merlion.models.anomaly.windstats import WindStats, WindStatsConfig

    # creates the same kind of model in 2 equivalent ways
    model1a = WindStats(WindStatsConfig(wind_sz=60))
    model1b = ModelFactory.create("WindStats", wind_sz=60)

    # save the model & load it in 2 equivalent ways
    model1a.save("tmp")
    model2a = WindStats.load("tmp")
    model2b = ModelFactory.load("tmp")

Finally, we support ensembles of models in :py:mod:`merlion.models.ensemble`.

.. automodule:: merlion.models
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
    base
    factory
    defaults
    anomaly
    anomaly.forecast_based
    forecast
    ensemble
    automl


Subpackages
-----------

.. toctree::
   :maxdepth: 2

   merlion.models.anomaly
   merlion.models.anomaly.forecast_based
   merlion.models.forecast
   merlion.models.ensemble
   merlion.models.automl


Submodules
----------

merlion.models.base module
--------------------------

.. automodule:: merlion.models.base
   :members:
   :undoc-members:
   :show-inheritance:

merlion.models.factory module
-----------------------------

.. automodule:: merlion.models.factory
   :members:
   :undoc-members:
   :show-inheritance:


merlion.models.defaults module
------------------------------

.. automodule:: merlion.models.defaults
   :members:
   :undoc-members:
   :show-inheritance:

