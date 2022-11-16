forecast
========

.. automodule:: merlion.models.forecast
   :members:
   :undoc-members:
   :show-inheritance:

Base classes:

.. autosummary::
    base
    sklearn_base

Univariate models:

.. autosummary::
    arima
    sarima
    ets
    prophet
    smoother
    lstm

`Multivariate <tutorials/forecast/2_ForecastMultivariate>` models:

.. autosummary::
    vector_ar
    trees

`Exogenous regressor <tutorials/forecast/3_ForecastExogenous>` models:

.. autosummary::
    trees
    prophet
    sarima
    vector_ar
    arima

Note that the AutoML variants
:py:mod:`AutoSarima <merlion.models.automl.autosarima>` and
:py:mod:`AutoProphet <merlion.models.automl.autoprophet>`
also support exogenous regressors.


Base classes
------------

forecast.base
^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.base
   :members:
   :undoc-members:
   :show-inheritance:

forecast.sklearn\_base
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.sklearn_base
   :members:
   :undoc-members:
   :show-inheritance:

Univariate models
-----------------

forecast.arima
^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.arima
   :members:
   :undoc-members:
   :show-inheritance:

forecast.sarima
^^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.sarima
   :members:
   :undoc-members:
   :show-inheritance:

forecast.ets
^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.ets
   :members:
   :undoc-members:
   :show-inheritance:


forecast.prophet
^^^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.prophet
   :members:
   :undoc-members:
   :show-inheritance:

forecast.smoother
^^^^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.smoother
   :members:
   :undoc-members:
   :show-inheritance:

forecast.lstm
^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.lstm
   :members:
   :undoc-members:
   :show-inheritance:

Multivariate models
-------------------

forecast.vector\_ar
^^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.vector_ar
   :members:
   :undoc-members:
   :show-inheritance:

forecast.trees
^^^^^^^^^^^^^^
.. automodule:: merlion.models.forecast.trees
   :members:
   :undoc-members:
   :show-inheritance:
