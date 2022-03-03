merlion.transform package
=========================
This package provides a number of useful data pre-processing transforms. Each
transform is a callable object that inherits either from `TransformBase` or
`InvertibleTransformBase`.

We will introduce the key features of transform objects using the `Rescale`
class. You may initialize a ``transform`` in three ways:

.. code-block:: python

    from merlion.transform.factory import TransformFactory
    from merlion.transform.normalize import Rescale

    # Use the initializer
    transform = Rescale(bias=5.0, scale=3.2)

    # Use the class's from_dict() method with the arguments you would normally
    # give to the initializer
    kwargs = dict(bias=5.0, scale=3.2)
    transform = Rescale.from_dict(kwargs)

    # Use the TransformFactory with the class's name, and the keyword arguments
    # you would normally give to the inializer
    transform = TransformFactory.create("Rescale", **kwargs)

After initializing a ``transform``, one may use it as follows:

.. code-block:: python

    transform.train(time_series)              # set any trainable params
    transformed = transform(time_series)      # apply the transform to the time series
    inverted = transform.invert(transformed)  # invert the transform
    state_dict = transform.to_dict()          # serialize to a JSON-compatible dict

Note that ``transform.invert()`` is supported even if the transform doesn't
inherit from `InvertibleTransformBase`! In this case, ``transform.invert()``
implements a *pseudo*-inverse that may not recover the original ``time_series``
exactly. Additionally, the dict returned by ``transform.to_dict()`` is exactly
the same as the dict expected by the class method ``TransformCls.from_dict()``.

.. automodule:: merlion.transform
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
    factory
    base
    bound
    moving_average
    normalize
    resample
    sequence
    anomalize

Submodules
----------

merlion.transform.base module
-----------------------------

.. automodule:: merlion.transform.base
   :members:
   :undoc-members:
   :show-inheritance:

merlion.transform.bound module
------------------------------

.. automodule:: merlion.transform.bound
   :members:
   :undoc-members:
   :show-inheritance:

merlion.transform.factory module
--------------------------------

.. automodule:: merlion.transform.factory
   :members:
   :undoc-members:
   :show-inheritance:

merlion.transform.moving\_average module
----------------------------------------

.. automodule:: merlion.transform.moving_average
   :members:
   :undoc-members:
   :show-inheritance:

merlion.transform.normalize module
----------------------------------

.. automodule:: merlion.transform.normalize
   :members:
   :undoc-members:
   :show-inheritance:

merlion.transform.resample module
---------------------------------

.. automodule:: merlion.transform.resample
   :members:
   :undoc-members:
   :show-inheritance:

merlion.transform.sequence module
---------------------------------

.. automodule:: merlion.transform.sequence
   :members:
   :undoc-members:
   :show-inheritance:


merlion.transform.anomalize module
----------------------------------

.. automodule:: merlion.transform.anomalize
   :members:
   :undoc-members:
   :show-inheritance:
