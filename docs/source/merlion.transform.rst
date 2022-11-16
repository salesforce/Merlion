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

Base primitives:

.. autosummary::
    factory
    base
    sequence

Resampling:

.. autosummary::
    resample
    moving_average

Normalization:

.. autosummary::
    bound
    normalize

Miscellaneous

.. autosummary::
    anomalize

Base primitives
---------------

transform.factory
^^^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.factory
   :members:
   :undoc-members:
   :show-inheritance:

transform.base
^^^^^^^^^^^^^^
.. automodule:: merlion.transform.base
   :members:
   :undoc-members:
   :show-inheritance:

transform.sequence
^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.sequence
   :members:
   :undoc-members:
   :show-inheritance:

Resampling
----------

transform.resample
^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.resample
   :members:
   :undoc-members:
   :show-inheritance:

transform.moving\_average
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.moving_average
   :members:
   :undoc-members:
   :show-inheritance:

Normalization
-------------

transform.normalize
^^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.normalize
   :members:
   :undoc-members:
   :show-inheritance:

transform.bound
^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.bound
   :members:
   :undoc-members:
   :show-inheritance:


Miscellaneous
-------------

transform.anomalize
^^^^^^^^^^^^^^^^^^^
.. automodule:: merlion.transform.anomalize
   :members:
   :undoc-members:
   :show-inheritance:
