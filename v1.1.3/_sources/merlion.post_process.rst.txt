merlion.post\_process package
=============================
This package implements some simple rules to post-process the output of an
anomaly detection model. This includes rules for reshaping a sequence to follow
a standard normal distribution (:py:mod:`merlion.post_process.calibrate`), sparsifying
a sequence based on a threshold (:py:mod:`merlion.post_process.threshold`), and composing
together sequences of post-processing rules (:py:mod:`merlion.post_process.sequence`).

.. automodule:: merlion.post_process
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
    base
    factory
    sequence
    calibrate
    threshold

Submodules
----------

merlion.post\_process.base module
---------------------------------

.. automodule:: merlion.post_process.base
   :members:
   :undoc-members:
   :show-inheritance:

merlion.post\_process.factory module
------------------------------------

.. automodule:: merlion.post_process.factory
   :members:
   :undoc-members:
   :show-inheritance:

merlion.post\_process.sequence module
-------------------------------------

.. automodule:: merlion.post_process.sequence
   :members:
   :undoc-members:
   :show-inheritance:

.. _merlion.post_process.calibrate:

merlion.post\_process.calibrate module
--------------------------------------

.. automodule:: merlion.post_process.calibrate
   :members:
   :undoc-members:
   :show-inheritance:

merlion.post\_process.threshold module
--------------------------------------

.. automodule:: merlion.post_process.threshold
   :members:
   :undoc-members:
   :show-inheritance:
