enginir.validation
==================

Metrics
-------

.. autofunction:: enginir.validation.metrics.rmsec
.. autofunction:: enginir.validation.metrics.rmsecv
.. autofunction:: enginir.validation.metrics.rmsep
.. autofunction:: enginir.validation.metrics.bias
.. autofunction:: enginir.validation.metrics.sep
.. autofunction:: enginir.validation.metrics.r2_score
.. autofunction:: enginir.validation.metrics.rpd
.. autofunction:: enginir.validation.metrics.rpiq
.. autofunction:: enginir.validation.metrics.metrics_report

Cross-validation
----------------

.. autofunction:: enginir.validation.cross_validate
.. autoclass:: enginir.validation.RoundRobinGroups
   :members:
.. autofunction:: enginir.validation.kennard_stone
.. autodata:: enginir.validation.DEFAULT_GROUPED_N_SPLITS

Aggregation
-----------

.. autofunction:: enginir.validation.aggregate_regression_predictions

Number of components
--------------------

.. autofunction:: enginir.validation.choose_components

.. autodata:: enginir.validation.COMPONENT_CHOICE_RULES

.. autofunction:: enginir.validation.guarded_choice

.. autofunction:: enginir.validation.noise_components

.. autofunction:: enginir.validation.structure_ceiling

Outlier detection
-----------------

.. autofunction:: enginir.validation.outliers.leverage_outliers
.. autofunction:: enginir.validation.outliers.hotelling_t2
.. autofunction:: enginir.validation.outliers.spectral_outliers
.. autofunction:: enginir.validation.outliers.dmodx
.. autofunction:: enginir.validation.outliers.dmodx_limit
.. autodata:: enginir.validation.LIMIT_LEVEL
.. autofunction:: enginir.validation.outliers.modelling_power
