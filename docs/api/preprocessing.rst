enginir.preprocessing
=====================

Scatter Correction
------------------

.. autoclass:: enginir.preprocessing.SNV
   :members: fit, transform

.. autoclass:: enginir.preprocessing.MSC
   :members: fit, transform

.. autoclass:: enginir.preprocessing.EMSC
   :members: fit, transform

Derivatives
-----------

.. autoclass:: enginir.preprocessing.SavitzkyGolay
   :members: fit, transform

Baseline Correction
-------------------

.. autoclass:: enginir.preprocessing.AsLSBaseline
   :members: fit, transform

.. autoclass:: enginir.preprocessing.Detrend
   :members: fit, transform

.. autoclass:: enginir.preprocessing.Smoother
   :members: fit, transform

Recipe
------

.. autoclass:: enginir.preprocessing.PreprocessingRecipe
   :members: to_dict, from_dict, build_steps, output_wavelengths, sg_trim, asls_direction, describe
   :member-order: bysource
