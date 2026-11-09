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

Band Selection
--------------

.. autoclass:: enginir.preprocessing.BandSelection
   :members: fit, transform

Recipe
------

.. autoclass:: enginir.preprocessing.PreprocessingRecipe
   :members: to_dict, from_dict, build_steps, output_wavelengths, output_band_indices, selected_columns, n_bands_before_selection, has_band_selection, sg_trim, asls_direction, describe, describe_bands
   :member-order: bysource

.. autofunction:: enginir.preprocessing.recipe.format_ranges

.. autofunction:: enginir.preprocessing.recipe.parse_ranges
