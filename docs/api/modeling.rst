enginir.modeling
================

.. automodule:: enginir.modeling
   :no-members:

.. automodule:: enginir.modeling._service
   :no-members:

The fixed modeling semantics of a sample set (pixels as observations, ROI
groups for validation, ROI-mean reporting) are listed in
:class:`enginir.project.SampleSet`.

.. autofunction:: enginir.modeling.build_sample_set

.. autofunction:: enginir.modeling.fit_model_run

.. autofunction:: enginir.modeling.aggregate_sample_set_predictions

.. autoclass:: enginir.modeling.SampleSetData
   :members:

.. autoclass:: enginir.modeling.ModelRunResult
   :members:

The saved ``prediction_model`` artifact holds ``coef_`` and ``intercept_`` of
the fitted model (``y = X @ coef_ + intercept_`` on preprocessed
signal-domain spectra), the wavelengths, the target name and, for MSC/EMSC
recipes, the fitted preprocessing state.
