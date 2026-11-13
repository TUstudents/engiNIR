enginir.modeling
================

.. automodule:: enginir.modeling
   :no-members:

.. automodule:: enginir.modeling._service
   :no-members:

The fixed modeling semantics of a calibration set (pixels as the data points of the fit, ROI
groups for validation, ROI-mean reporting) are listed in
:class:`enginir.project.SampleSet`.

.. autofunction:: enginir.modeling.build_sample_set

.. autofunction:: enginir.modeling.plan_sample_set

.. autofunction:: enginir.modeling.fit_model_run

.. autofunction:: enginir.modeling.stored_hyperparameters

.. autofunction:: enginir.modeling.aggregate_sample_set_predictions

.. autofunction:: enginir.modeling.roi_to_signal_domain

.. autofunction:: enginir.modeling.roi_spectrum_summaries

.. autofunction:: enginir.modeling.propose_bands

.. autofunction:: enginir.modeling.fold_warnings

.. autofunction:: enginir.modeling.predicted_fold_warnings

.. autofunction:: enginir.modeling.fold_warning_text

.. autoclass:: enginir.modeling.BandProposal
   :no-members:

.. autoclass:: enginir.modeling.BandStep
   :no-members:

.. autoclass:: enginir.modeling.BandProposalCancelledError
   :no-members:

.. autoclass:: enginir.modeling.RoiSpectrumSummary
   :no-members:

.. autoclass:: enginir.modeling.SampleSetData
   :members:

.. autoclass:: enginir.modeling.ModelRunResult
   :members:

The saved ``prediction_model`` artifact holds ``coef_`` and ``intercept_`` of
the fitted model (``y = X @ coef_ + intercept_`` on preprocessed
spectra in the model's signal), the wavelengths, the target name and, for MSC/EMSC
recipes, the fitted preprocessing state.
