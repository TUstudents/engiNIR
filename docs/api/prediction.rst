enginir.prediction
==================

.. automodule:: enginir.prediction
   :no-members:

.. automodule:: enginir.prediction._service
   :no-members:

Running predictions
-------------------

.. autofunction:: enginir.prediction.predict_with_model_run

.. autofunction:: enginir.prediction.preflight_prediction

.. autofunction:: enginir.prediction.build_dataset_ref_from_spectral_data

.. autofunction:: enginir.prediction.wavelength_mismatch

Prediction ROIs
---------------

.. automodule:: enginir.prediction._rois
   :no-members:

.. autofunction:: enginir.prediction.predict_rois

.. autofunction:: enginir.prediction.roi_prediction_statistics

.. autoclass:: enginir.prediction.RoiPrediction
   :members: dmodx_limit_at

.. autoclass:: enginir.prediction.RoiPixels
   :no-members:

.. autoclass:: enginir.prediction.RoiStatistics
   :no-members:

.. autoclass:: enginir.prediction.RoiPredictionSummary
   :no-members:

Prediction intervals
--------------------

.. automodule:: enginir.prediction._uncertainty
   :no-members:

.. autofunction:: enginir.prediction.pixel_half_width

.. autofunction:: enginir.prediction.mean_half_width

.. autoclass:: enginir.prediction.IntervalInputs
   :no-members:

Results and progress
--------------------

.. autoclass:: enginir.prediction.PredictionResult
   :members:

.. autoclass:: enginir.prediction.PredictionPreflight
   :members:

.. autoclass:: enginir.prediction.PredictionProgressEvent
   :members:

.. autodata:: enginir.prediction.PredictionProgressCallback

.. autodata:: enginir.prediction.PredictionCancelCallback

.. autoexception:: enginir.prediction.PredictionCancelledError

Model runs using MSC or EMSC store their fitted preprocessing state in the
prediction-model artifact; prediction restores it and never refits on
prediction data.
