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
