"""Applying saved model runs to new datasets (headless, Qt-free).

:func:`predict_with_model_run` predicts a whole dataset chunk by chunk and can
stream the predictions into the project file; :func:`preflight_prediction`
performs its checks without predicting. The signal-domain checks and
conversion it relies on live in :mod:`enginir.project`.
"""

from ._service import (
    DEFAULT_CHUNK_SIZE,
    PredictionCancelCallback,
    PredictionCancelledError,
    PredictionPreflight,
    PredictionProgressCallback,
    PredictionProgressEvent,
    PredictionResult,
    build_dataset_ref_from_spectral_data,
    predict_with_model_run,
    preflight_prediction,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "PredictionCancelCallback",
    "PredictionCancelledError",
    "PredictionPreflight",
    "PredictionProgressCallback",
    "PredictionProgressEvent",
    "PredictionResult",
    "build_dataset_ref_from_spectral_data",
    "predict_with_model_run",
    "preflight_prediction",
]
