"""Sample sets and model runs from project ROIs (headless, Qt-free).

:func:`build_sample_set` collects labelled ROI pixels; :func:`fit_model_run`
cross-validates (ROI-grouped) and fits a PLS or PCR model on them and
describes the result as a :class:`~enginir.project.ModelRun`.
"""

from ._service import (
    ModelRunResult,
    SampleSetData,
    aggregate_sample_set_predictions,
    build_sample_set,
    fit_model_run,
)

__all__ = [
    "ModelRunResult",
    "SampleSetData",
    "aggregate_sample_set_predictions",
    "build_sample_set",
    "fit_model_run",
]
