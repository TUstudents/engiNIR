"""Sample sets and model runs from project ROIs (headless, Qt-free).

:func:`build_sample_set` collects labelled ROI pixels; :func:`fit_model_run`
cross-validates (ROI-grouped) and fits a PLS or PCR model on them and
describes the result as a :class:`~enginir.project.ModelRun`.
:func:`propose_bands` proposes band ranges by forward interval PLS with
nested validation.
"""

from ._bands import BandProposal, BandProposalCancelledError, BandStep, propose_bands
from ._service import (
    ModelRunResult,
    RoiSpectrumSummary,
    SampleSetData,
    aggregate_sample_set_predictions,
    build_sample_set,
    fit_model_run,
    fold_warning_text,
    fold_warnings,
    plan_sample_set,
    predicted_fold_warnings,
    roi_spectrum_summaries,
    roi_to_signal_domain,
)

__all__ = [
    "BandProposal",
    "BandProposalCancelledError",
    "BandStep",
    "ModelRunResult",
    "RoiSpectrumSummary",
    "SampleSetData",
    "aggregate_sample_set_predictions",
    "build_sample_set",
    "fit_model_run",
    "fold_warning_text",
    "fold_warnings",
    "plan_sample_set",
    "predicted_fold_warnings",
    "propose_bands",
    "roi_spectrum_summaries",
    "roi_to_signal_domain",
]
