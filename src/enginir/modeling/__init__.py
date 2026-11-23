"""Sample sets and model runs from project ROIs (headless, Qt-free).

:func:`build_sample_set` collects labelled ROI pixels; :func:`fit_model_run`
cross-validates (ROI-grouped) and fits a PLS or PCR model on them and
describes the result as a :class:`~enginir.project.ModelRun`.
:func:`propose_bands` proposes band ranges by forward interval PLS with
nested validation.
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_names

if TYPE_CHECKING:  # the names load on first use (PEP 562): see enginir._lazy
    from ._bands import BandProposal, BandProposalCancelledError, BandStep, propose_bands
    from ._roi_spectra import RoiSpectrumSummary, roi_spectrum_summaries, roi_to_signal_domain
    from ._service import (
        ModelRunResult,
        SampleSetData,
        aggregate_sample_set_predictions,
        build_sample_set,
        fit_model_run,
        fold_warning_text,
        fold_warnings,
        plan_sample_set,
        predicted_fold_warnings,
        stored_hyperparameters,
    )

__getattr__, __dir__ = lazy_names(__name__, globals(), {
    **dict.fromkeys(("BandProposal", "BandProposalCancelledError", "BandStep",
                     "propose_bands"), "._bands"),
    **dict.fromkeys(("ModelRunResult", "RoiSpectrumSummary", "SampleSetData",
                     "aggregate_sample_set_predictions", "build_sample_set", "fit_model_run",
                     "fold_warnings", "predicted_fold_warnings"), "._service"),
    # light: read by the viewer before the modelling libraries are loaded
    **dict.fromkeys(("fold_warning_text", "stored_hyperparameters"), "._stored"),
    **dict.fromkeys(("RoiSpectrumSummary", "roi_spectrum_summaries", "roi_to_signal_domain"),
                    "._roi_spectra"),
    "plan_sample_set": "._sample_set",
})

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
    "stored_hyperparameters",
]
