"""Typed objects of the engiNIR project workspace (headless, Qt-free).

A project records datasets with their calibrations, ROIs with their reference
target values, preprocessing recipes, sample sets, fitted model runs and
prediction runs. :mod:`enginir.io.project_store` reads and writes them as
plain dicts; this package turns those dicts into validated objects
(:func:`hydrate_project_store`) and checks the references between them
(:func:`validate_project_relationships`). :func:`transform_spectra_to_signal_domain`
converts raw spectra into the signal domain of a model, identically for
training and prediction. The normative specification is in
``contracts/*_v3.md`` and ``docs/workspace_domain_model.md``.
"""

from ._models import (
    CalibrationSpec,
    DatasetRef,
    ModelRun,
    PredictionRun,
    SampleSet,
    SignalDomainContract,
    TargetAssignment,
    TargetDefinition,
    TrainingROI,
)
from ._signal_domain import (
    CalibrationStats,
    SignalDomainCompatibilityError,
    check_calibration_signal_domain_compatibility,
    check_signal_domain_compatibility,
    transform_spectra_to_signal_domain,
)
from ._validation import ProjectRelationshipError, validate_project_relationships
from ._hydration import ProjectWorkspace, hydrate_project_store

__all__ = [
    "CalibrationSpec",
    "CalibrationStats",
    "DatasetRef",
    "ModelRun",
    "PredictionRun",
    "ProjectRelationshipError",
    "ProjectWorkspace",
    "SampleSet",
    "SignalDomainCompatibilityError",
    "SignalDomainContract",
    "TargetAssignment",
    "TargetDefinition",
    "TrainingROI",
    "check_calibration_signal_domain_compatibility",
    "check_signal_domain_compatibility",
    "hydrate_project_store",
    "transform_spectra_to_signal_domain",
    "validate_project_relationships",
]
