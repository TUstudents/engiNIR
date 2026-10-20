enginir.project
===============

The typed objects of a project and their validators: datasets, their
calibrations, ROIs, targets and values, signal domains, sample sets, model
runs and prediction runs. :class:`enginir.Project` creates and stores them;
use this package directly for advanced work on project files.

.. automodule:: enginir.project
   :no-members:

.. automodule:: enginir.project._models
   :no-members:

.. automodule:: enginir.project._validation
   :no-members:

.. automodule:: enginir.project._hydration
   :no-members:

ProjectWorkspace
----------------

.. autoclass:: enginir.project.ProjectWorkspace
   :members:

hydrate_project_store
---------------------

.. autofunction:: enginir.project.hydrate_project_store

``hydrate_project_store()`` converts the metadata-first dictionary returned by
``load_project_store()`` into current typed objects and validates all id-based
relationships without importing Qt.

DatasetRef
----------

.. autoclass:: enginir.project.DatasetRef
   :members:

``DatasetRef`` validates that the cube band dimension, ``n_bands``, and
``wavelengths`` length agree.

CalibrationSpec
---------------

.. autoclass:: enginir.project.CalibrationSpec
   :members:

``CalibrationSpec`` checks on construction that it describes one executable
calibration. For ``dark_white_reference`` this means exactly one of:
embedded dark/white arrays or scalar black/white levels.

TrainingROI
-----------

.. autoclass:: enginir.project.TrainingROI
   :members:

``TrainingROI`` stores metadata and derives its canonical HDF5 spectra path
from ``roi_id``. The raw ROI spectra remain the scientific truth and are loaded
on demand from the project file.

TargetDefinition
----------------

.. autoclass:: enginir.project.TargetDefinition
   :members:

TargetAssignment
----------------

.. autoclass:: enginir.project.TargetAssignment
   :members:

SignalDomainContract
--------------------

.. autoclass:: enginir.project.SignalDomainContract
   :members:

SampleSet
---------

.. autoclass:: enginir.project.SampleSet
   :members:

``SampleSet`` fixes the modeling semantics:
``observation_unit=pixel``, ``target_level=roi``, ``grouping_unit=roi``,
``validation_unit=roi``, ``weighting_policy=per_pixel_unweighted``,
``reporting_level=roi``, and ``roi_aggregation=mean``.

ModelRun
--------

.. autoclass:: enginir.project.ModelRun
   :members:

PredictionRun
-------------

.. autoclass:: enginir.project.PredictionRun
   :members:

ProjectRelationshipError
------------------------

.. autoexception:: enginir.project.ProjectRelationshipError

validate_project_relationships
------------------------------

.. autofunction:: enginir.project.validate_project_relationships

The validator checks id-based references between first-class project objects
and logs failures before raising ``ProjectRelationshipError``.

Signal domain
-------------

.. automodule:: enginir.project._signal_domain
   :no-members:

.. autofunction:: enginir.project.transform_spectra_to_signal_domain

.. autofunction:: enginir.project.check_signal_domain_compatibility

.. autofunction:: enginir.project.check_calibration_signal_domain_compatibility

.. autoclass:: enginir.project.CalibrationStats
   :members:

.. autoexception:: enginir.project.SignalDomainCompatibilityError
