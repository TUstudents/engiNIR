enginir.Project
===============

:class:`Project` runs the project workflow in one call per step: register
datasets, define targets, store ROIs with reference values, fit models and
predict. It creates calibrations, sample sets, signal domains, stored recipes
and provenance itself; see :doc:`/user_guide/workflows` for a complete
example.

.. automodule:: enginir._workflow
   :no-members:

Project
-------

.. autoclass:: enginir.Project
   :members:

Dataset
-------

.. autoclass:: enginir._workflow.Dataset
   :members:

Model
-----

.. autoclass:: enginir._workflow.Model
   :members:

Prediction
----------

.. autoclass:: enginir._workflow.Prediction
   :members:

StoredPrediction
----------------

.. autoclass:: enginir._workflow.StoredPrediction
   :members:

Shared rules
------------

The viewer and :class:`Project` resolve datasets and calibrations, build
ROIs, and find the sample set and signal domain of a fit with these functions, so a project written from a script and one written
in the viewer follow the same rules.

.. autofunction:: enginir._workflow.resolve_dataset

.. autofunction:: enginir._workflow.make_calibration

.. autofunction:: enginir._workflow.resolve_calibration

.. autofunction:: enginir._workflow.resolve_prediction_calibration

.. autofunction:: enginir._workflow.make_roi

.. autofunction:: enginir._workflow.resolve_sample_set

.. autofunction:: enginir._workflow.resolve_signal_domain

.. autofunction:: enginir._workflow.recipe_id_for

.. autofunction:: enginir._workflow.check_data_file_location

.. autofunction:: enginir._workflow.data_file_labels

.. autofunction:: enginir._workflow.read_header

.. autofunction:: enginir._workflow.read_frames

.. autofunction:: enginir._workflow.roi_grid

.. autoclass:: enginir._workflow.GridCell
   :no-members:

Calibration defaults
--------------------

.. autodata:: enginir._workflow.DEFAULT_BLACK_LEVEL

.. autodata:: enginir._workflow.DEFAULT_WHITE_LEVEL
