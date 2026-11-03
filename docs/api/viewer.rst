enginir.viewer
==============

.. autofunction:: enginir.viewer.ensure_app
.. autofunction:: enginir.viewer.run
.. autofunction:: enginir.viewer.open_viewer
.. autofunction:: enginir.viewer.open_project

Start the viewer with :func:`open_viewer` (a scan, optionally with dark and
white references and a project) or :func:`open_project` (a project on its
own; :func:`open_viewer` also detects a project file). :func:`ensure_app`
and :func:`run` start Qt from a script. The user guide (:doc:`/user_guide/viewer`)
describes the workspaces.

The classes below are the viewer's building blocks, documented for
developers. ``HSIViewer`` is the main window; the Explore, Model and Predict
workspaces are ``EDAView``, ``ModelingView`` and ``PredictionView``. All of
them read from ``ViewerState`` (the open scan, cursor and processing) and
``ProjectState`` (the project). To script the workflow, use
:class:`enginir.Project` instead (:doc:`/api/workflow`).

WelcomeWindow
-------------

.. autoclass:: enginir.viewer._welcome.WelcomeWindow
   :members: file_selected, project_selected, project_created

Loading helpers
---------------

.. automodule:: enginir.viewer._loading
   :no-members:

.. autoclass:: enginir.viewer._jobs.BusyStatusWidget
   :members: set_idle, set_busy, set_progress, snapshot

.. autoclass:: enginir.viewer._loading.ViewerBusyContext
   :members: begin, end

Recent files
------------

.. autofunction:: enginir.viewer._recent.config_dir
.. autofunction:: enginir.viewer._recent.load
.. autofunction:: enginir.viewer._recent.add
.. autofunction:: enginir.viewer._recent.remove

ViewerState
-----------

.. autoclass:: enginir.viewer.ViewerState
   :members:
   :undoc-members:
   :exclude-members: staticMetaObject, frame_changed, row_changed, band_changed, units_changed

ProjectState
------------

``ProjectState`` is the viewer's Qt adapter over :class:`enginir.Project`.
It mirrors the project's collections, keyed by stable id (``datasets``,
``active_calibration_ids``, ``calibrations``, ``rois``, ``targets``,
``target_assignments``, ``recipes``, ``signal_domains``, ``sample_sets``,
``model_runs``, ``predictions``), stores every change through the project
and emits a signal for it. It holds no workflow rule of its own. When one
window writes, every other window of the process on the same project
re-reads it (``reload``) on the next event-loop pass.

.. autoclass:: enginir.viewer.ProjectState
   :members: new_project, open_project, reload, locate_data_file, register_dataset, set_active_calibration, define_target, assign_target_to_roi, add_roi, register_recipe, register_signal_domain, create_sample_set, build_sample_set, register_model_run, fit_model_run, register_prediction_run, predict_dataset, remove_roi, get_target_assignment
   :exclude-members: staticMetaObject, project_loaded, datasets_changed, calibrations_changed, rois_changed, targets_changed, target_assignments_changed, recipes_changed, signal_domains_changed, sample_sets_changed, model_runs_changed, predictions_changed, model_fitted, active_model_changed, prediction_completed

EDA view
--------

.. automodule:: enginir.viewer._eda
   :no-members:

.. autoclass:: enginir.viewer._eda.EDAView
   :no-members:

Modeling view
-------------

.. automodule:: enginir.viewer._modeling
   :no-members:

.. autoclass:: enginir.viewer._modeling.ModelingView
   :no-members:

.. autoclass:: enginir.viewer._modeling.NewROIDialog
   :members: roi_name, target_value

.. autoclass:: enginir.viewer._modeling.TargetSelectionDialog
   :members: target_name, target_unit, selected_target_id, creates_target

.. autoclass:: enginir.viewer._modeling.FitModelRunDialog
   :members: target_id, recipe, set_recipe, model_type, n_components, signal_unit, cv

Prediction view
---------------

.. automodule:: enginir.viewer._prediction
   :no-members:

.. autoclass:: enginir.viewer._prediction.PredictionView
   :no-members:

.. autoclass:: enginir.viewer._prediction.PredictionRunDialog
   :members: model_run_id, data_path, dataset_id, calibration

Panels
------

.. autoclass:: enginir.viewer._panels.HistogramPanel
   :no-members:

.. autoclass:: enginir.viewer._panels.DC3Panel
   :no-members:

.. autoclass:: enginir.viewer._panels.SensorPanel
   :no-members:

.. autoclass:: enginir.viewer._panels.SpatialProfilePanel
   :no-members:

.. autoclass:: enginir.viewer._panels.SpectrumPanel
   :no-members:

Workspace surfaces
------------------

.. automodule:: enginir.viewer._model_diagnostics
   :no-members:

.. autoclass:: enginir.viewer._model_diagnostics.ModelDiagnosticsWidget
   :no-members:

.. autoclass:: enginir.viewer._prediction.PredictionMapWidget
   :no-members:

Plot panels
-----------

.. automodule:: enginir.viewer._plot_panel
   :no-members:

.. autoclass:: enginir.viewer._plot_panel.PlotSpec
   :no-members:

.. autoclass:: enginir.viewer._plot_panel.PlotTable
   :no-members:

.. autoclass:: enginir.viewer._plot_panel.PlotFrame
   :no-members:

.. autoclass:: enginir.viewer._plot_panel.PlotPanel
   :no-members:

.. autofunction:: enginir.viewer._plot_panel.print_copy

.. automodule:: enginir.viewer._plot_grid
   :no-members:

.. autoclass:: enginir.viewer._plot_grid.PlotGrid
   :no-members:

.. autoclass:: enginir.viewer._plot_grid.PlotPageWidget
   :no-members:

.. automodule:: enginir.viewer._plot_registry
   :no-members:

HSIViewer
---------

.. autoclass:: enginir.viewer._viewer.HSIViewer
   :members: show, export_csv, export_workspace_plots, workspace_plot_frames, reset_layout,
             add_data_file, show_data_file, wait_for_data_file, apply_shown_calibration,
             locate_data_file, show_roi_table, show_project_rois, open_roi

Data files
----------

.. automodule:: enginir.viewer._data_files
   :no-members:

.. autoclass:: enginir.viewer._data_files.AddDataFileDialog
   :no-members:

.. autoclass:: enginir.viewer._data_files.DataFileRequest
   :members: add_dataset_kwargs

Project ROIs
------------

.. automodule:: enginir.viewer._project_rois
   :no-members:

.. autoclass:: enginir.viewer._project_rois.ProjectRoisView
   :no-members:

.. autoclass:: enginir.viewer._project_rois.RoiMosaic
   :no-members:

.. autofunction:: enginir.viewer._project_rois.mosaic_layout

ROI table
---------

.. automodule:: enginir.viewer._roi_table
   :no-members:

.. autoclass:: enginir.viewer._roi_table.RoiTableWidget
   :members: show_data_file, focus_target, select_roi, paste_clipboard

.. autofunction:: enginir.viewer._roi_table.parse_value

ROI grid
--------

.. automodule:: enginir.viewer._roi_grid
   :no-members:

.. autoclass:: enginir.viewer._roi_grid.RoiGridDialog
   :members: set_values, set_grid, toggle_cell, redraw_preview

.. autoclass:: enginir.viewer._roi_grid.GridDefaultsDialog
   :no-members:

.. autoclass:: enginir.viewer._grid_settings.GridDefaults
   :no-members:

.. autofunction:: enginir.viewer._grid_settings.load_grid_defaults

.. autofunction:: enginir.viewer._grid_settings.save_grid_defaults

Project browser and inspector
-----------------------------

.. automodule:: enginir.viewer._browser
   :no-members:

.. autoclass:: enginir.viewer._browser.ProjectBrowser
   :no-members:

.. automodule:: enginir.viewer._inspector
   :no-members:

.. autoclass:: enginir.viewer._inspector.ProjectInspector
   :no-members:
