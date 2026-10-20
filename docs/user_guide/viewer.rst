Viewer
======

``enginir-viewer`` is a desktop application for hyperspectral images, built
on PyQt6 and pyqtgraph. It runs the whole workflow interactively in three
workspaces and stores every step in a project file through
:class:`enginir.Project`, so a project edited in the viewer and one written
from Python follow the same rules. :doc:`/hsi_viewer` is the full reference
of controls and panels.

Starting
--------

.. code-block:: bash

   enginir-viewer                                     # welcome screen with recent files
   enginir-viewer scan.eiff                           # open a scan
   enginir-viewer scan.eiff --dark dark.eiff --white white.eiff
   enginir-viewer scan.eiff --project project.h5      # scan and project together
   enginir-viewer project.h5                          # a project on its own

From Python:

.. code-block:: python

   from enginir.viewer import open_project, open_viewer, run

   viewer = open_viewer("scan.eiff", dark="dark.eiff", white="white.eiff")
   project_viewer = open_project("project.h5")
   run()  # blocks until the window closes; not needed in Jupyter

The welcome screen opens a scan, opens a project or creates a new one. Files
load in the background with a progress indicator.

Window
------

Around the three workspace tabs the window keeps:

- a status strip with project, dataset, signal, frame, row and band
- the **Project Browser**, listing datasets, calibrations, ROIs, targets,
  values, sample sets, signal domains, models and predictions, and the
  **Inspector**, showing the details of the selected item
- a **Log** dock (collapsed until opened); **Help > Log File** shows where
  the log file is (by default ``~/.enginir/enginir.log``)

A project opened without a scan shows its stored objects in every
workspace; drawing ROIs needs a scan.

Explore
-------

The **Spatial Image** canvas is the centre of Explore: click to move the
cursor, pan and zoom, and draw ROIs. Docked panels show the data at the
cursor:

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Panel
     - Shows
     - Dock
   * - **Histogram**
     - intensity distribution of the current frame
     - left
   * - **RAW**
     - the cube at the selected band (height × frames), raw counts
     - left
   * - **Sensor Frame**
     - the current frame (height × bands), processed
     - right
   * - **Spatial Profile**
     - intensity along the frame at the selected band, with a draggable row
     - right
   * - **Spectral Profile**
     - the spectrum at the selected row, with a draggable band
     - bottom

The controls set what the panels show:

- **Signal**: raw counts, reflectance or absorbance. Without dark and white
  references, the file's detector levels or the EVK defaults (black 128,
  white 3500) are used.
- **Filtering**: Savitzky-Golay smoothing, first or second derivative, with
  window and polynomial order.
- **Normalization**: SNV, MSC, EMSC and detrending, combinable.
- **Axis**: wavelength or band index; pixels or millimetres.

Processing is always applied in the same order as in model training: signal
conversion, then filtering, then normalization (SNV, MSC, EMSC, detrend).

To record training data:

1. Create or open a project (**Project > New** or **Open**).
2. Choose the target to model with **Target...** (or define a new one).
3. Click **Add ROI** and drag a rectangle on the Spatial Image.
4. Name the ROI and, optionally, enter its reference value.

The ROI is stored at once with its raw spectra, its dark and white reference
pixels and the calibration in use, and appears in the browser. **Delete ROI**
removes the selected ROI; **Export CSV** writes the ROI spectra to a CSV file.

Model
-----

Model shows the selected model and its diagnostics:

- summary cards for target, sample set, model, components, cross-validation,
  ROI and pixel counts, R², RMSECV and bias
- predicted vs reference, residuals, the distribution of reference values,
  the scores of the first two latent variables, and the regression
  coefficients or loadings (selectable)
- click a point to select its ROI

The controls assign reference values (**Assign target**), build the sample
set of all ROIs with a value (**Build sample set**) and fit PLS or PCR
(**Fit model**). Cross-validation keeps the pixels of one ROI together, and
reported metrics are per ROI: held-out pixel predictions are averaged per ROI
before RMSECV, R², bias, RPD and RPIQ are computed. Until a model can be
fitted, the readiness message says what is missing.

Predict
-------

**Predict...** applies the selected model to a scan. The scan is registered
in the project (or its registration and calibration are reused), and the
prediction is refused before it starts when the scan cannot produce the
model's signal: another instrument format, number of bands, calibration mode
or signal unit. Wavelength values are metadata and are not compared. Models
with MSC or EMSC apply the preprocessing state stored at training.

The workspace shows the prediction map, the model, dataset and calibration
used, the compatibility summary, and how many reflectance values were clipped.
Prediction streams through the scan in bounded memory and reports its phase
(loading, calibrating, preprocessing, predicting, writing).

Projects
--------

There is no save step: every ROI, value, sample set, model and prediction is
written when it is made. ROIs and targets are identified by stable ids; names
are labels. Projects open metadata-first, so ROI spectra and model or
prediction arrays are read only when a plot or job needs them. To script
what the viewer does, use :class:`enginir.Project` (:doc:`workflows`).

From Python
-----------

The viewer state can be driven from code:

.. code-block:: python

   viewer.state.set_y_unit("Absorbance")
   viewer.state.set_preprocess_filter("sg_d1")
   viewer.state.set_sg_window(15)
   viewer.state.set_preprocess_snv(True)
   viewer.export_csv("rois.csv")

See :doc:`logging` for log output in scripts and notebooks.
