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

Every plot has a header with its title, a maximize button and a ⋯ menu that
saves the current view as PNG or SVG, exports the plotted values as CSV, or
copies the image. The Spatial Image and the Spectral Profile maximize within
Explore. A docked plot floats over the whole window instead. The same
button, a double-click on the header or ``Esc`` restores it. The CSV records
the project, the data file, the signal unit and the frame, row or band shown.

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
pixels and the calibration in use, and appears in the browser. The status bar
confirms each write, for example "✓ ROI 'tablet 3' saved to project".
**Delete ROI** removes the selected ROI; **Export CSV** writes the ROI spectra
to a CSV file.

A project can collect ROIs from several data files. The image shows only the
ROIs drawn on the data file it displays; the browser lists all ROIs of the
project. Attaching a data file the project does not know yet is not a
mismatch. The viewer warns only when a registered data file has changed (a
different format, shape or wavelength axis), and names the ROIs drawn from
it.

Model
-----

Model shows the selected model and its diagnostics:

- summary cards for target, sample set, model, components, cross-validation,
  ROI and pixel counts, R², RMSECV and bias
- the **Fit** page: predicted vs reference (with R², RMSECV and bias),
  residuals, the distribution of reference values, and the training scores of
  the first two latent variables
- the **Spectral** page: the regression coefficients and the loadings of the
  first latent variable
- click a point to select its ROI

Each plot has a header with its title. The maximize button, or a double-click
on the header, makes the plot fill the page; the same button, a double-click
or ``Esc`` restores it. The ⋯ menu saves the plot as PNG or SVG (300 dpi,
light background) or copies it to the clipboard. **Export CSV** writes the
plotted values. The CSV starts with ``#`` lines that record the project,
model run, sample set, recipe, components and cross-validation it came from.

**File > Export Plots...** saves every plot of the current workspace (PNG and
CSV, named by plot) into one folder. That includes Model pages you have not
opened. A maximized plot stays maximized the next time the viewer opens, and
**View > Reset Layout** restores the default layout.

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

The workspace shows the prediction map, the distribution of the predicted
values, the model, dataset and calibration used, the compatibility summary,
and how many reflectance values were clipped.

The map header has the output selector (for models with several targets),
**Pan**, **Zoom** and **Reset View**. Large maps are shown as a preview that
keeps every n-th frame and row. The axes, the hover readout and the exported
CSV still give the true frame and row, and the CSV records the step. Map and
distribution export like every other plot (PNG, SVG, CSV, copy) and maximize
within the page.
Prediction streams through the scan in bounded memory and reports its phase
(loading, calibrating, preprocessing, predicting, writing).

Projects
--------

A project collects ROIs from any number of data files. **File > Add Data
File...** (or **Open Data...** while a project is open) adds a file to the
project in the same window and asks how it is calibrated:

- **Dark and white references**, each from a file or from frames of the data
  file itself, such as a white tile or shutter-closed frames at the start of
  the tray;
- **Same references as another data file**, for one reference set per session;
- **Detector levels** (the file's, or black 128 / white 3500);
- **Raw counts**.

The dialog plots the mean dark and white spectra and reports the share of
saturated values before you accept them. Files with floating-point values are
taken as already calibrated.

The data file list in the Spatial Image header switches between the
project's files, as does a double-click on a file in the browser. Each file
is shown with its own calibration and its own ROIs. A file that was moved or
deleted is listed as "(not found)"; its ROIs, values and models stay usable.
Drawing ROIs keeps the calibration of the file. If you load other references
first, the viewer asks before new ROIs use them, and the ROIs drawn before
keep their own.

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
