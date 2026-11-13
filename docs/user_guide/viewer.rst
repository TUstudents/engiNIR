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

On Linux the viewer picks the display system itself. Under WSL (Windows
Subsystem for Linux, WSLg) it uses Wayland, so menus stay in front of the
window; Qt draws the title bar with the engiNIR icon. WSLg leaves a
drop-down list on screen after you choose an item (a WSLg fault, reported as
microsoft/wslg#1522); the viewer works around it, so lists close as usual.
On other Linux
desktops that offer both Wayland and X11 it uses X11. To choose yourself, set
``QT_QPA_PLATFORM``, for example ``QT_QPA_PLATFORM=xcb enginir-viewer`` for
X11.

Window
------

Around the three workspace tabs the window keeps:

- a status strip with project, dataset, signal, frame, row and band
- the **Project Browser**, listing datasets, reference corrections, ROIs, targets,
  values, calibration sets, input contracts, models and predictions, and the
  **Inspector**, showing the details of the selected item
- a **Log** dock (collapsed until opened); **Help > Log File** shows where
  the log file is (by default ``~/.enginir/enginir.log``)

A project opened without a scan shows its stored objects in every
workspace; drawing ROIs needs a scan.

Explore
-------

The **Spatial Image** canvas is the center of Explore: click to move the
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
the project, the dataset, the signal unit and the frame, row or band shown.

The controls set what the panels show:

- **Signal**: raw counts (DN), reflectance or absorbance. The project keeps
  the raw counts of every ROI with its references or detector levels, so any
  of them can be computed at any time. Only the signals the shown dataset
  can give are enabled: a file added as raw counts gives raw counts (DN) only, an already
  corrected file has no raw counts. Your choice is kept when you show another dataset that can give it.
- **Smoothing and derivatives**: Savitzky-Golay smoothing, first or second derivative, with
  window and polynomial order. As in SIMCA, the first and last half-window
  bands have no full window and are left out: they are blank in the spectrum
  and the sensor frame, and the model uses the remaining bands.
- **Scatter correction**: SNV, MSC, EMSC and detrending, combinable.
- **Axis**: wavelength or band index; pixels or millimeters.

Processing is always applied in the same order as in model calibration: signal
conversion, then smoothing and derivatives, then scatter correction (SNV, MSC,
EMSC, detrend).
AsLS baseline correction, which comes before smoothing and derivatives, is available in
scripts only (:class:`~enginir.preprocessing.PreprocessingRecipe`); a model
fitted with it shows "AsLS" in its name, and **Fit Again...** keeps it.

To record calibration data:

1. Create or open a project (**Project > New** or **Open**). A new project
   gets the ``.h5`` extension even when you type the name without it.
2. Choose the target to model with **Target...** (or define a new one).
3. Click **Add ROI** and drag a rectangle on the Spatial Image.
4. Name the ROI and, optionally, enter its target value.

For a tray of samples, **Add ROI grid** creates them in one go. The dialog
opens at once with the defaults from **Options > ROI Grid Defaults** and
lays out equal ROIs in pixels, with x along frames and y along rows:

- **Number**: columns (x) and rows (y).
- **ROI size**: the width and height of every ROI. Choose it smaller than a
  well to keep sample edges out.
- **Origin**: the frame and row of the first ROI's top-left corner (A1).
- **Spacing**: the gap between neighboring ROIs; the dialog shows the
  resulting pitch and the frames and rows the grid covers.
- **Names**: a pattern from the list (``{file}-{row}{column:02d}`` gives
  ``tray03-B04``; ``{n:03d}`` numbers them) or your own, and the order.

The grid is drawn on the image while you type. Click a cell button to mark
an empty position (**Select all** / **Select none** help with sparse trays);
empty positions keep their names, so B04 stays B04. ROIs outside the image are
drawn red and must be moved, shrunk or marked empty before **OK**. Names may
repeat ROIs of other datasets, for example when every scan is called
``stream_hsi.h5``: ROIs are told apart by their id and dataset, and the
dialog only says how many names repeat. **OK** stores all ROIs, or none if
one of them cannot be stored. Each ROI records its grid position and the grid
layout.

Changes in the dialog apply to that grid only: the next grid starts from the
Options defaults again. Set them once for your tray holder in **Options >
ROI Grid Defaults**, or press **Make defaults** in the grid dialog to keep
the current values for every new grid (stored in your user settings).

The ROI is stored at once with its raw spectra, its dark and white reference
pixels and the reference correction in use, and appears in the browser. The status bar
confirms each write, for example "✓ ROI 'tablet 3' saved to project".
**Delete ROI** removes the selected ROI; **Export CSV** writes the ROI spectra
to a CSV file, one row per pixel with the ROI's id (``sample_id``), name and
dataset.

A project can collect ROIs from several datasets. The image shows only the
ROIs drawn on the dataset it displays; the browser lists all ROIs of the
project. Attaching a dataset the project does not know yet is not a
mismatch. The viewer warns only when a registered dataset has changed (a
different format, shape or wavelength axis), and names the ROIs drawn from
it.

The **Project ROIs** tab of Explore shows every ROI stored in the project,
from the project file alone; the datasets are not needed:

- **ROI mosaic:** each ROI's stored pixels are redrawn as a tile, one block
  per dataset, with its name and the value of the target selected in Model.
  The tiles sit in a compact grid: the rows and columns of the ROI grid that
  made them (an empty well keeps its place), or for ROIs drawn by hand, rows
  and columns in scan order. Each tile is outlined in the color of its value,
  on the scale of the ROI mean spectra (gray without a value). Labels leave
  out what all shown ROIs share (``API_10p_L1_01`` becomes ``L1_01``);
  text grows and shrinks with the tiles when you zoom. **View** in the plot
  header shows the tiles at their positions in the scan instead, and
  stretches the contrast of each dataset on its own instead of all tiles
  together (the default, for comparing samples). A dataset filter narrows
  the mosaic down.
- **ROI mean spectra:** one line per ROI, colored by that value (ROIs
  without a value in gray).
- **Selected ROI pixels:** the median and the 5–95 % band of the selected
  ROI's pixels.

The spectra use the signal unit and preprocessing chosen in Explore. Each
ROI is converted with its own stored references, by the same function the
model fit uses. In **Raw counts (DN)** you see the stored counts. ROIs that cannot be
shown in the chosen unit, for example raw counts without references in
absorbance, are counted in the tab's status line.

The docks that show the dataset at the cursor (Histogram, RAW, Sensor
Frame, Spatial Profile) belong to the Dataset tab: on Project ROIs they are
hidden, and they come back as you left them. The ROI table takes the
right-hand column instead, on the ROIs shown; **View > ROI Table** brings it
back after you closed it.

A project with ROIs opens on this tab. Clicking a tile selects the ROI
everywhere; double-clicking it opens its dataset with the ROI selected.
Tiles and spectra load in the background, and only while the tab is shown.

Model
-----

Model shows the selected model and its diagnostics:

- summary cards for target, calibration set (its ROI count and preprocessing),
  model (by name), components, cross-validation (for example "7 folds,
  round-robin over 20 samples"), ROI and pixel counts; **Metrics per ROI**
  (R²Y of the fit, Q² = the cross-validated R², RMSEE (per ROI), RMSECV (per ROI), bias, RPD) and
  **Metrics per pixel** (the same over single spectra, which are noisier); and
  a warning when the validation is likely optimistic (see
  :doc:`fit_specification`)
- a **?** button in every plot header that explains how to read the plot
  (:doc:`model_plots` collects these texts)
- the **Fit** page: predicted vs reference (with R², RMSECV (per ROI), bias, and RMSEE (per ROI)
  and adjusted R²; a unit of ``1`` is not repeated after each number),
  residuals, the distribution of target values (one bar per value when
  there are at most 12 distinct values, as for levels 1, 4, 7, 10), and the
  calibration scores of the first two latent variables with Hotelling T²
  ellipses at 0.95 (dotted) and 0.99 (dashed); one point per ROI, the mean of
  the pixels the model was fitted on (the drawn pixels with a pixel subsample,
  without the excluded ones). There are two pairs of ellipses, named in the
  legend. The gray **pixel** ellipses are the limits for single calibration
  pixels; an ROI mean averages thousands of pixels and sits well inside
  them. The colored **ROI** ellipses are computed from the ROI means (centered on them and tilted with their covariance, with
  limits for the number of ROIs, from 4 ROIs): an ROI outside them is
  unusual among the ROIs. Models fitted with an earlier version show no
  scores until they are fitted again
- the **Spectral** page: the regression coefficients, the scaled
  coefficients (CoeffCS) and the loadings of the first three latent variables,
  to see which components are concentration and which are physics (scatter,
  particle size); below them VIP (PLS) with lines at 1.0 and 0.8 and the
  mean spectrum shaded behind, VIP sorted from the largest value with the top
  bands labeled, and the selectivity ratio (PLS and PCR); see
  :doc:`band_selection`. Plots over wavelength leave a gap where a band
  selection leaves out bands
- the **Components** page, four plots that name their level: RMSECV (per ROI) with
  its standard error and RMSEE (per ROI), per ROI, with the model's number and every
  rule's suggestion with the number that decided it (and what the noise
  guard changed); R²Y and Q² per ROI, with the gap between them shaded; each
  component on its own (Q²ₐ per ROI, ΔR²X per pixel, the Durbin–Watson value
  of its weights; noise components gray); and R²X and R²Y per pixel.
  **Fit Again...** opens the fit dialog with the model's settings, including
  its signal unit, to try another number
- the **Distance** page: the distance to the model (DModX) and Hotelling T²
  of the calibration pixels, with their 0.99 limits and the share above them.
  A 0.99 limit expects 1 % of the pixels above it; much more means the
  pixels are not one homogeneous population. The histograms show bins up to
  the 99.5th percentile and the rest as one gray bar, with the largest value
  stated. Below them, **Pixels Above the Limits, per ROI** shows for every
  ROI the share of its pixels above each limit, sorted from the largest, with
  bars colored by the target value: ROIs whose spectra the model does
  not describe (a different substrate, saturation, drift along a scan) stand
  out there. Click a bar to select the ROI.
- click a point to select its ROI

Each plot has a header with its title. The maximize button, or a double-click
on the header, makes the plot fill the page; the same button, a double-click
or ``Esc`` restores it. The ⋯ menu saves the plot as PNG or SVG (300 dpi,
light background) or copies it to the clipboard. **Export CSV** writes the
plotted values. The CSV starts with ``#`` lines that record the project,
model, calibration set, recipe, components and cross-validation it came from.

**File > Export Plots...** saves every plot of the current workspace (PNG and
CSV, named by plot) into one folder. That includes Model pages you have not
opened. A maximized plot stays maximized the next time the viewer opens, and
**View > Reset Layout** restores the default layout.

Target values are typed in the **ROI table**, opened from **Target
values...** in Model, from **ROI table** in
Explore, or from the View menu. It opens by itself with the Project ROIs
tab, in the right-hand column of the window, and is not shown on the Dataset tab; opened there, it takes the Sensor Frame's place as a tab. Closing
or opening it on either tab is kept for that tab.

- One row per ROI (dataset, name, pixel count, **Set**, sample) and one
  column per target. **Set** is Calibration or Prediction; a prediction ROI
  is a tablet to predict and is never used to fit a model. The dataset
  column shows the data file's name without its folder (the ROI name tells
  the sample); its tooltip shows the folder. Rows stay grouped by data file,
  also when several files share one name.
- **Sample** names the physical sample an ROI was taken from. Give the same
  label to ROIs from one sample (for example three regions of one tablet),
  so that a fit can hold them out together. Labels are text; surrounding
  spaces are removed and an empty cell means the ROI is its own sample.
  **Ctrl+V** pastes a column of labels.
- It opens on the ROIs of the page and the target selected in Model: the
  shown dataset on the Dataset tab, the datasets chosen on the Project
  ROIs tab, all datasets in Model and Predict. The filters choose a dataset, or only ROIs that still lack a value.
- Type a value and press **Enter**: it is stored at once and the next row
  is edited. A decimal comma is accepted.
- Clear a cell to remove a value; the ROI is then left out of models for
  that target.
- **Ctrl+V** pastes a column of values downwards from the current cell. All
  values are checked first: one that is not a number, or more values than
  rows, writes nothing.
- Selecting a row selects the ROI in the browser and on the image.
- The table works from the project file alone; the datasets are not
  needed.

**Fit Model** opens one dialog with everything the fit needs: the target,
the signal unit, the preprocessing (it starts from the Explore preview, but
what the dialog shows is what the model is trained with), PLS or PCR, the
number of components, the validation and which ROIs are held out together
(**Hold out together**: each ROI, ROIs with the same **Sample** label, or
ROIs with the same target value). **Bands** sets **Keep** and **Exclude**
ranges (``1100-1350, 1500-2300``, applied after every other step), with a
**Water bands** menu, **Select on plot...** to drag the ranges on the mean
spectrum of the target's ROIs, and **Propose...** to run the band proposal
with nested validation and take its ranges (:doc:`band_selection`); the
Explore preview then shows the left-out bands as missing. **Components** takes a number, or **Auto**
to choose it from the component curve by a rule (one SE by default; also
lowest RMSECV (per ROI), Q² limit, F-test and Q² with look-ahead) with **drop noise**
(the noise guard, on by default; see :doc:`fit_specification`). The dialog
warns when the validation would be optimistic, for example when the ROIs of a
scan share one value but are held out in different folds. The
**Fitting** group sets how the
model is fitted: the same number of pixels from every ROI (at most the
smallest ROI, with a seed so the draw repeats), the scaling of X and Y, and
leaving out outlying pixels by DModX (level, one or two rounds, at most a
share of every ROI, unweighted or weighted as in SIMCA). The Project
browser's inspector shows these settings for a stored model, with its
algorithm (kernel PLS or NIPALS, PCA from XᵀX or by full SVD; **Fit Again**
keeps it), the excluded pixels per ROI and the flagged ROIs. Below, the dialog counts per dataset how many ROIs have a value for the target and says why a fit cannot run
yet; **OK** is enabled only when it can. The model uses all ROIs with a
value, from all datasets. Fitting again with the same target, ROIs and
preprocessing reuses the stored calibration set; the calibration set and preprocessing
are stored only together with a fitted model. Cross-validation keeps the
pixels of one ROI together (and, if chosen, the ROIs of one sample), and
reported metrics are per ROI: held-out pixel predictions are averaged per ROI
before RMSECV (per ROI), R², bias, RPD and RPIQ are computed. Until a model can be
fitted, the readiness message says what is missing.

Predict
-------

Predict works on project data only: the tablets to predict are ROIs marked
**Prediction only**, and a scan to predict is a project dataset.

**Mark the tablets.** Add the tray with **File > Add Dataset...** and
choose its reference correction there, as for a calibration tray (the references of this
session, or the detector levels). Draw the tablets with **Add ROI** or **Add
ROI grid** and tick **Prediction only**; the choice is remembered for the
dataset. The **Set** column of the ROI table (Calibration or Prediction, chosen from a
drop-down) switches an ROI later.
Prediction ROIs never enter a model fit, even with a value: a value they
have only gives the residual and the RMSEP (per ROI) of the prediction, an independent
test. They are dashed in the mosaic and in the ROI spectra.

**Predict ROIs...** predicts the prediction ROIs of one dataset or of all
of them with one or more models. It reads only the project file: each ROI's
stored pixels are converted with its own reference correction, exactly as the fit
converts calibration ROIs, and predicted with the model's recipe. The **Prediction
per ROI** plot shows each ROI's mean prediction (the reported value) with
the 5–95 % range of its pixels, one color per model, and the target value
(×) when the ROI has one. Each mean is written next to its dot. A ring marks
unusual spectra: a tablet with a larger share of pixels above the DModX limit
(0.99) than 90 % of the model's own calibration ROIs. The **ROI table** gets
columns per model while there are results: mean, SD, % excluded, % above
DCrit and residual (read-only, not stored); click a header to sort the
tablets, for example by predicted value. Its CSV is the per-ROI table: ROI, dataset, model, pixels, excluded pixels, mean, median,
SD, 5 % and 95 %, mean DModXPS and T²PS, shares above the limits, target value and
residual, with RMSEP (per ROI) and bias per model.

**Exclude pixels above the DModXPS limit** (with its level) leaves glare,
exposed substrate and direct reflections out of every ROI statistic; the
statistics are recalculated at once, without predicting again. It starts as
the model was fitted: on at its level for a model fitted with DModX
exclusion, off otherwise. Next to the level, the Controls dock shows the
DModX limit (DCrit) it gives for each model.

**Predict Map...** predicts every pixel of one project dataset with one
model and shows the results of its prediction ROIs, computed from their
stored pixels (so an ROI has the same numbers either way). The dialog names
the models and datasets, shows the chain the prediction applies (the dataset's reference correction, the model's signal unit and recipe, the model, and what
the model was trained with), and refuses a file that cannot give the model's
signal, with the reason: another instrument format, number of bands,
reference correction mode or signal unit. **Add dataset...** in the dialog opens the
Add dataset dialog, which says how the model's calibration scans were
corrected but never reuses their references. Wavelength values never block
a prediction; when they differ from the model's, the **Wavelengths** row
warns with the largest difference. Models with MSC or EMSC apply the
preprocessing state stored at calibration.

The map's color scale is the target's range in the model's calibration values;
predictions below and above it have their own colors (cyan, pink), so the
substrate and other material the model never saw do not flatten the scale.
**Color scale** in the Controls dock (or **Color limits...** in the map's ⋯
menu) sets your own limits, for one scale over a series of trays; the color
bar and the CSV record the scale. The prediction ROIs are outlined with
their mean; with the pixel exclusion on, excluded pixels are grayed. **Map** in
the Controls dock turns the outlines and the graying off. Hovering the map
shows the pixel in the status line at the bottom of the window: its frame and
row, the prediction (and whether it is outside the color scale) with its
95 % interval, DModXPS against DCrit (and whether the pixel is excluded), T²PS
and the prediction ROI under it. Hovering a tablet in the per-ROI plot shows its values there. The map header has the **Layer** selector (the
prediction, **DModXPS** and **T²PS**, with the pixels above the 0.99 limits
marked, and **Prediction interval**, the ± half-width of each pixel's 95 %
prediction interval with the pixels outside the model, which have none,
marked; see :ref:`prediction-intervals`), the output selector (models with several targets), **Pan**,
**Zoom** and **Reset View**. Large maps are shown as a preview that keeps
every n-th frame and row; the axes, hover readout and CSV give the true
frame and row. A value a model does not have is shown as "not available".

The **Prediction Distribution** histogram shows the pixels behind the per-ROI
numbers: the kept pixels of the prediction ROIs, one outline per model, and
it follows the pixel removal. Without ROI results, or with **Histogram: Whole
map** in the Controls dock, it shows the pixels of the shown map. Map and per-ROI results always describe the same dataset: a map
of a file without prediction ROIs drops the earlier per-ROI results and says
so; the Predict map file list shows each file's prediction ROIs.

The summary on the right starts with one line naming what is shown (the dataset and the model), then groups the **Prediction ROIs** (RMSEP (per ROI) and bias per
model, refused ROIs), the **Chain** (signal unit and recipe, target, bands,
input), the **Scan** (reference correction, map shape and range, clipped reflectance
values) and the **Applicability** (share of pixels above the limits, the
median 95 % interval and the share without one, wavelengths). The per-ROI values are in the ROI table.

Predictions are not stored: the project keeps the datasets, their ROIs and
the models, and a prediction is a result to inspect and export. The last
three map predictions of the session are kept in memory, and each prediction
writes one entry to the log. Prediction streams through a scan in bounded
memory and reports its phase.

Projects
--------

A project collects ROIs from any number of datasets. **File > Add Dataset...** (or **Open Data...** while a project is open) adds a file to the
project in the same window and asks how it is corrected:

- **Dark and white references**, each from a file or from frames of the dataset itself, such as a white tile or shutter-closed frames at the start of
  the tray;
- **Same references as another dataset**, for one reference set per session;
- **Detector levels**, black and white, editable: the levels stored in the
  file, else the instrument defaults 128 and 3500. This is the preselected
  choice;
- **Raw counts**, without references or levels (only raw counts (DN) can be
  shown and modeled).

The dialog asks for no signal unit: DN, reflectance and absorbance are
chosen in Explore and in **Fit Model**. Only a file that already holds
corrected values (floating point) asks what it holds, reflectance or
absorbance.

The dialog plots the mean spectrum of the file in counts, from a few middle
frames, together with what the chosen reference correction uses: the mean dark and
white spectra with their share of saturated values, the references of the
other dataset, or the black and white detector levels. You can see where
the data lies between dark and white before you accept. Files with
floating-point values are taken as already corrected.

The dataset list in the Spatial Image header switches between the
project's files, as does a double-click on a file in the browser. Each file
is shown with its own reference correction and its own ROIs. A project opened
on another computer or platform finds its files by itself when they lie in
the same place relative to the project file, or on the same drive (``Z:\``
on Windows is ``/mnt/z`` in WSL); the tooltip of the dataset list shows the
path it is read from. This changes nothing in the project. A file that is not found is listed as
"(not found)"; its ROIs, values and models stay usable. Choosing it offers
**Locate file...**: the file you pick is accepted only if it is the same
dataset, with the same shape and wavelengths and, for every ROI, the same
pixels as stored in the project. Only the project's path to the file
changes; a different file is refused and nothing changes. When other files
are missing too and lie in the same new place, the viewer offers to locate
them as well; each is checked the same way.

A project with ROIs opens on the **Project ROIs** tab without loading any
dataset. The **Dataset** tab then loads the file you showed last for this
project (remembered in your user settings, not in the project). Several
windows on one project stay in step: a value typed in one appears in the
others.

Drawing ROIs keeps the reference correction of the file. If you load other references
first, the viewer asks before new ROIs use them, and the ROIs drawn before
keep their own.

Datasets are shown by their file name. When several share a file name
(scans saved as ``stream_hsi.h5`` in different folders), each is shown with
the folder that tells them apart, e.g. ``stream_hsi.h5 (API 10% - nozzle…)``,
in the dataset list, the ROI table, the Project ROIs tab, the fit dialog
and the browser; the full path is the tooltip.

Every object in a project has a stable id, its identity. Names of ROIs,
targets and models are labels: you can rename them and they may repeat. The
viewer refers to everything by id, so repeated names never mix up ROIs, values
or models; where names may repeat it also shows the dataset or the id.

There is no save step: every ROI, value, calibration set and model is written when
it is made (predictions are not stored). ROIs and targets are identified by stable ids; names
are labels. Projects open metadata-first, so ROI spectra and model or
prediction arrays are read only when a plot or job needs them. To script
what the viewer does, use :class:`enginir.Project` (:doc:`workflows`).

Walkthrough: one model from several trays
-----------------------------------------

A typical reference correction session scans several trays of samples, with dark and
white references recorded once. From an empty project to a model:

1. **Create the project.** On the welcome screen choose **Create Project...** and
   pick where the project file goes. The window opens on Explore with no dataset.
2. **Add the first tray.** **File > Add Dataset...** opens the file and
   asks for its reference correction. **Detector levels** (the instrument defaults) is
   preselected. If the tray has a white tile and shutter-closed frames,
   choose **Dark and white references** and pick them as frame ranges of the
   file itself, or choose the reference files. Check the mean spectra and the share
   of saturated values, then **OK**: the dataset and its reference correction are
   stored together.
3. **Draw the samples.** Set your tray holder once in **Options > ROI Grid
   Defaults** (rows, columns, ROI size, spacing). **Add ROI grid** then
   shows the grid on the tray; correct the origin if the tray sits elsewhere,
   click empty positions, and **OK**. Every cell becomes an ROI named after the file and its position,
   e.g. ``tray01-B04``, with its raw pixels and reference pixels.
4. **Add the next trays.** Repeat **Add Dataset...** and choose **Same
   references as** the first tray for the rest of the session, then draw their
   grids. The dataset list in the image header switches between the trays;
   each shows only its own ROIs.
5. **Enter the target values.** **Target...** in Model defines the target
   (for example ``API`` in ``%``). **Target Values...** opens the ROI table:
   type each value and press **Enter**, or paste a column from a spreadsheet
   with **Ctrl+V**. Cells left empty are samples without a value.
6. **Look at what you collected.** The **Project ROIs** tab shows every ROI
   of every tray, with its value, and the mean spectra colored by value in
   the signal unit and preprocessing chosen in Explore. An outlying spectrum
   or a misplaced cell shows up here; double-click its tile to see it on its
   tray.
7. **Fit.** **Fit Model** shows, per tray, how many ROIs have a value. Choose
   the signal unit, preprocessing, PLS or PCR, the number of components and
   the validation, and **OK**. The fit uses every ROI with a value from all
   trays; cross-validation keeps the pixels of one sample together.
8. **Come back later.** Opening the project shows the Project ROIs tab at
   once, without reading any tray. Values, models and predictions work even if
   the tray files are gone; a moved tray file is found again with **Locate
   file...**, which accepts it only if its pixels match.

The same session from Python is in :doc:`workflows`.

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
