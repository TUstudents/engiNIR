Workflows
=========

engiNIR can be used in two ways. Both use the same readers, preprocessing,
models and validation code; they differ in who keeps track of the analysis.

**Project mode** (:class:`enginir.Project`) records every step in a project
file: datasets with their calibrations, ROIs with reference values, the
preprocessing recipe, the sample set, the model and its predictions. Results
are reproducible and traceable without any bookkeeping code. Use it unless you
know exactly which steps you need; it is also what the viewer uses.

**Function mode** uses the building blocks directly on arrays, as
scikit-learn estimators, for your own analysis. You decide every step and are
responsible for keeping the analysis reproducible.

Project mode
------------

One call per step:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Step
     - Call
   * - Create or open a project
     - ``Project.create(path)``, ``Project.open(path)``
   * - Register a scan with its calibration
     - ``project.add_dataset(path, dark=..., white=..., unit="Absorbance")``
   * - Define the property to model
     - ``project.add_target("API content", unit="% w/w")``
   * - Record ROIs with reference values
     - ``dataset.add_roi(name, frames=(first, last), rows=(first, last), values={...})``
   * - Fit and cross-validate
     - ``project.fit(target, recipe=PreprocessingRecipe(...), model="PLS", n_components=...)``
   * - Review the model
     - ``model.metrics``, ``model.diagnostics``
   * - Predict a new scan
     - ``model.predict(path)``, then ``prediction.map`` or ``prediction.y_pred``

References need not be separate files. ``dark_frames=(first, last)`` and
``white_frames=(first, last)`` average frames of the same file (a white tile
or shutter-closed frames scanned in the tray), and ``references_from=path``
reuses the references of another registered scan, so one session needs one
reference set:

.. code-block:: python

   tray1 = project.add_dataset("tray1.eiff", dark_frames=(0, 9), white_frames=(990, 999))
   tray2 = project.add_dataset("tray2.eiff", references_from="tray1.eiff")

``project.set_value(roi, target, value)`` sets or replaces a reference value;
``value=None`` removes it.

A tray of samples becomes one ROI per cell with :func:`~enginir._workflow.roi_grid`
and ``dataset.add_rois``, all stored or none:

.. code-block:: python

   from enginir._workflow import roi_grid

   # 4 x 8 wells: ROIs of 40 x 36 pixels, 10 pixels apart, the first at frame 15, row 20
   cells = roi_grid(n_rows=4, n_columns=8, size=(40, 36), origin=(15, 20),
                    spacing=(10, 10), skip=[(3, 7)],
                    name_pattern="{file}-{row}{column:02d}", file="tray1")
   tray1.add_rois(cells)

A scan that was moved or renamed is pointed at its new place with
``dataset.locate(new_path)``. The file is accepted only if it is the same
scan (format, shape, wavelengths, and the stored pixels of every ROI);
otherwise nothing changes:

.. code-block:: python

   project.dataset("tray1.eiff").locate("archive/2026-11/tray1.eiff")

Calibrations, sample sets, signal domains and provenance are created for you,
reused when identical, and validated before anything is written. Cross-
validation keeps the pixels of one ROI together; metrics are per ROI.
Prediction is refused before it starts when a scan cannot produce the model's
signal (instrument format, number of bands, calibration, signal unit).

The script below runs the whole workflow on a synthetic scan of eight
tablets:

.. literalinclude:: ../examples/project_workflow.py
   :language: python

It prints the cross-validated model and the mean prediction of each tablet,
which follows the true contents 1 to 9::

   Model('PLS 2 LV, SNV', rmsecv=0.612, r2=0.945, bias=0.0838)
   [1.7 2.7 3.7 4.4 5.4 6.3 7.4 8.4]

``project.fit`` validates with 7 round-robin folds over the ROIs (``cv=5``
for five, ``cv="logo"`` to leave one ROI out), centres the spectra and
scales the reference values to unit variance. ``fit_spec`` controls the
rest, and is stored with the model so that ``model.refit()`` repeats it:
holding out the ROIs of one sample together, drawing the same number of
pixels from every ROI, and leaving out outlying pixels by DModX. See
:doc:`fit_specification`, which also compares the numbers with SIMCA 18:

.. code-block:: python

   model = project.fit("API content", recipe=PreprocessingRecipe(snv=True),
                       n_components=2, fit_spec={"pixels": {"per_roi": 40, "seed": 1}})
   model.ref.fit_specification      # FitSpec(fold_rule='round_robin', ...)

Reopen the project later with ``Project.open("tablets.h5")``; datasets are
available by path and ROIs, targets and models by name, e.g.
``project.model("PLS 2 LV, SNV").diagnostics``.

The layers below :class:`~enginir.Project` (:mod:`enginir.project`,
:mod:`enginir.modeling`, :mod:`enginir.prediction`) remain available for
advanced use. The project file layout is in :doc:`/hdf5_format`, the objects
and their rules in :doc:`/workspace_domain_model`.

Names and ids
-------------

Names are labels: they can change, and a project made in the viewer may
repeat them (ROIs from scans that share a file name). Addressing by name is a
convenience; a name that several objects share is an error listing their ids,
never a guess. Every object has a stable id, and scripts that must give the
same result later use the objects or their ids:

.. code-block:: python

   roi = project.roi(roi_id="6f2c…")              # or project.roi("tray03-B04")
   api = project.target(target_id="91ab…")
   project.set_value(roi, api, 4.2)               # objects instead of names
   model = project.fit(api, n_components=3)
   same = project.model(model_run_id=model.ref.model_run_id)

A stored model can be fitted again on exactly its own data: ``model.refit()``
uses its sample set (the ROI ids), preprocessing, signal, model settings and
validation, not the ROIs that have values today, and stores the result as a
new model. Unchanged data give the same metrics; changed reference values and
differing metrics are reported in the log:

.. code-block:: python

   again = project.model(model_run_id=model.ref.model_run_id).refit()
   again.metrics["rmsecv"] == model.metrics["rmsecv"]   # True for unchanged data

``project.rois``, ``project.targets`` and ``project.models`` are keyed by
name and refuse repeated names; ``project.workspace`` holds every collection
keyed by id. Functions that create objects by name (``add_roi``,
``add_rois``, ``add_target``, ``fit(name=...)``) refuse a name that exists, so
a script finds what it made.

Function mode
-------------

The readers in :mod:`enginir.io` return arrays; the transformers in
:mod:`enginir.preprocessing` and the models in :mod:`enginir.models` follow
the scikit-learn API; :mod:`enginir.validation` has cross-validation, figures
of merit and outlier statistics, and :mod:`enginir.visualization` the plots.

.. code-block:: python

   import numpy as np
   from sklearn.pipeline import make_pipeline

   from enginir.io import load_eiff
   from enginir.models import PLSRegression
   from enginir.preprocessing import EMSC, SNV
   from enginir.validation import cross_validate, metrics_report
   from enginir.visualization import plot_prediction

   cube = load_eiff("scan.eiff").X                       # (frames, height, bands), raw counts
   absorbance = -np.log10(np.clip((cube - 128.0) / (3500.0 - 128.0), 1e-6, 1.0))
   X = absorbance[:40].reshape(-1, cube.shape[-1])       # the pixels of 8 samples, 5 frames each
   y = np.repeat(reference_values, 5 * cube.shape[1])   # one lab value per sample
   samples = np.repeat(np.arange(8), 5 * cube.shape[1])

   model = make_pipeline(EMSC(order=0), SNV(), PLSRegression(n_components=4))
   cv = cross_validate(model, X, y, cv="logo", groups=samples)   # leave one sample out
   model.fit(X, y)
   print(metrics_report(y, model.predict(X), y_pred_cv=cv["y_pred"]))
   plot_prediction(y, cv["y_pred"], units="% w/w")

Fitted steps such as MSC and EMSC belong inside the pipeline, so that
cross-validation fits them on the training folds only. Group the
cross-validation by sample, as above: pixels of one sample are not
independent, and splitting them across folds overstates the model.
