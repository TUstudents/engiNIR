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

   Model('PLS 2 LV, SNV', rmsecv=0.745, r2=0.919, bias=0.0743)
   [1.7 2.6 3.7 4.4 5.4 6.3 7.4 8.4]

Reopen the project later with ``Project.open("tablets.h5")``; datasets are
available by path and ROIs, targets and models by name, e.g.
``project.model("PLS 2 LV, SNV").diagnostics``.

The layers below :class:`~enginir.Project` (:mod:`enginir.project`,
:mod:`enginir.modeling`, :mod:`enginir.prediction`) remain available for
advanced use. The project file layout is in :doc:`/hdf5_format`, the objects
and their rules in :doc:`/workspace_domain_model`.

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
