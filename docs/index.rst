engiNIR
=======

engiNIR turns near-infrared spectra and hyperspectral images into validated
calibration models and property predictions, with every step recorded. It is
built for pharmaceutical process analytics. This is the documentation of
engiNIR |release|.

It reads EVK EIFF, ENVI, HDF5, CSV/Excel, SPC, SPA and JCAMP-DX data;
calibrates raw counts with dark and white references; preprocesses with
Savitzky-Golay, SNV, MSC, EMSC and detrending; fits PLS and PCR models
cross-validated by region of interest; and predicts properties of new spectra
and whole images. The desktop viewer ``enginir-viewer`` runs the same workflow
interactively (:doc:`user_guide/viewer`).

There are two ways to use the library.

**Project mode** records every step in a project file, so each result can be
reproduced and traced to its data (:doc:`user_guide/workflows`):

.. code-block:: python

   from enginir import PreprocessingRecipe, Project

   project = Project.create("tablets.h5")
   scan = project.add_dataset("scan.eiff", dark="dark.eiff", white="white.eiff")
   project.add_target("API content", unit="% w/w")
   scan.add_roi("tablet 1", frames=(0, 4), rows=(0, 9), values={"API content": 1.0})
   # ... more ROIs
   model = project.fit("API content", recipe=PreprocessingRecipe(snv=True), n_components=2)
   model.predict("new_scan.eiff").map

**Function mode** uses the building blocks directly on arrays, as
scikit-learn estimators:

.. code-block:: python

   from sklearn.pipeline import make_pipeline

   from enginir.io import load_spectra
   from enginir.models import PLSRegression
   from enginir.preprocessing import SNV, SavitzkyGolay
   from enginir.validation import cross_validate, metrics_report

   X, wavelengths, meta = load_spectra("spectra.csv")
   y = ...  # measured reference values, one per spectrum

   model = make_pipeline(SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
                         SNV(), PLSRegression(n_components=6))
   cv = cross_validate(model, X, y)
   model.fit(X, y)
   print(metrics_report(y, model.predict(X), y_pred_cv=cv["y_pred"]))

.. toctree::
   :maxdepth: 1
   :caption: Start Here

   user_guide/installation
   user_guide/workflows

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/io
   user_guide/preprocessing
   user_guide/models
   user_guide/validation
   user_guide/visualization
   user_guide/viewer
   user_guide/logging

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/workflow
   api/io
   api/preprocessing
   api/models
   api/modeling
   api/prediction
   api/project
   api/validation
   api/visualization
   api/viewer
   api/logging
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Format References

   eiff_format
   spc_format
   hdf5_format
   hsi_viewer

.. toctree::
   :maxdepth: 1
   :caption: Architecture

   viewer_architecture
   workspace_domain_model
   contract_coverage

Citation
--------

If you use engiNIR in your work, please cite:

   Johannes Poms. (2026). TUstudents/engiNIR: A universal engineering tool for
   near-infrared spectra and hyperspectral imaging. Zenodo.
   https://doi.org/10.5281/zenodo.23235907
