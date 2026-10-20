Loading and Saving Spectra
==========================

The :mod:`enginir.io` module provides readers and writers for common NIR
spectral data formats.  The high-level entry point is :func:`~enginir.io.load_spectra`,
which auto-detects the format from the file extension.

Return type
-----------

All loaders return a :class:`~enginir.io.SpectralData` named tuple::

   X, wavelengths, meta = load_spectra("data.csv")   # tuple unpacking
   data = load_spectra("data.csv")
   data.X            # spectral matrix
   data.wavelengths  # nm axis
   data.meta         # format-specific metadata dict

For point spectra ``X`` has shape ``(n_samples, n_wavelengths)``.
For image formats ``X`` has shape ``(rows, cols, n_wavelengths)`` (ENVI) or
``(n_frames, height, n_wavelengths)`` (EIFF) — see :ref:`image-cubes`.

Auto-detection
--------------

.. code-block:: python

   from enginir.io import load_spectra

   X, wl, meta = load_spectra("sample.csv")     # CSV / Excel
   X, wl, meta = load_spectra("sample.spc")     # Galactic SPC
   X, wl, meta = load_spectra("sample.eiff")    # EVK EIFF hyperspectral
   X, wl, meta = load_spectra("sample.hdr")     # ENVI image cube

Supported extensions:

=============================  =========================
Extension                      Format
=============================  =========================
``.csv``                       CSV (wavelengths as column headers)
``.xlsx``                      Excel
``.spc``                       Galactic / Thermo SPC
``.spa``                       Thermo SPA
``.dx``, ``.jdx``              JCAMP-DX
``.hdr``, ``.img``, ``.dat``,  ENVI image cube
``.raw``
``.h5``, ``.hdf5``             HDF5
``.eiff``                      EVK HELIOS hyperspectral
=============================  =========================

CSV / Excel
-----------

The expected layout has wavelength values as column headers::

   sample_id | 1000.0 | 1002.0 | ... | 2500.0
   sample_1  |  0.42  |  0.41  | ... |  0.38

.. code-block:: python

   from enginir.io import load_csv, save_csv

   X, wl, meta = load_csv("spectra.csv")
   print(meta["sample_names"])   # list of row identifiers

   # round-trip
   save_csv("out.csv", X, wl, sample_names=meta["sample_names"])

For Excel, pass ``sheet_name`` to select the worksheet::

   X, wl, meta = load_csv("spectra.xlsx", sheet_name="Cal set")

.. _image-cubes:

Hyperspectral image cubes
--------------------------

Image loaders (ENVI, EIFF, HDF5) return a 3-D array with the spectral axis
last: ``(n_frames, height, n_bands)`` for the push-broom formats and
``(lines, samples, bands)`` for ENVI. Use :func:`~enginir.utils.flatten_cube` to
get one spectrum per pixel for preprocessing and models, and
:func:`~enginir.utils.unflatten_cube` to restore the spatial layout:

.. code-block:: python

   from enginir.io import load_spectra
   from enginir.preprocessing import SNV
   from enginir.utils import flatten_cube, unflatten_cube

   cube, wl, meta = load_spectra("scan.eiff")    # (n_frames, height, n_bands), uint16
   X_2d, spatial_shape = flatten_cube(cube)      # (n_frames * height, n_bands), a view
   X_snv = SNV().fit_transform(X_2d)             # float32 working copy
   cube_snv = unflatten_cube(X_snv, spatial_shape)

Imaging readers keep the stored dtype (EIFF gives raw ``uint16`` counts) and
read big-endian ENVI files into native byte order; ENVI cubes are read with a
single copy of the data. :func:`~enginir.io.save_envi` writes
band-interleaved-by-pixel (BIP) files, which matches the in-memory layout and
needs no copy. Point-spectrum readers (CSV, SPC, SPA, JCAMP-DX) return
``float64``.

Raw HDF5 Spectral Cubes
-----------------------

``load_spectra()`` opens ``.h5`` / ``.hdf5`` files as raw spectral data unless
the file is an engiNIR project. The default raw reader accepts either a
root ``/spectra`` dataset or the KESTRELEYE raw-HDF5 profile with a root
``/data`` dataset. KESTRELEYE files record
``hdf5_schema="kestreleye_raw_hdf5"`` and support Blosc-compressed datasets via
the ``hdf5plugin`` dependency. If no ``/wavelengths`` dataset exists, band
indices are used as wavelengths and ``meta["x_units"]`` is ``"band_index"``. KESTRELEYE cube shape is interpreted as
``(track, cross_track, bands)``. The track length varies by acquisition, and
cross-track pixels or band count may vary with camera setup; enginir reads
those dimensions from the file. Other vendor HDF5 schemas are not covered by
this profile and should be added explicitly when needed.

HDF5 Project Files
------------------

A project file (HDF5, schema ``viewer-v3``) records a whole analysis:

- project metadata and schema version
- dataset references
- dataset-owned calibration records
- training ROIs
- target definitions and assignments
- preprocessing recipes
- signal-domain contracts
- sample sets
- model runs
- prediction runs

See :doc:`/hdf5_format` for the schema overview and object layout.

.. code-block:: python

   import numpy as np
   from enginir.io import (
       create_project,
       save_dataset_ref,
       save_calibration_spec,
       save_training_roi,
       save_target_definition,
       save_target_assignment,
       load_project_store,
   )

   wavelengths = np.linspace(900.0, 1700.0, 50, dtype=np.float32)
   spectra = np.zeros((12, 50), dtype=np.uint16)

   create_project("project.h5")

   save_dataset_ref(
       "project.h5",
       "dataset_1",
       {
           "source_path": "/data/train_a.eiff",
           "source_format": "eiff",
           "source_modality": "evk_raw",
           "cube_shape": (20, 15, 50),
           "n_bands": 50,
           "wavelengths": wavelengths,
           "created_at": "2026-04-02T10:00:00+00:00",
           "loader_metadata": {"instrument": "EVK AlphaVision"},
       },
   )
   save_calibration_spec(
       "project.h5",
       "dataset_1",
       "cal_1",
       {
           "mode": "dark_white_reference",
           "output_signal_unit": "Absorbance",
           "black_level": 128.0,
           "white_level": 3500.0,
           "created_at": "2026-04-02T10:01:00+00:00",
       },
   )
   save_training_roi(
       "project.h5",
       "roi_1",
       {
           "dataset_id": "dataset_1",
           "calibration_id": "cal_1",
           "name": "ROI 1",
           "source_file": "/data/train_a.eiff",
           "n_pixels": 12,
           "created_at": "2026-04-02T10:02:00+00:00",
           "spatial_bounds": (0, 2, 0, 3),
           "wavelengths": wavelengths,
           "provenance": {
               "dataset_id": "dataset_1",
               "calibration_id": "cal_1",
               "signal_unit": "Absorbance",
               "preview_recipe_summary": "SNV preview only",
               "created_at": "2026-04-02T10:02:00+00:00",
           },
       },
       spectra,
   )
   save_target_definition(
       "project.h5",
       "target_moisture",
       {
           "name": "moisture",
           "unit": "% w/w",
           "value_type": "float",
           "missing_value_policy": "allowed",
           "created_at": "2026-04-02T10:03:00+00:00",
       },
   )
   save_target_assignment(
       "project.h5",
       "target_moisture",
       "roi_1",
       {"value": 12.3, "assigned_at": "2026-04-02T10:04:00+00:00"},
   )

Loading everything back:

.. code-block:: python

   from enginir.io import load_project_store, load_prediction_run_artifact_refs

   store = load_project_store("project.h5")

   # Prediction-run outputs stay disk-backed until explicitly requested.
   pred_refs = load_prediction_run_artifact_refs("project.h5", "prediction_run_1")

``load_project_store()`` reads metadata only by default: ROI spectra and
model or prediction arrays stay on disk unless you pass
``include_roi_spectra=True`` or ``include_artifact_arrays=True``.

``load_project_store()`` is the raw reader: it checks the file structure and
decodes values. To work with a project, open it with :class:`enginir.Project`,
which validates every object and every reference and gives you datasets,
ROIs, targets, models and predictions by name:

.. code-block:: python

   from enginir import Project

   project = Project.open("project.h5")
   model = project.model("PLS 2 LV, SNV")
   model.metrics["rmsecv"]

   prediction = model.predict("prediction_scan.eiff",
                              dark="dark.eiff", white="white.eiff")
   prediction.map            # (frames, height), read from the project file

``predict`` registers the scan with a calibration in the model's signal unit
(or reuses the registered one) and checks that the scan can produce what the
model was trained on (instrument format, number of bands, calibration and
signal unit) before anything is read or written. Wavelength values are
metadata and are not compared. The stored prediction run records the dataset,
the calibration actually applied, the model, its signal domain and its
preprocessing recipe. Model runs with ``MSC`` or ``EMSC`` store their fitted
preprocessing state; prediction restores it without fitting on the new data.

See :doc:`workflows` for the complete workflow from a raw scan to a
prediction map.

The typed project-domain objects live in :mod:`enginir.project`. Hydration
turns the raw store into validated objects and checks all references between
them:

.. code-block:: python

   from enginir.io import load_project_store
   from enginir.project import hydrate_project_store

   workspace = hydrate_project_store(load_project_store("project.h5"))
   roi = workspace.rois["roi_1"]
   calibration = workspace.calibrations[roi.calibration_id]
   spectra = roi.load_spectra("project.h5")   # (n_pixels, n_bands), read on demand

Invalid stored values raise a :class:`ValueError` that names the object's
location in the file (for example ``/rois/roi_1``).

The services below :class:`~enginir.Project` live in :mod:`enginir.modeling`
and :mod:`enginir.prediction`; see :doc:`/api/modeling` and
:doc:`/api/prediction`.
