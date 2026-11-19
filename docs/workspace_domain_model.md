# Workspace domain model

An overview of the objects a project holds: for each, its term (as the
viewer and the guides say it; `contracts/terminology_v3.md`), its Python class
and where the project file stores it. The detail is in the contracts (the
column on the right) and in `hdf5_format.md`; this page links them. The file
format is `enginir-1.7`.

## Objects

| Term | Python class | Project file | Contract |
|---|---|---|---|
| Project | `enginir.Project` (the workflow facade) | `/project_meta/` (`project_id`, `FileSchemaVersion`) | project-schema |
| Dataset | `DatasetRef`; `enginir.Dataset` wraps it | `/datasets/{dataset_id}/` | project-dataset-attachment |
| Reference correction | `ReferenceCorrection` | `/datasets/{dataset_id}/reference_corrections/{id}/` | dataset-reference-correction |
| ROI (calibration ROI, prediction ROI: its `job`) | `TrainingROI` | `/rois/{roi_id}/` with its raw pixels `spectra` | roi-target |
| Target | `TargetDefinition` | `/targets/{target_id}/` | roi-target |
| Target value | `TargetAssignment` | `/target_assignments/{target_id}/{roi_id}/` | roi-target |
| Preprocessing recipe | `PreprocessingRecipe` | `/recipes/{recipe_id}/` (id from the content) | sample-set-model-run |
| Input contract | `SignalDomainContract` | `/signal_domains/{signal_domain_id}/` | sample-set-model-run |
| Calibration set | `SampleSet` | `/sample_sets/{sample_set_id}/` | sample-set-model-run |
| Model | `ModelRun`; `enginir.Model` wraps it | `/model_runs/{model_run_id}/` with `artifact_refs/` | sample-set-model-run, model-workspace-diagnostics |
| Prediction (stored) | `PredictionRun`; `enginir.StoredPrediction` | `/predictions/{prediction_run_id}/` (only when a script stores one) | prediction-runtime |

A prediction in the viewer, of prediction ROIs or of a whole dataset as a
prediction map, lives in the session and is not stored.

## Relationships

- A dataset owns its reference corrections; one of them is active. A
  reference correction never moves to another dataset.
- An ROI belongs to one dataset and keeps the reference correction that was
  active when it was drawn, its raw pixels as recorded and the dataset's
  wavelength axis. Its `job` makes it a calibration ROI or a prediction ROI;
  its `sample` label groups ROIs of one physical sample.
- A target value pairs an ROI with a target.
- A model is fitted on a calibration set (its ROIs and target, built with a
  preprocessing recipe) and records its input contract: the signal, bands
  and wavelength axis a scan must provide. Its fit specification
  (`fit_spec`), the software it was fitted with and its limits (`dcrit_095`,
  `t2_crit_095`) are stored with it.
- Every object has a stable id; names are mutable labels. A stored object is
  never overwritten with other content; the exceptions (a dataset's path, its
  label, its wavelength table, a target value) are listed in the
  project-schema contract (§5).

## Lifecycle

From an empty project to a prediction; the viewer calls the same rules.

| Step | `enginir.Project` | Objects stored |
|---|---|---|
| 1. Create | `Project.create(path)` | project metadata |
| 2. Add a dataset | `project.add_dataset(path, dark=, white=, unit=)`; references also from frames of the file (`dark_frames=`, `white_frames=`) or another dataset (`references_from=`); wavelengths from a table (`wavelength_table=`) | dataset, reference correction (reused when identical; becomes active) |
| 3. Define a target | `project.add_target(name, unit=)` | target |
| 4. Draw ROIs | `dataset.add_roi(name, frames=, rows=, values=)` or `dataset.add_rois(roi_grid(...))` | ROI with its pixels, a target value per value |
| 5. Fit | `project.fit(target, recipe=, model=, n_components=)` | recipe (by content), calibration set (reused), input contract (reused), model |
| 6. Predict | `model.predict_rois()` (prediction ROIs, from the project file), `model.predict(path)` (a dataset) | nothing (a prediction lives in memory); the low-level writers of `enginir.io.project_store` store one, read back as `project.predictions` |

Each step validates first and then writes, in dependency order: a failed step
stores nothing, and no stored object refers to a missing one. In the viewer,
`ProjectState` calls `Project.save_*` and emits a signal (`rois_changed`,
`model_runs_changed`, ...) after each change.

## Loading

- `load_project_store()` reads the metadata; ROI pixels and model artifacts
  stay on disk unless asked for (`include_roi_spectra=`,
  `include_artifact_arrays=`). `hydrate_project_store()` builds the typed
  objects and checks their relationships, without Qt. `Project.open()` does
  both.
- Heavy arrays load when needed: a model's artifacts on first use
  (`ProjectState.ensure_model_run_artifact_refs`), ROI pixels for a fit or the
  Project ROIs tiles.
- Raw ROI pixels stay in their stored dtype (uint16 counts, or float32 for
  corrected data); working arrays are float32. Arrays are stored with Blosc
  (engiNIR 1.7.3; gzip before), see `hdf5_format.md`.
