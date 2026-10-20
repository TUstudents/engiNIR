# Workspace Domain Model

Specification of the objects of a project, held by `enginir.Project` (as an
`enginir.project.ProjectWorkspace`) and persisted to HDF5 v3 project files.
The viewer's `ProjectState` mirrors them for its widgets.

---

## Entity definitions

| Entity | Key | Defined in | Description |
|--------|-----|-----------|-------------|
| `DatasetRef` | `dataset_id` | `project/_models.py` | Pointer to one HSI data file: source path, format, modality, cube shape, wavelengths |
| `CalibrationSpec` | `calibration_id` | `project/_models.py` | How to convert raw DN to a signal unit: mode (dark_white_reference / precalibrated_import / raw_dn), optional dark/white ref arrays or scalar levels, output signal unit |
| `TrainingROI` | `roi_id` | `project/_models.py` | Spatial bounding box on one dataset: name, spatial_bounds (f0, f1, r0, r1), n_pixels, wavelengths, references one dataset + one calibration, provenance dict |
| `TargetDefinition` | `target_id` | `project/_models.py` | What we are predicting: name, unit (e.g. "wt%"), value_type ("float"), missing_value_policy |
| `TargetAssignment` | `(roi_id, target_id)` | `project/_models.py` | One numeric value linking one ROI to one target |
| `SampleSet` | `sample_set_id` | `project/_models.py` | A labeled collection of ROIs for one target: roi_ids, preprocessing_recipe_id, observation/target/grouping/validation/weighting/reporting policies |
| `SignalDomainContract` | `signal_domain_id` | `project/_models.py` | Compatibility contract: required signal unit, calibration mode, band count, wavelengths, tolerance, band mapping policy |
| `ModelRun` | `model_run_id` | `project/_models.py` | One fitted model: references one sample set + one signal domain, model type, hyperparameters, cv_spec, metrics, artifact_refs, optional display name |
| `PredictionRun` | `prediction_run_id` | `project/_models.py` | One prediction: references one model run + one dataset + one calibration, artifact_refs (spatial_map, y_pred) |

---

## Relationships

```
DatasetRef  1───*  CalibrationSpec     (exactly one is active per dataset in v3)
DatasetRef  1───*  TrainingROI         (one dataset has many ROIs)
CalibrationSpec 1───*  TrainingROI     (one calibration used by many ROIs)

TargetDefinition  1───*  TargetAssignment  (one target assigned to many ROIs)
TrainingROI       1───*  TargetAssignment  (one ROI can have many target values)

SampleSet  *───1  TargetDefinition     (one sample set targets one property)
SampleSet  *───*  TrainingROI          (via roi_ids list)

ModelRun  *───1  SampleSet             (one model trained on one sample set)
ModelRun  *───1  SignalDomainContract   (one model has one signal contract)

PredictionRun  *───1  ModelRun          (one prediction uses one model)
PredictionRun  *───1  DatasetRef        (one prediction on one dataset)
PredictionRun  *───1  CalibrationSpec   (one prediction with one calibration)
```

---

## HDF5 storage mapping (v3 project files)

```
/project_meta/                @schema_version, @created_at, @updated_at
/datasets/{dataset_id}/       source_path, source_format, source_modality,
                              cube_shape, n_bands, wavelengths, loader_metadata,
                              @active_calibration_id
  /calibrations/{cal_id}/     mode, output_signal_unit, dark_ref_data,
                              white_ref_data, black_level, white_level
/targets/{target_id}/         name, unit, value_type, missing_value_policy
/target_assignments/{target_id}/{roi_id}/  value, assigned_at
/rois/{roi_id}/               dataset_id, calibration_id, name, wavelengths,
                              n_pixels, spatial_bounds, source_file,
                              spectra (uint16 dataset), /provenance/
/recipes/{recipe_id}/         eight required executable fields: filter,
                              sg_window, sg_polyorder, snv, msc, emsc,
                              detrend, detrend_order
/signal_domains/{sd_id}/      input_format, source_modality, required_signal_unit,
                              required_calibration_mode, n_bands, wavelengths, ...
/sample_sets/{ss_id}/         target_id, roi_ids, preprocessing_recipe_id,
                              observation_unit, validation_unit, ...
/model_runs/{mr_id}/          sample_set_id, signal_domain_id, model_type,
                              hyperparameters, cv_spec, metrics, artifact_refs
                              /artifact_refs/prediction_model/ (coef_, intercept_, ...)
                              /artifact_refs/diagnostics/ (loadings, scores, ...)
/predictions/{pr_id}/         dataset_id, calibration_id, model_run_id,
                              signal_domain_id, preprocessing_recipe_id,
                              artifact_refs (spatial_map, y_pred)
```

---

## Workspace lifecycle

From an empty project to a prediction; the viewer calls the same rules from
its workspaces.

| Step | `enginir.Project` | Objects stored |
|---|---|---|
| 1. Create | `Project.create(path)` | project metadata |
| 2. Register data | `project.add_dataset(path, dark=, white=, unit=)` | `DatasetRef`, `CalibrationSpec` (reused when identical; becomes active) |
| 3. Define a target | `project.add_target(name, unit=)` | `TargetDefinition` |
| 4. Draw ROIs | `dataset.add_roi(name, frames=, rows=, values=)` | `TrainingROI` with spectra and reference pixels, `TargetAssignment` per value |
| 5. Fit | `project.fit(target, recipe=, model=, n_components=)` | recipe (by content), `SampleSet` (reused), `SignalDomainContract` (reused), `ModelRun` |
| 6. Predict | `model.predict(path)` | `DatasetRef`/`CalibrationSpec` if new, `PredictionRun` with its arrays |

Each step validates first and then writes, in dependency order, so a failed
step stores nothing, and a stored object never references a missing one. In
the viewer, `ProjectState` calls `Project.save_*` and emits a signal
(`rois_changed`, `model_runs_changed`, …) after each change.

---

## Memory-aware hydration rules

A project is split into light metadata and heavy array
artifacts.

- `load_project_store()` is the canonical reader and is metadata-first by default:
  - ROI spectra stay on disk unless `include_roi_spectra=True`
  - nested `artifact_refs` arrays stay on disk unless `include_artifact_arrays=True`
- `hydrate_project_store()` converts that dictionary into typed objects and
  validates relationships without importing Qt.
- `Project.open()` uses both; the viewer's `ProjectState.open_project()` opens
  a `Project` and emits its signals.
- Heavy persisted arrays are loaded lazily only when needed:
  - `ModelRun.artifact_refs` through `ensure_model_run_artifact_refs(...)`
  - `PredictionRun.artifact_refs` through `ensure_prediction_artifact_refs(...)`
- This means project-first shell startup, browser navigation, inspector updates,
  and metadata edits do not materialize model CV arrays, prediction maps, or ROI
  spectra unnecessarily.

### Working dtype rules

The project file may contain mixed numeric dtypes; the viewer
and modeling/prediction pipelines follow these defaults:

- raw ROI spectra remain the scientific truth and stay in their stored raw dtype
- viewer/modeling/prediction working arrays default to `float32`
- persisted prediction/model artifact arrays such as `y_pred`, `spatial_map`,
  `y_true_roi`, and `y_pred_roi` are also written as `float32`

These rules are intended to reduce peak memory without changing the frozen v3
project semantics.
