# HDF5 Format Reference

enginir uses HDF5 for two distinct storage shapes:

1. **Simple spectral files** via `save_hdf5()` / `load_hdf5()`
2. **Project files v3** via the project-file API in `enginir.io`

This page is the concrete on-disk reference for the current implementation. For
scientific workflow semantics, read it together with the v3 contracts in
`contracts/`.

---

## 1. Simple spectral files

A simple spectral HDF5 file stores one spectral array, one wavelength axis, and
an optional metadata group.

```text
/
├── spectra        dataset -- (n_samples, n_wavelengths) or image cube
├── wavelengths    dataset -- (n_wavelengths,) nm
└── meta/          group -- optional metadata
    ├── @<attr>    scalar / string attrs
    └── <dataset>  array values
```

The KESTRELEYE raw-HDF5 profile is also accepted by the default reader when
`/spectra` is absent and a root `/data` dataset is present. It is treated as a
cube shaped `(along-track, cross-track, wavelength)` and metadata records
`hdf5_schema = "kestreleye_raw_hdf5"` plus `hdf5_spectra_key = "data"`.
Blosc-compressed HDF5 datasets are supported through the bundled `hdf5plugin`
dependency. If no `/wavelengths` dataset is present, band indices are used as
the x-axis and `meta["x_units"]` is `"band_index"` (otherwise `"nm"`).
For KESTRELEYE exports, the along-track length is acquisition-dependent, while
cross-track pixels and band count are camera-setup dependent; all three values
are read from the dataset shape instead of being treated as constants.
Other vendor HDF5 layouts are not implied by this profile; they should be added
as explicit named schemas when sample files and contracts are available.

**Axis convention for image cubes**: data is stored in the same orientation it
was loaded. No reordering occurs. The convention throughout the library is
`(along-track, cross-track, wavelength)`:

| Axis | Meaning |
|------|---------|
| 0 | along-track -- scan direction / time |
| 1 | cross-track -- spatial pixels across the slit |
| 2 | wavelength bands |

`load_envi()` returns `(rows, cols, n_bands)` and `load_eiff()` returns
`(n_frames, height, n_bands)`; both map to
`(along-track, cross-track, wavelength)`.

### API

```python
from enginir.io import save_hdf5, load_hdf5

save_hdf5("spectra.h5", X, wavelengths, meta={"instrument": "Lab NIR"})
X, wavelengths, meta = load_hdf5("spectra.h5")
```

`load_spectra()` dispatches here for `.h5` / `.hdf5` extensions; a project
file is detected and rejected with an error pointing to `load_project_store()`. `save_hdf5()` raises `ValueError` if
`len(wavelengths) != X.shape[-1]`.

---

## 2. Project files v3

A v3 project file is the persistent scientific workspace. It stores first-class
objects and relationships keyed by stable ids, not by labels.

Current project schema version:

```text
viewer-v3
```

### 2.1. Required top-level structure

Every canonical v3 project file must contain these required top-level object
namespaces. The groups may be empty in a newly created project.

```text
/
├── project_meta/
├── datasets/
├── targets/
├── target_assignments/
├── recipes/
├── signal_domains/
├── rois/
├── sample_sets/
├── model_runs/
└── predictions/
```

Canonical v3 project files must not use root `/calibration` or root `/models`
as their scientific source of truth. Root `/calibration` remains a rejected
pre-v3 shape. An extra legacy root `/models` group is ignored with a warning;
model runs are read exclusively from `/model_runs`.

### 2.2. Generic encoding rules

The object-specific save helpers all use the same mapping encoder:

- nested Python dictionaries become HDF5 groups
- scalar strings, bytes, booleans, integers, floats, and NumPy scalars become
  attributes
- zero-dimensional arrays become attributes
- string/object arrays become UTF-8 string datasets
- numeric arrays with `ndim <= 1` and `size <= 16` become attributes
- larger numeric arrays become compressed datasets
- `None` values are omitted in most payloads

Consequences for readers:

- contract field names are stable, but a small vector may be an attribute while
  a larger vector is a dataset
- readers must handle both HDF5 attrs and datasets for array-like metadata
- object ids are represented by their group names, not duplicated as required
  child fields
- no Python pickle state is part of the v3 project schema

Default compression for dataset writes is `gzip`.

### 2.3. Strict rejection rules

`load_project_store()` and the project-specific helpers reject non-canonical v3
files before reading project objects:

- missing `/project_meta`
- `project_meta/@schema_version` other than `viewer-v3`
- any missing required top-level object group
- root `/calibration`
- ROI-local `@y_value`
- prediction artifact groups with `@artifact_write_status = "writing"`
- prediction artifact groups with an unknown `@artifact_write_status` value

Unknown extra groups, datasets, and attributes are warning-only when
the file otherwise satisfies the complete current schema. Their contents are
not loaded, merged, or used as aliases for missing current fields. Other old
project shapes are allowed to fail loudly; no migration adapter is part of this
format. A legacy root `/models` group is one example of a harmless ignored
extra; `/model_runs` remains authoritative.

### 2.4. Metadata-first loading contract

The canonical project reader is metadata-first by default and supports two
independent opt-in switches for heavy arrays:

```python
# Opt in only when the caller really needs immediate array hydration.
load_project_store(
    "project.h5",
    include_roi_spectra=True,
    include_artifact_arrays=True,
)
```

By default, these stay disk-backed:

- `/rois/{roi_id}/spectra`
- `/rois/{roi_id}/calibration_refs/dark`
- `/rois/{roi_id}/calibration_refs/white`

Nested arrays under `artifact_refs` also stay disk-backed by default:

- model diagnostics and CV output arrays
- prediction `y_pred`
- prediction `spatial_map`

Scalar artifact metadata such as shapes, output names, min/max summaries,
chunking information, and write status remains available for project browsing,
inspector panels, and summaries.

Explicit lazy-load helpers:

- `load_roi_spectra()`, `load_roi_spectra_batch()`, `load_roi_spectra_matrix()`
- `iter_roi_calibration_refs()`
- `load_model_run_artifact_refs()` (optionally only selected `keys`)
- `load_model_run_prediction_artifact()`
- `load_prediction_run_artifact_refs()` (optionally only selected `keys`)
- `load_prediction_spatial_map_preview()`

---

## 3. Object layouts

The following layouts describe required v3 fields for the typed project models.
Optional fields are marked explicitly. Additional unknown fields may be present,
but readers should not require them for the current contract.

### 3.1. Project metadata

```text
/project_meta/
├── @project_id
├── @schema_version          "viewer-v3"
├── @created_at
└── @updated_at
```

### 3.2. Dataset references and dataset-owned calibration

```text
/datasets/{dataset_id}/
├── @source_path
├── @source_format
├── @source_modality
├── @cube_shape              (n_frames, height, n_bands)
├── @n_bands
├── @import_timestamp
├── @active_calibration_id   present after a calibration is saved
├── wavelengths              float32, shape (n_bands,)
├── loader_metadata/
└── calibrations/{calibration_id}/
    ├── @mode                raw_dn | dark_white_reference | precalibrated_import
    ├── @output_signal_unit  DN | Reflectance | Absorbance
    ├── @dark_ref_source     optional
    ├── @white_ref_source    optional
    ├── @black_level         optional
    ├── @white_level         optional
    ├── @created_at
    ├── dark_ref_data        optional embedded reference array
    └── white_ref_data       optional embedded reference array
```

Calibration is dataset-owned in v3. A project-global `/calibration` group is an
invalid pre-v3 shape.

Object ids are taken from the HDF5 path (`{dataset_id}`, `{calibration_id}`,
...). Writers do not need to store them as attributes; a stored id attribute
(for example `@dataset_id` on a calibration written from `to_dict()`) is
accepted when it equals the path id and rejected when it contradicts it. An
`@active_calibration_id` must name a calibration of the same dataset.

### 3.3. ROIs

```text
/rois/{roi_id}/
├── @dataset_id
├── @calibration_id
├── @name                   mutable display label
├── @source_file
├── @n_pixels
├── @created_at
├── @spatial_bounds         (x0, x1, y0, y1)
├── wavelengths             float32, shape (n_bands,)
├── spectra                 raw ROI matrix, shape (n_pixels, n_bands)
├── calibration_refs/       optional ROI-local extraction references
│   ├── dark
│   └── white
└── provenance/
```

`spectra` is the persisted raw ROI matrix and remains the scientific truth for
ROI extraction in v3. It keeps the source dtype: uint16 for raw DN sources,
float32 for precalibrated floating sources (float values are never truncated
to integers). Precalibrated ROIs store no `calibration_refs/`, because their
calibration mode carries no dark/white reference context. Target values do
not live on ROIs.

### 3.4. Targets and target assignments

```text
/targets/{target_id}/
├── @name
├── @unit
├── @value_type             currently float
├── @missing_value_policy   allowed | forbidden_for_modeling
└── @created_at

/target_assignments/{target_id}/{roi_id}/
├── @value                  finite float
└── @assigned_at
```

Target assignments are first-class objects keyed by `target_id` and `roi_id`.
ROI `@y_value` is not authoritative in v3. If present as an extra field it is
logged and ignored.

### 3.5. Preprocessing recipes

```text
/recipes/{recipe_id}/
├── @filter
├── @sg_window
├── @sg_polyorder
├── @snv
├── @msc
├── @emsc
├── @detrend
└── @detrend_order
```

Recipes are replayed during sample-set building, model fitting, and prediction.
The path component `{recipe_id}` is the stable identity. All eight fields are
required, including disabled switches. Unknown fields are logged and discarded
rather than treated as executable parameters.
Recipe content is immutable under one `{recipe_id}`: re-saving identical
content is allowed, while changed executable content requires a new id.
Stateful preprocessing that requires fitted training state must persist that
state before reopened-project prediction can use it.

For model runs using MSC or EMSC, the prediction-critical model artifact stores
that state under
`/model_runs/{model_run_id}/artifact_refs/prediction_model/preprocessing_state/`.
MSC stores its fitted reference spectrum. EMSC stores its fitted reference,
weights, optional interference basis, method, and polynomial order. Prediction
restores these values without fitting on prediction data.

### 3.6. Signal-domain contracts

```text
/signal_domains/{signal_domain_id}/
├── @input_format
├── @source_modality
├── @required_signal_unit
├── @required_calibration_mode
├── @requires_dark_ref
├── @requires_white_ref
├── @calibrated_input_allowed
├── @n_bands
├── wavelengths
├── @wavelength_tolerance_nm
├── @band_mapping_policy    exact | reject
├── @output_dtype           float32
└── @created_at
```

In v3, `exact` and `reject` do not imply interpolation or automatic band
remapping. They mean the imported data is already aligned for the saved model
workflow.

### 3.7. Sample sets

```text
/sample_sets/{sample_set_id}/
├── roi_ids
├── @target_id
├── @observation_unit       pixel
├── @target_level           roi
├── @grouping_unit          roi
├── @validation_unit        roi
├── @weighting_policy       per_pixel_unweighted
├── @reporting_level        roi
├── @roi_aggregation        mean
├── @preprocessing_recipe_id
├── dataset_ids
└── @created_at
```

The modeling semantics above are frozen for v3. Pixel-level observations are
grouped and validated by ROI, then reported at ROI level using mean aggregation.

### 3.8. Model runs

```text
/model_runs/{model_run_id}/
├── @sample_set_id
├── @signal_domain_id
├── @model_type             PLS | PCR
├── @name                   optional display name (mutable label)
├── hyperparameters/
├── cv_spec/
├── metrics/
├── artifact_refs/
│   ├── prediction_model/
│   ├── diagnostics/
│   └── cv_outputs/
└── @created_at
```

Model runs are not merely fitted numeric arrays. They persist enough references
and semantics to interpret the artifacts through the sample set, signal-domain
contract, recipe, target, and datasets.

#### `hyperparameters/`

Constructor keyword arguments for the normalized `model_type` backend
(`PLS` → `enginir.models.pls.PLSRegression`, `PCR` → `enginir.models.pcr.PCR`).
The estimator is reconstructable as `Backend(**hyperparameters)`; no pickled
state is stored.

#### `cv_spec/`

```text
@type                    GroupKFold | LeaveOneGroupOut
@n_splits                number of CV folds actually run
@requested               the requested strategy: "auto", "logo", or an integer fold count
@n_groups                number of ROI groups
@grouping_unit           roi
@validation_unit         roi
@reporting_level         roi
@roi_aggregation         mean
```

#### `metrics/`

Primary metrics are ROI-aggregated (one cross-validated prediction per
held-out ROI, aggregated by arithmetic mean); `pixel_*` variants are secondary
per-pixel diagnostics.

```text
@rmsecv                  primary ROI-aggregated RMSECV
@r2                      primary ROI-aggregated R²
@bias                    primary ROI-aggregated bias, mean(y_pred - y_ref)
@rpd                     primary ROI-aggregated RPD
@rpiq                    primary ROI-aggregated RPIQ
@primary_reporting_level roi
@primary_roi_aggregation mean
@n_rois
@n_pixels
@pixel_rmsecv            secondary per-pixel RMSECV
@pixel_r2                secondary per-pixel R²
@pixel_bias              secondary per-pixel bias
@pixel_rpd               secondary per-pixel RPD
@pixel_rpiq              secondary per-pixel RPIQ
```

#### `artifact_refs/prediction_model`

This is the lightweight payload used for reopened-project prediction. Loading it
must not hydrate diagnostics arrays.

Fields:

```text
@model_type              PLSRegression | PCR
wavelengths             float32, shape (n_bands,)
target_names            string array
coef_                    shape (n_bands,) or (n_bands, n_targets)
intercept_               shape (1,) or (n_targets,)
preprocessing_state      optional group, fitted MSC/EMSC parameters
```

Both model types are replayed as the affine map
`y = X_preprocessed @ coef_ + intercept_`, where `X_preprocessed` is the
dataset after signal-domain conversion and the sample set's preprocessing
recipe.

#### `artifact_refs/diagnostics`

Diagnostics are loaded only by diagnostics workflows. The training scores
(`x_scores_`, `scores_`) have one row per pixel; X weights and Y loadings let
VIP be computed without a refit.

PLSRegression diagnostics:

```text
wavelengths
coef_
x_loadings_
x_weights_
y_loadings_
x_scores_
```

PCR diagnostics:

```text
wavelengths
coef_
loadings_
scores_
```

#### `artifact_refs/cv_outputs`

```text
roi_ids                  string array, one per held-out ROI
y_true_roi               float32, reference value per ROI
y_pred_roi               float32, aggregated cross-validated prediction per ROI
```

### 3.9. Prediction runs

```text
/predictions/{prediction_run_id}/
├── @dataset_id
├── @calibration_id
├── @model_run_id
├── @signal_domain_id
├── @preprocessing_recipe_id
├── @created_at
└── artifact_refs/
    ├── @artifact_write_status     writing | complete
    ├── @source_path
    ├── @output_dtype              float32
    ├── @prediction_shape
    ├── @spatial_shape             optional first two map dimensions
    ├── @spatial_map_shape         optional full spatial-map shape
    ├── @n_outputs
    ├── @output_axis               shape of the output dimensions, e.g. (n_outputs,); multi-output only
    ├── @output_index              0 for single-output predictions
    ├── output_names               optional string array
    ├── @chunk_axis                optional; frames (image cubes) or rows (spectra matrices)
    ├── @chunk_size                optional; frames (or rows) per chunk
    ├── @n_chunks                  optional
    ├── @completed_chunks          optional
    ├── @clip_low_count            optional
    ├── @clip_high_count           optional
    ├── @near_zero_denominator_count optional
    ├── @prediction_min            optional
    ├── @prediction_max            optional
    ├── y_pred                     canonical float32 dataset
    └── spatial_map                optional legacy/materialized map dataset
```

`artifact_write_status` is set to `writing` during incremental writes and to
`complete` only after finalization. A prediction run still marked `writing` is
not a completed v3 project state and is rejected on project open, as is any
status value other than `writing` or `complete`. An absent attribute means
`complete`: writers that persist an already-complete run in one operation may
omit it. Failed or cancelled staged runs are discarded (the prediction group
is deleted); no `failed` or `cancelled` status value is ever persisted.
Staging refuses an id that already exists, and discarding refuses a completed
run, so a failed attempt can never remove finished results.

Prediction summary views should use metadata such as `prediction_shape`,
`spatial_map_shape`, `prediction_min`, and `prediction_max` rather than
hydrating `y_pred` or `spatial_map`.

Current writers store `y_pred` once and derive spatial views from
`spatial_map_shape`. Readers still accept older projects containing a separate
materialized `spatial_map` dataset.

---

## 4. Creating and writing a project

```python
from enginir.io import create_project

create_project("project.h5")
```

This creates the full v3 top-level layout and writes `project_meta`. Object
helpers replace the existing group for the same id:

- `save_dataset_ref()`
- `save_calibration_spec()`
- `save_training_roi()`
- `save_target_definition()`
- `save_target_assignment()`
- `save_preprocessing_recipe()`
- `save_signal_domain_contract()`
- `save_sample_set()`
- `save_model_run()`
- `save_prediction_run()`

For normal work, use `enginir.Project`, which creates the bookkeeping objects
and checks references before writing. The writers above validate each object
on their own but not its references to other objects.

Project object replacement is staged in a temporary HDF5 group and published
only after serialization succeeds. A serialization failure therefore leaves
the previously published object intact.

Chunked prediction keeps one raw-source HDF5 handle and one project artifact
writer session open for the duration of the chunk loop. The default image
chunk is 512 frames; raw chunks retain their stored dtype until calibration.

---

## 5. Loading the project graph

Use `load_project_store()` for the canonical v3 read path:

```python
from enginir.io import load_project_store

store = load_project_store("project.h5")
print(store["project_meta"]["schema_version"])
print(store["datasets"]["dataset_1"]["active_calibration_id"])
print(store["rois"]["roi_1"]["dataset_id"])
```

Use explicit loaders for heavy arrays:

```python
from enginir.io import (
    load_model_run_prediction_artifact,
    load_prediction_run_artifact_refs,
    load_prediction_spatial_map_preview,
)

model_blob = load_model_run_prediction_artifact("project.h5", "model_run_1")
preview = load_prediction_spatial_map_preview("project.h5", "prediction_run_1")
y_pred = load_prediction_run_artifact_refs(
    "project.h5",
    "prediction_run_1",
    keys=("y_pred",),
)["y_pred"]
```

---

## 6. Prediction artifact writing

Small or imported prediction artifacts may be written with `save_prediction_run()`.
For runtime prediction and large maps, the canonical scalable path is:

1. `initialize_prediction_run_artifacts()`
2. `open_prediction_run_artifact_writer()`, calling `write_chunk()` for each chunk
3. `finalize_prediction_run_artifacts()` only after all chunks succeed
4. `discard_prediction_run_artifacts()` on failure

Example:

```python
from enginir.io import (
    initialize_prediction_run_artifacts,
    open_prediction_run_artifact_writer,
    finalize_prediction_run_artifacts,
)

initialize_prediction_run_artifacts(
    "project.h5",
    "prediction_run_1",
    {
        "dataset_id": "dataset_1",
        "calibration_id": "cal_1",
        "model_run_id": "model_run_1",
        "signal_domain_id": "signal_domain_1",
        "preprocessing_recipe_id": "recipe_1",
        "artifact_refs": {
            "source_path": "/data/predict.eiff",
            "prediction_shape": (300,),
            "spatial_shape": (20, 15),
            "spatial_map_shape": (20, 15),
            "output_dtype": "float32",
            "chunk_size": 4,
            "n_chunks": 5,
        },
        "created_at": "2026-04-02T10:09:00+00:00",
    },
    y_pred_shape=(300,),
    spatial_map_shape=(20, 15),
)

with open_prediction_run_artifact_writer("project.h5", "prediction_run_1") as writer:
    for row_start, frame_start, y_chunk, map_chunk in chunks:
        writer.write_chunk(
            row_start=row_start,
            y_pred_chunk=y_chunk,
            frame_start=frame_start,
            spatial_map_chunk=map_chunk,
        )

finalize_prediction_run_artifacts(
    "project.h5",
    "prediction_run_1",
    {
        "completed_chunks": 5,
        "prediction_min": 0.1,
        "prediction_max": 12.8,
        "clip_low_count": 0,
        "clip_high_count": 3,
    },
)
```

---

## 7. Inspecting project files

```bash
# List the full top-level structure
h5dump -n project.h5

# Show project metadata attrs
h5dump -A -g /project_meta project.h5

# Show one ROI group
h5dump -A -g /rois/roi_1 project.h5

# Show one model-run prediction artifact group
h5dump -A -g /model_runs/model_run_1/artifact_refs/prediction_model project.h5

# Show one prediction-run output dataset
h5dump -d /predictions/prediction_run_1/artifact_refs/y_pred project.h5
```

HDFView, MATLAB, Julia, and R (`hdf5r`) can inspect these files without any
enginir dependency.

---

## 8. Operational notes and limits

- `load_project_store()` is the canonical v3 read path.
- `create_project()` creates a fresh file in write mode.
- Object-specific save helpers replace existing groups for the same id.
- Project files do not embed full source datasets; they store dataset
  references plus persisted ROI spectra and project artifacts.
- Project runtime model artifacts currently support reopened-project prediction
  for PLS and PCR model runs.
