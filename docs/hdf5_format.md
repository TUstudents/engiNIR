# HDF5 format reference

enginir uses HDF5 for two distinct storage shapes:

1. **Simple spectral files** via `save_hdf5()` / `load_hdf5()`
2. **Project files** (format `enginir-1.5`) via the project-file API in `enginir.io`

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

## 2. Project files (format `enginir-1.5`)

A v3 project file is the persistent scientific workspace. It stores first-class
objects and relationships keyed by stable ids, not by labels.

Project file format (`/project_meta/@FileSchemaVersion`):

```text
enginir-1.5
```

engiNIR 1.5 reads and writes only this format. A file with any other
format, or without `FileSchemaVersion` (every project written before 1.5,
which stored `schema_version = "viewer-v3"`), is refused with one message;
there is no migration or repair. The format changes only when a field
becomes required, is removed or changes meaning; later releases keep
writing `enginir-1.5` while it does not.

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
├── predictions/     empty unless a script stores a prediction
└── staging/         optional; staged prediction writes not yet finalized (§3.9)
```

Canonical v3 project files must not use root `/calibration` or root `/models`
as their scientific source of truth. Root `/calibration` is refused. A root `/models`
group is an extra: it is ignored with a warning; models are read exclusively
from `/model_runs`.

### 2.2. Generic encoding rules

The object-specific save helpers all use the same mapping encoder:

- nested Python dictionaries become HDF5 groups
- scalar strings, bytes, booleans, integers, floats, and NumPy scalars become
  attributes
- zero-dimensional arrays become attributes
- string/object arrays become UTF-8 string datasets
- numeric arrays with `ndim <= 1` and `size <= 16` become attributes
- larger numeric arrays become compressed datasets
- values that are not defined in a case are written, never omitted: NaN for
  numbers, an empty string for text, an empty array for arrays (Python
  `None` is written the same way)

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
- `project_meta/@FileSchemaVersion` other than `enginir-1.5`, or missing
  (checked first, before anything else is read)
- any missing required top-level object group
- a missing or malformed **core** field: ids and references, datasets and
  reference corrections, ROI spectra, wavelengths, references and `job`, targets and
  target values, every recipe field, input contracts, calibration sets,
  a model's `model_type`, `hyperparameters`, `fit_spec`, `software` and
  executable `prediction_model` (with its wavelengths and target names), and a stored prediction's references, `y_pred`,
  shapes and dtype (`project-schema-contract_v3.md`, Core and display data);
  nothing is filled from a default
- a timestamp that does not parse or has no UTC offset
- root `/calibration`

**Display data** never stops a project from opening: metrics, the component
curve, the calibration diagnostics, `cv_spec`, `cv_outputs` and prediction
telemetry may be missing or malformed; the reader logs it, and the viewer
shows "–" in its place. Writers still write all of it.

Unknown extra groups, datasets, and attributes (e.g. an ROI `@y_value`) are
warning-only when the file otherwise satisfies the format. Their contents are not
loaded, merged, or used as aliases for missing fields. A root `/models`
group is one example of a harmless ignored extra; `/model_runs` remains
authoritative. `/staging/` is not an extra: a stale staging group (left by
a crash) is logged once and otherwise ignored (§3.9).

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

Scalar artifact metadata such as shapes, output names, min/max summaries,
and chunking information remain available for project browsing,
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

The following layouts describe the fields of the `enginir-1.5` format.
Writers write every field shown, except those marked "optional"; a value
that is not defined in a case is NaN (numbers), an empty string or an empty
array. The reader refuses a file with a missing core field and tolerates
missing display data (§2.3). Additional unknown fields are logged and
ignored.

### 3.1. Project metadata

```text
/project_meta/
├── @project_id
├── @FileSchemaVersion       "enginir-1.5", the file format; set once
├── @FileCreator             "engiNIR <version>" that created the file; set once
├── @FileWriter              "engiNIR <version>" that last wrote the file; every write
├── @created_at              set once
└── @updated_at              every write, together with @FileWriter
```

`<version>` is the installed package version (`enginir.__version__`), e.g.
`engiNIR 1.5.1`, or a development build as it is (`1.5.1.dev3+g1234abc`).
`FileCreator` and `FileWriter` are informational and never decide whether a
file is read. Every timestamp in the file (`created_at`, `updated_at`,
`import_timestamp`, `assigned_at`) is ISO 8601 in UTC with an explicit
offset and microseconds, e.g. `2026-11-12T09:56:05.852000+00:00`.

### 3.2. Dataset references and dataset-owned reference correction

```text
/datasets/{dataset_id}/
├── @source_path
├── @source_format
├── @source_modality
├── @cube_shape              (n_frames, height, n_bands)
├── @n_bands
├── @import_timestamp
├── @active_calibration_id   the dataset's reference correction (written with the dataset)
├── wavelengths              float32, shape (n_bands,)
├── loader_metadata/
└── calibrations/{calibration_id}/
    ├── @mode                raw_dn | dark_white_reference | precalibrated_import
    ├── @output_signal_unit  DN | Reflectance | Absorbance
    ├── @dark_ref_source     empty string when the mode does not use it
    ├── @white_ref_source    empty string when the mode does not use it
    ├── @black_level         NaN when the mode does not use it
    ├── @white_level         NaN when the mode does not use it
    ├── @created_at
    ├── dark_ref_data        embedded reference array; empty when not used
    └── white_ref_data       embedded reference array; empty when not used
```

Reference correction is dataset-owned. A project-global `/calibration` group
is refused.

Object ids are taken from the HDF5 path (`{dataset_id}`, `{calibration_id}`,
...). Writers do not need to store them as attributes; a stored id attribute
(for example `@dataset_id` on a reference correction written from `to_dict()`) is
accepted when it equals the path id and rejected when it contradicts it. An
`@active_calibration_id` must name a reference correction of the same dataset.

### 3.3. ROIs

```text
/rois/{roi_id}/
├── @dataset_id
├── @calibration_id
├── @name                   mutable display label
├── @sample                 sample label (mutable; empty string: none); ROIs with equal
│                           labels are one physical sample for fold groups
├── @job                    calibration | prediction (mutable, save_roi_job);
│                           prediction ROIs never enter a calibration set
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
ROI extraction in v3. It keeps the source dtype: uint16 for Raw counts (DN) sources,
float32 for already corrected floating sources (float values are never truncated
to integers). Already corrected ROIs store no `calibration_refs/`, because their
reference correction mode carries no dark/white reference context. Target values do
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
├── @detrend_order
├── @sg_edges            drop | interp (normalized to drop without a Savitzky-Golay filter)
├── @asls                true | false
├── @asls_lambda         float64 (normalized to 10000 when @asls is false)
├── @asls_p              float64 (0.001 when @asls is false)
├── @asls_iterations     integer (10 when @asls is false)
├── bands_keep           float64 (n, 2), inclusive [low, high] ranges to keep; n = 0: every band
└── bands_exclude        float64 (n, 2), inclusive ranges to leave out; n may be 0
```

Recipes are replayed during calibration-set building, model fitting, and
prediction. The path component `{recipe_id}` is the stable identity. Every
field is written for every recipe, including disabled switches; a field
without effect holds its normalized default, so equal recipes have equal
content and ids (`project-schema-contract_v3.md`, Object identity rules §4). With `@sg_edges = "drop"` the
preprocessed spectra have 2⌊`sg_window`/2⌋ fewer bands: the model's
arrays over bands (`prediction_model`: `wavelengths`, `coef_`, `x_mean`,
`x_std`, `rotations`, `loadings`; `diagnostics`: `wavelengths`, loadings,
coefficients, VIP) use that shorter axis (the "model bands" below); MSC/EMSC
state uses the axis after Savitzky-Golay, before a band selection. `@asls_lambda` and `@asls_p` are float64,
`@asls_iterations` an integer. `bands_keep` and `bands_exclude` (band
selection, the last step) are always written, with shape (0, 2) when empty;
their values are on the axis after Savitzky-Golay (nm, or band numbers for data
without wavelengths), ordered, sorted and merged, and the model and its
stored wavelengths use the selected bands. Unknown fields are logged and discarded
rather than treated as executable parameters.
Recipe content is immutable under one `{recipe_id}`: re-saving identical
content is allowed, while changed executable content requires a new id.
Stateful preprocessing that requires fitted calibration state must persist that
state before reopened-project prediction can use it.

For models using MSC or EMSC, the prediction-critical model artifact stores
that state under
`/model_runs/{model_run_id}/artifact_refs/prediction_model/preprocessing_state/`.
MSC stores its fitted reference spectrum. EMSC stores its fitted reference,
weights, optional interference basis, method, and polynomial order. Prediction
restores these values without fitting on prediction data.

### 3.6. Input contracts

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
├── @band_mapping_policy    exact (the only value; reserved for a future remapping contract)
├── @output_dtype           float32
└── @created_at
```

`exact` does not imply interpolation or automatic band remapping. It means
the imported data is already aligned for the saved model workflow.

### 3.7. Calibration sets

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

The modeling semantics above are frozen for v3. Pixels as the data points of the fit are
grouped and validated by ROI, then reported at ROI level using mean aggregation.

### 3.8. Models

```text
/model_runs/{model_run_id}/
├── @sample_set_id
├── @signal_domain_id
├── @model_type             PLS | PCR
├── @name                   display name (mutable label; empty string: none)
├── hyperparameters/
├── cv_spec/
├── fit_spec/
├── software/               versions the fit ran with
├── metrics/
├── artifact_refs/
│   ├── prediction_model/
│   ├── diagnostics/
│   ├── cv_outputs/
│   └── exclusion/          empty arrays without DModX exclusion
└── @created_at
```

Models are not merely fitted numeric arrays. They persist enough references
and semantics to interpret the artifacts through the calibration set, input contract, recipe, target, and datasets.

#### `hyperparameters/`

Constructor keyword arguments for the normalized `model_type` backend
(`PLS` → `enginir.models.pls.PLSRegression`, `PCR` → `enginir.models.pcr.PCR`).
The estimator is reconstructable as `Backend(**hyperparameters)`; no pickled
state is stored.

Scaling of X and Y:

```text
@x_scaling               center | uv | pareto   (default center)
@y_scaling               center | uv | pareto   (PLS only, default uv)
```

#### `cv_spec/`

```text
@type                    RoundRobinGroups | GroupKFold | LeaveOneGroupOut
@n_splits                number of CV folds actually run
@requested               the requested strategy: "auto", "logo", an integer fold count,
                         or "fit_spec" when the folds came from a fit specification
@n_groups                number of fold groups G (= number of ROIs with group_by roi)
@grouping_unit           roi
@validation_unit         roi
@reporting_level         roi
@roi_aggregation         mean
@warnings                comma-separated fold warnings: few_groups, one_level_per_file;
                         empty string when none (§7.4 of the model-run contract)
```

#### `fit_spec/`

What was requested (`sample-set-model-run-contract_v3.md` §7).

```text
folds/
  @rule                  round_robin | leave_one_group_out | group_kfold (optional
                         rule, requested explicitly; only with group_by roi)
  @n_folds               requested fold count F (k = min(F, number of fold groups))
  @group_by              roi | sample | target
  groups                 int64, one per ROI in sample-set roi_ids order: index of
                         its fold group in group order, as used by the fit;
                         exactly 0 .. G-1, each used (attribute, or a dataset
                         for large sample sets)
pixels/
  @per_roi               "all" or the number of pixels drawn per ROI
  @seed                  integer seed of the pixel subsample, 0 <= seed < 2**63
  @checksum              SHA-256 (hex) of the drawn indices; "" for "all"
exclusion/
  @method                none | dmodx
  @level                 probability of the DCrit limit
  @rounds                1 | 2
  @max_fraction          largest excluded share of an ROI's drawn pixels, all rounds
  @formula               DModX definition: dmodx-1 (unweighted) | dmodx-2 (MPow-weighted)
components/
  @rule                  fixed | one_se | min_rmsecv | q2_limit | f_test | q2_lookahead
  @noise_guard           on | off
  @max                   largest number of components of the curve (Amax)
  @rule_choice           the rule's number before the noise guard and ceiling
  @selected              number of components of the run (= hyperparameters/n_components)
```

Every key is written for every model, also when it has no effect.

#### `software/`

```text
@enginir                 installed engiNIR version, e.g. 1.5.1
@numpy                   installed numpy version
@scipy                   installed scipy version
@scikit-learn            installed scikit-learn version
```

Written by every fit and refit, from the package metadata; part of the
model's content. A refit whose metrics differ names the versions that differ
(`sample-set-model-run-contract_v3.md` §6).

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
@n_pixels                drawn pixels (all pixels for per_roi = all); the pixel_* metrics use them
@n_excluded_pixels       pixels excluded by DModX from the final model
@pixel_rmsee             RMSEE over the fitted calibration pixels, N - 1 - A
@pixel_r2_adj            adjusted R² over the fitted calibration pixels
@rmsee                   RMSEE over ROI means, R - 1 - A; NaN when R - 1 - A < 1
@r2_adj                  adjusted R² over ROI means; NaN when R - 1 - A < 1
@interval_coverage_roi   share of held-out ROI means inside their 0.95 mean interval (PLS/PCR;
                         NaN with count 0 when there is no interval)
@interval_coverage_roi_n ROIs counted for it
@interval_coverage_pixel share of held-out pixels inside their 0.95 pixel interval
@interval_coverage_pixel_n pixels counted for it
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
x_mean, x_std            float64 (n_bands,), the effective center and divisor
rotations, loadings      (n_bands, A), scores = scaled X @ rotations
score_variances          (A,), calibration score variances (N - 1); NaN for a component without variance
@s0                      calibration pooled residual SD (unweighted); NaN when K - A < 1
@n_train                 kept calibration pixels N of the final model (after DModX exclusion)
@n_components            A
@t2_crit_099, @dcrit_099 limits for the applicability diagnostics (NaN when undefined)
```

`x_mean` is the center in the model's input space: for PCR the PCA mean is
folded in (`x_mean_ + pca_.mean_ * x_std_`), and `rotations` = `loadings` =
the PCA components.

The projection fields serve the applicability diagnostics of predictions
(DModXPS, T²PS; `prediction-runtime-contract_v3.md`). Every PLS and PCR
model has them.

Both model types are replayed as the affine map
`y = X_preprocessed @ coef_ + intercept_`, where `X_preprocessed` is the
dataset after signal conversion and the calibration set's preprocessing
recipe.

#### `artifact_refs/diagnostics`

Diagnostics are loaded only by diagnostics workflows. The calibration scores
(`x_scores_`, `scores_`) have one row per pixel.

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

Both (`sample-set-model-run-contract_v3.md` §7.5):

```text
coef_cs                  coefficients for scaled, centered X and scaled Y (CoeffCS)
t2                       float32, Hotelling's T² per drawn pixel (ROI order, then drawn index)
dmodx                    float32, normalized unweighted DModX per drawn pixel (NaN when K - A < 1)
@t2_crit_095, @t2_crit_099   T²Crit with A components
@t2_crit_2_095, @t2_crit_2_099   T²Crit for the first two components (ellipse; NaN when A < 2)
@dcrit_099               DCrit at 0.99 (normalized; NaN when K - A < 1)
roi_share_above_dcrit    float32 per ROI (cv_outputs/roi_ids order), share of pixels above DCrit
roi_share_above_t2crit   float32 per ROI, share of pixels above T²Crit (0.99)
roi_scores               float32 (n_rois, A), mean calibration score of each ROI's fitted pixels
roi_score_pixels         int (n_rois,), fitted pixels behind each ROI score (0: NaN scores)
```

Both (band importance, §7.5; one target):

```text
mean_spectrum            float32, mean preprocessed spectrum of the fitted pixels (model axis)
vip                      float32, VIP per band (PLS only; mean of VIP² is 1)
selectivity_ratio        float32, explained / residual variance of the target projection per band (one target; NaN: no residual)
```

#### `artifact_refs/component_curve`

Every PLS and PCR model (§7.4). One entry per number of components
1 … Amax:

```text
n_components             int, 1 .. Amax
rmsecv                   ROI-level RMSECV
rmsecv_se                its standard error
q2                       Q² per component
q2_cum                   Q²(cum)
r2x_cum                  R²X(cum)
r2y_cum                  R²Y(cum) over drawn pixels (never decreases)
r2y_cum_roi              R²Y(cum) over ROI means
rmsee_roi                RMSEE over ROI means, √(RSS/(R − a − 1)); NaN when R − a − 1 < 1
dw_weights               Durbin–Watson of each component's weight vector
noise                    bool: ΔR²X < 0.01 and Q²ₐ ≤ L, a noise component
@suggested_one_se, @suggested_min_rmsecv, @suggested_q2_limit
@suggested_f_test, @suggested_q2_lookahead
@structure_ceiling       1 + number of components with ΔR²X ≥ 0.005, at most Amax
```

#### `artifact_refs/cv_outputs`

```text
roi_ids                  string array, one per held-out ROI
y_true_roi               float32, target value per ROI
y_pred_roi               float32, aggregated cross-validated prediction per ROI
```

#### `artifact_refs/exclusion`

Always written; with `fit_spec/exclusion/@method = none` its arrays are
empty and `@dcrit` is NaN. Indices refer to the row order of each ROI's
stored spectra.

```text
roi_ids                  string array, ROIs with excluded pixels, sample-set order
counts                   int32, number of excluded pixels per listed ROI
pixel_indices            uint32, the excluded indices, concatenated in roi_ids order
flagged_roi_ids          string array, ROIs with more pixels above DCrit than max_fraction allows
@dcrit                   DCrit of the last exclusion round of the final model
```

### 3.9. Prediction runs

```text
/predictions/{prediction_run_id}/     a stored prediction (only when a script stores one)
├── @dataset_id                    core
├── @calibration_id                core
├── @model_run_id                  core
├── @signal_domain_id              core
├── @preprocessing_recipe_id       core
├── @created_at                    core
└── artifact_refs/
    ├── @source_path               telemetry
    ├── @output_dtype              core: float32
    ├── @prediction_shape          core
    ├── @spatial_shape             telemetry: first two map dimensions
    ├── @spatial_map_shape         core for image datasets: the map shape of y_pred
    ├── @n_outputs                 telemetry
    ├── @output_axis               telemetry: shape of the output dimensions, e.g. (n_outputs,); empty for one output
    ├── @output_index              telemetry: 0 for single-output predictions
    ├── output_names               telemetry: string array
    ├── @chunk_axis                telemetry; frames (image cubes) or rows (spectra matrices)
    ├── @chunk_size                telemetry; frames (or rows) per chunk
    ├── @n_chunks                  telemetry
    ├── @completed_chunks          telemetry
    ├── @clip_low_count            telemetry
    ├── @clip_high_count           telemetry
    ├── @near_zero_denominator_count telemetry
    ├── @prediction_min            telemetry
    ├── @prediction_max            telemetry
    └── y_pred                     core: float32 dataset

/staging/{prediction_run_id}/         optional: a staged write not finalized yet
```

The facade and the viewer never store predictions; only the low-level
writers below do, when a script asks for it. A staged write builds the run
under `/staging/{prediction_run_id}/` and finalizes it by moving the group
to `/predictions/{prediction_run_id}/`, so `/predictions` only ever holds
complete runs and there is no status attribute. A failed or canceled staged
write deletes its staging group. A staging group left by a crash or power
loss does not stop the project from opening: the reader logs one warning,
and only `discard_staged_predictions()` deletes it (a write, which never
touches `/predictions`). Staging refuses an id that already exists in
`/predictions` or `/staging`; one writer at a time works on a project file.

Core attributes decide whether the run is read (a missing one refuses the
project); the others are telemetry: written by the writers, tolerated when
missing (§2.3). Clipping counts are 0 when nothing was clipped.

Prediction summary views should use metadata such as `prediction_shape`,
`spatial_map_shape`, `prediction_min`, and `prediction_max` rather than
hydrating `y_pred`. `y_pred` is stored once; spatial views are derived from
`spatial_map_shape`.

---

## 4. Creating and writing a project

```python
from enginir.io import create_project

create_project("project.h5")
```

This creates the full top-level layout and writes `project_meta` with
`FileSchemaVersion`, `FileCreator`, `FileWriter` and the timestamps. Every
later write updates `FileWriter` and `updated_at`. Object
helpers replace the existing group for the same id:

- `save_dataset_ref()`
- `save_calibration_spec()`
- `save_training_roi()`
- `save_roi_job()`, `save_roi_sample()` (labels, in place)
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
chunk is 512 frames; raw chunks retain their stored dtype until reference correction.

---

## 5. Loading the project graph

Use `load_project_store()` for the canonical v3 read path:

```python
from enginir.io import load_project_store

store = load_project_store("project.h5")
print(store["project_meta"]["FileSchemaVersion"], store["project_meta"]["FileWriter"])
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

1. `initialize_prediction_run_artifacts()` creates the run under `/staging/`
2. `open_prediction_run_artifact_writer()`, calling `write_chunk()` for each chunk
3. `finalize_prediction_run_artifacts()` only after all chunks succeed; it
   moves the run to `/predictions/`
4. `discard_prediction_run_artifacts()` on failure; `discard_staged_predictions()`
   removes staging groups left by a crash

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
        "created_at": "2026-04-02T10:09:00.000000+00:00",
    },
    y_pred_shape=(300,),
)

with open_prediction_run_artifact_writer("project.h5", "prediction_run_1") as writer:
    for row_start, y_chunk in chunks:
        writer.write_chunk(
            row_start=row_start,
            y_pred_chunk=y_chunk,
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
  for PLS and PCR models.
