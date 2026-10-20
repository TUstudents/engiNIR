# Contract coverage ledger

Status: active engineering ledger for engiNIR 1.0 and the v3 project schema.

Re-audit the mappings whenever persistence or contract behavior changes.

This document maps the authoritative v3 contracts to the implementation that
reads, writes, validates, tests, and documents each persisted object and its
invariants. It is deliberately evidence-based: a requirement is **covered**
only when a concrete enforcement path and a focused test both exist.

## Status legend

- **Covered** — implemented and exercised by a focused test.
- **Partial** — some enforcement or tests exist, but the complete contract is
  not proved.
- **Gap** — no adequate enforcement, test, or authoritative definition exists.
- **Intentional** — explicitly permitted behavior, with enforcement or tests.

Gap identifiers (`G01`, `G02`, ...) refer to the prioritized register at the
end of this document.

## Authoritative sources

- `contracts/project-schema-contract_v3.md`
- `contracts/dataset-calibration-contract_v3.md`
- `contracts/roi-target-contract_v3.md`
- `contracts/sample-set-model-run-contract_v3.md`
- `contracts/prediction-runtime-contract_v3.md`
- `contracts/project-dataset-attachment-contract_v3.md`
- `contracts/model-workspace-diagnostics-contract_v3.md`
- `contracts/viewer-workspace-ui-contract_v3.md`
- `contracts/headless-workflow-contract_v3.md`

Planning notes and design drafts are not evidence of current coverage; only
the contracts, the code and the tests are.

### Contract-to-ledger index

| Contract | Ledger evidence |
|---|---|
| Project schema | Persisted-object ledger; schema/identity and relationship invariant tables |
| Dataset/calibration | Dataset, calibration, signal-domain and prediction object rows; scientific-data table |
| ROI/target | ROI, provenance, calibration snapshot, target and assignment rows; relationship/scientific tables |
| Sample set/model run | Sample-set, model-run and preprocessing-state rows; modeling and artifact tables |
| Prediction runtime | Prediction object row; artifact/runtime/memory table; prediction workspace row |
| Project/dataset attachment | Workspace/attachment acceptance table |
| Model diagnostics | Model artifact row; diagnostics artifact and workspace acceptance rows |
| Viewer workspace UI | Workspace/attachment acceptance table, including job and navigation rows |

### Normative section audit

The table below records how each contract's normative sections were consumed.
Goals, problem statements, non-goals, and historical comparisons do not create
additional enforcement rows; open questions become gaps when they affect the
current implementation.

| Contract | Normative sections traced |
|---|---|
| Project schema | first-class objects and empty-project rule → object ledger; identity/relationships → schema and relationship tables; hydration/memory/serialization → validation-layer and artifact tables; schema validation/invariants → schema, relationship and scientific tables; update/encoding questions → `G01`, `G03`, `G08`, `G13` |
| Dataset/calibration | three domain objects and responsibility split → object ledger; import/calibrate/ROI/model/predict workflows → scientific/runtime tables; compatibility → signal compatibility row; persistence/UI/acceptance → object and workspace tables |
| ROI/target | ROI/provenance/target/assignment objects → object ledger; workflow/compatibility/scope → relationship and scientific tables; persistence/UI/acceptance → object and workspace tables |
| Sample set/model run | objects and frozen semantics → object/scientific tables; X/Y/groups/CV/reporting/weighting → modeling rows; blockers → sample/model service rows and `G09`; persistence/UI/acceptance → artifact and workspace rows |
| Prediction runtime | pipeline/preflight/predictor/signal-domain → runtime rows; chunking/memory/preprocessing/calibration/progress → runtime rows; artifact persistence/map handoff/acceptance → prediction object, artifact and workspace rows |
| Project attachment | viewer states/placeholder/create/attach/mismatch/mutation/ROI/save/API/error/tests → workspace/attachment rows and project/ROI object rows |
| Model diagnostics | layout/summary/metrics/plots/backend expectations → diagnostics workspace row and `G15`; persistence/selection/empty states/acceptance → model artifact and workspace rows |
| Viewer workspace UI | global frame/browser/modes/workspaces/plots/jobs/actions/preflight/redraw/navigation/acceptance → workspace/attachment table and `G15`, `G17` |

## Validation layers

The project currently has four distinct validation layers:

1. `enginir.io.project_store._check_project_file()` checks the schema version,
   required root groups, and root extras before decoding HDF5.
2. `enginir.io.project_store.load_project_store()` decodes the known project graph
   and supports lazy omission of heavy arrays. It does not by itself validate
   every nested object.
3. `enginir.project.hydrate_project_store()` checks required object fields,
   discards extras, builds typed objects, and calls
   `validate_project_relationships()`.
4. `ProjectState` and the modeling/prediction services enforce mutation-time
   and runtime scientific rules.

Consequently, “readable by `load_project_store()`” is weaker than “valid after
`hydrate_project_store()`”. This distinction is an open API/documentation gap
(`G01`).

## Persisted-object ledger

| Object and HDF5 identity | Reader / hydrator | Writer / mutation path | Validators | Focused tests | User/API documentation | Status |
|---|---|---|---|---|---|---|
| Project metadata `/project_meta` | `load_project_store`; project metadata selection in `hydrate_project_store` | `create_project`; `_open_project` updates `updated_at` | `_check_project_file` checks version/root groups; hydration requires four fields | `test_v3_project_store_round_trip_and_skip_flags`; wrong-version and missing-group tests in `test_hdf5_project.py` | `docs/hdf5_format.md`; `docs/workspace_domain_model.md`; `docs/user_guide/io.rst` | **Covered** (`G02` closed). |
| `DatasetRef` `/datasets/{dataset_id}` | `load_project_store`; `_hydration._build` | `save_dataset_ref`; `ProjectState.register_dataset` | `DatasetRef.__post_init__` validates path/format/modality, positive cube shape, band count, wavelength shape; relationship validator checks mapping identity | dataset round-trip; `test_dataset_ref_rejects_inconsistent_band_metadata`; prediction dataset persistence tests | dataset/calibration contract; `docs/hdf5_format.md`; `docs/user_guide/io.rst` | **Covered** (`G03`, `G07` closed). |
| `CalibrationSpec` `/datasets/{dataset_id}/calibrations/{calibration_id}` and dataset `active_calibration_id` | nested decode by `load_project_store`; `_hydration._build` | `save_calibration_spec`; `ProjectState.set_active_calibration` | `CalibrationSpec` validates modes, units, mutually consistent references/levels, finite values and matching array shapes; relationships validate calibration owner | `test_calibration_spec_contract_validation`; signal-domain matrix; ROI-local-reference modeling test | dataset/calibration contract; `docs/hdf5_format.md`; `docs/user_guide/io.rst` | **Covered** (`G03`, `G04`, `G05` closed). |
| `TrainingROI` `/rois/{roi_id}` metadata and `/spectra` | metadata via `load_project_store` with `_check_roi_arrays` shape/dtype scan; `_hydration._build`; spectra through `load_roi_spectra[_batch]` | `save_training_roi` prevalidates `n_pixels`/wavelengths against spectra dimensions; `ProjectState.add_roi`; `remove_roi_from_project` | `TrainingROI` validates metadata and required provenance; relationships validate dataset/calibration ownership; writer validates paired ROI-local references against the supplied spectra shape | ROI numeric/provenance tests; ROI round-trip/batch tests; EDA ROI creation tests; ROI-reference modeling test; `test_save_training_roi_rejects_metadata_spectra_mismatch`; `test_load_project_store_rejects_corrupt_roi_arrays` | ROI/target contract; `docs/hdf5_format.md`; viewer and I/O guides | **Covered** for array integrity (`G06` closed); direct writes remain untyped (`G03`). |
| ROI provenance `/rois/{roi_id}/provenance` | decoded with ROI; retained as a structured field by `_hydration._build` | generated by EDA extraction and written with ROI | required identity, signal unit, preview summary, and timestamp enforced by `TrainingROI` | `test_training_roi_requires_matching_complete_provenance`; EDA calibration-context tests | ROI/target contract; HDF5 format; preprocessing guide | **Covered** for required fields and owning-ROI identity. Extensible preview details are intentionally retained. |
| ROI-local calibration snapshots `/rois/{roi_id}/calibration_refs/{dark,white}` | lazy `iter_roi_calibration_refs`; `_check_roi_arrays` validates presence and shape of both snapshots during metadata-first open | `save_training_roi` optional dark/white arrays | writer requires both and exact spectra shape; modeling service requires snapshots when full-sensor references must be aligned | `test_project_state_fit_model_run_uses_roi_local_reference_snapshots`; HDF5 round-trip; `test_load_project_store_rejects_corrupt_roi_arrays` | ROI/target and dataset/calibration contracts; HDF5 format | **Covered** (`G06` closed). |
| `TargetDefinition` `/targets/{target_id}` | `load_project_store`; `_hydration._build` | `save_target_definition`; `ProjectState.define_target` | model validates ID/name, `value_type=float`, and missing policy | representative model/relationship tests; modeling-view target creation | ROI/target contract; HDF5 format; viewer/modeling guides | **Covered** (`G03`, `G07`, `G08` closed). |
| `TargetAssignment` `/target_assignments/{target_id}/{roi_id}` | `load_project_store`; `_hydration._build` | `save_target_assignment`; `ProjectState.assign_target_to_roi` | finite numeric value; non-empty IDs; relationship existence; duplicate active pair detection | relationship invalid-edge tests; sample-set tests; modeling-view finite-value tests | ROI/target contract; HDF5 format; viewer/modeling guides | **Covered** (`G03`, `G07` closed). |
| `PreprocessingRecipe` `/recipes/{recipe_id}` | `load_project_store`; `_hydration._build`; `PreprocessingRecipe.from_dict` | `save_preprocessing_recipe`; `ProjectState.register_recipe` | exact eight-field schema, strict persisted types and numeric constraints; extras discarded; content immutable by ID | `test_recipe.py` persisted-schema tests; `test_save_recipe_enforces_current_schema_and_discards_extras`; model/prediction lifecycle tests | project-schema contract; preprocessing guide/API; HDF5 format | **Covered** for current recipe schema, idempotence, and immutable identity. |
| `SignalDomainContract` `/signal_domains/{signal_domain_id}` | `load_project_store`; `_hydration._build` | `save_signal_domain_contract`; `ProjectState.register_signal_domain` | model validates IDs, enums, band/wavelength shape and tolerance; prediction compatibility validates dataset/calibration and output dtype | `test_signal_domain.py`; project type tests; prediction compatibility tests | dataset/calibration and prediction contracts; prediction API; I/O guide | **Covered** (`G03`, `G07` closed). |
| `SampleSet` `/sample_sets/{sample_set_id}` | `load_project_store`; `_hydration._build` | `save_sample_set`; `ProjectState.create_sample_set`; modeling `build_sample_set` | model freezes the seven v3 semantic constants; relationships resolve target, recipe, ROI and datasets; service builds labeled arrays and checks wavelength axes | `test_sample_set.py`; relationship tests; grouped model-run tests | sample/model contract; modeling API/guide; workspace model | **Covered** (`G03`, `G09` closed). |
| `ModelRun` `/model_runs/{model_run_id}` and `artifact_refs` | summary through `load_project_store`; lazy artifact loaders; `_hydration._build` | `save_model_run`; `ProjectState.register_model_run`; modeling `fit_model_run` | model validates IDs/model type; relationships resolve sample set/signal domain; fitting enforces grouped CV and ROI reporting; predictor artifact creation is backend-specific | `test_model_runs.py`; artifact lazy-load tests; reopened viewer shell test; v3 end-to-end test | sample/model and diagnostics contracts; modeling API/guide; HDF5 format | **Partial** — persisted metrics, CV spec, diagnostics and prediction-model artifact shapes are not schema-validated on open; IDs can be overwritten (`G10`, `G08`). |
| Fitted preprocessing state under model `prediction_model/preprocessing_state` | lazy prediction-model artifact loader; prediction preflight restoration | created by `fit_model_run` from fitted pipeline | MSC/EMSC state shape/content checked during restoration; missing state blocks before mutation | MSC reopened equivalence, missing-state atomicity, and combined MSC/EMSC restoration tests | sample/model and prediction contracts; modeling/prediction/preprocessing guides | **Covered** for current MSC/EMSC recipe parameters. |
| `PredictionRun` `/predictions/{prediction_run_id}` and disk-backed artifacts | summary `load_project_store`; `_hydration._build`; lazy artifact and preview loaders | `save_prediction_run`; staged initialize/write/finalize/discard service | model/relationships validate IDs; incomplete `writing` state is rejected; runtime validates compatibility, chunking and predictor replay | extensive `test_prediction_runs.py`; incremental HDF5 artifact test; prediction-view map/metadata tests | prediction runtime contract; prediction API; I/O/viewer guides; HDF5 format | **Partial** — cross-chain equality with the model’s recipe/signal domain is not checked; completion does not require the full metadata/output-axis contract; direct non-staged writer can overwrite IDs (`G11`, `G12`, `G08`). |

## Cross-cutting invariant ledger

### Schema, identity, and compatibility

| Invariant | Enforcement evidence | Test evidence | Status |
|---|---|---|---|
| Current `viewer-v3` version and every required root group | `_check_project_file` | wrong-version and missing-group tests | **Covered** |
| Unknown fields warn and do not become authoritative object fields | root scan plus hydration `_known_fields`; structured fields remain explicit exceptions | root `/models`, root/ROI extras, and nested dataset-extra tests | **Covered** at root and hydrated object boundaries. Raw `load_project_store()` still exposes nested extras until hydration (`G01`). |
| No aliases, inference, migration, or legacy-field promotion | strict hydration required fields; current recipe parser; legacy extras ignored | missing `source_modality`, recipe missing-field, `y_value`, `/models` tests | **Covered** for tested shapes; contract field tables reconciled against typed models and `docs/hdf5_format.md` (`G13` closed). |
| Mapping key equals stable object ID | `validate_project_relationships._validate_mapping_keys` | representative invalid-edge tests | **Covered** for typed objects. Recipe IDs are path-only by contract. |
| IDs are unique within their namespace | HDF5 path uniqueness and relationship map keys; hydration rejects calibration-id collisions across dataset subtrees | indirect round-trip evidence; `test_hydration_rejects_calibration_id_collision_across_datasets` | **Covered** — calibration ids are contract-frozen as project-globally unique (`G05` closed). |
| Stable IDs cannot silently change scientific content | `_require_new_or_unchanged` in the direct writers for datasets, ROIs, targets, signal domains, sample sets, model runs and one-shot predictions; recipe writer's own content check; calibration/assignment/staging exceptions are contract-documented | recipe immutability test; `test_stable_object_ids_are_immutable_except_labels`; updated failed-serialization test | **Covered** (`G08` closed). |
| Current writes emit only current fields | typed `ProjectState` paths; strict recipe writer | recipe-extra discard test | **Covered** (`G03` closed). |
| Failed object serialization preserves the previous published group | `_atomic_replacement_group` stages under a temporary root group and publishes after serialization | `test_failed_object_serialization_preserves_previous_group` | **Covered** |
| Project timestamps are updated after successful mutations | `_touch_project_updated_at` in writer paths | project round-trip gives indirect evidence | **Covered** (`G14` closed). |

### Relationship invariants

| Invariant | Enforcement evidence | Test evidence | Status |
|---|---|---|---|
| Calibration belongs to an existing dataset | relationship validator | invalid-edge matrix | **Covered** |
| Dataset active calibration exists and belongs to that dataset | mutation-time `ProjectState` behavior plus hydration-time `validate_project_relationships(active_calibration_ids=...)` | active calibration round-trip; `test_validate_project_relationships_checks_active_calibration_ids`; `test_hydration_rejects_dangling_or_foreign_active_calibration_id` | **Covered** (`G04` closed). |
| ROI dataset and calibration exist and agree | relationship validator | invalid-edge matrix | **Covered** |
| Target assignment references one existing ROI and target; active pair unique | relationship validator | invalid-edge matrix | **Covered** |
| Sample set references existing target, recipe, datasets and ROIs | relationship validator, with explicit stale-ROI allowance; `dataset_ids` must cover every live included ROI's dataset | valid/invalid relationship tests, stale-ROI test, and `test_validate_project_relationships_rejects_sample_set_omitting_roi_dataset` | **Covered**, including intentional stale saved-model behavior and the dataset-coverage invariant. Scientific membership completeness is separate (`G09`). |
| Model run references existing sample set and signal domain | relationship validator | invalid-edge tests and model lifecycle | **Covered** |
| Prediction references existing dataset, calibration, model, signal domain and recipe | relationship validator | representative graph tests and prediction lifecycle | **Partial** — it does not require prediction recipe/signal-domain IDs to equal the referenced model/sample-set chain (`G11`). |
| Deleting an ROI preserves saved models but marks affected sample sets stale | `ProjectState.remove_roi` and stale checks | project-state and EDA deletion tests | **Covered** as intentional policy. |

### Scientific data and modeling invariants

| Invariant | Enforcement evidence | Test evidence | Status |
|---|---|---|---|
| Dataset cube bands, `n_bands`, and wavelengths agree | `DatasetRef` | dataset metadata tests | **Covered** after hydration. Direct-write gap is `G03`. |
| Calibration modes have executable, internally consistent reference context | `CalibrationSpec` | calibration contract matrix | **Covered** after hydration. Dataset-dimension compatibility of full-sensor references remains a runtime concern by design (checked at fit/prediction time). |
| ROI raw spectra are scientific truth and remain lazy | load-store skip behavior; lazy ROI loaders; shape/dtype integrity scan uses dataset metadata only | HDF5 round-trip/batch and project-state lazy tests; corrupt-ROI-array test | **Covered** for laziness and stored shape integrity (`G06` closed). |
| ROI provenance matches owning dataset/calibration and valid signal unit | `TrainingROI` | provenance contract test | **Covered** |
| Target values are finite floats | `TargetAssignment`; GUI dialogs | project type and modeling-view tests | **Covered** |
| Recipe has exact executable fields and deterministic order | `PreprocessingRecipe`; modeling pipeline | recipe and pipeline tests | **Covered** |
| Sample-set semantics are pixel observations, ROI target/group/CV/reporting, per-pixel unweighted, ROI mean | `SampleSet`, modeling constants, `_validate_v3_sample_set`, grouped CV/aggregation | sample-set constant and aggregated-metric tests | **Covered** for service-created/fitted sets; stored membership completeness remains `G09`. |
| No ROI leakage across CV folds | grouped `cross_validate` path | grouped model-run tests and validation suite | **Covered** |
| Fold assignment follows the sample set's ROI order, not the random ids | `fit_model_run` labels each ROI with its position in `roi_ids` | `test_cross_validation_does_not_depend_on_the_random_roi_ids` | **Covered** |
| Primary metrics are ROI-aggregated; pixel metrics secondary | `fit_model_run` metric construction | aggregated-primary-metrics test | **Covered** |
| Stateful MSC/EMSC uses persisted training state, never prediction-data fit | fitted-state artifact extraction/restoration | reopened equivalence and missing-state tests | **Covered** |
| Signal compatibility uses format/modality/bands/calibration/dtype, not wavelength-value equality | prediction compatibility functions | `test_signal_domain.py` and prediction tests | **Covered** |

### Artifact, runtime, and memory invariants

| Invariant | Enforcement evidence | Test evidence | Status |
|---|---|---|---|
| Model/prediction heavy arrays remain lazy during navigation | metadata-only group reads and key loaders | project-state, HDF5, viewer-shell lazy tests | **Covered** |
| Persisted model artifact can reproduce prediction without refit | backend artifact builders/loaders; saved-artifact prediction | reopened prediction tests | **Partial** — covered end to end for PLS and current recipes; PCR has model/serializer evidence but lacks an equally direct reopened prediction lifecycle test (`G10`). |
| Diagnostics reopen without refit | persisted CV outputs/diagnostics and workspace payload loading | model-run persistence and reopened viewer-shell tests | **Partial** — no focused schema validator or complete PLS/PCR diagnostics acceptance matrix (`G10`, `G15`). |
| Prediction uses bounded chunks and avoids full float cube copies | single-handle chunk iterator and fixed 512-frame default | chunk-size, uint16, handle-reuse, progress, no-return-array and full-sensor-reference tests | **Covered** for current HDF5 path. Contract permits a temporary full-cube adapter (`G16`). |
| Prediction progress exposes phases and counters | progress events in prediction service | progress/metadata test and prediction-view cancellation test | **Covered** |
| Prediction artifacts initialize, stream, finalize, and discard on failure/cancel | staged artifact API and service `try`/cleanup | incremental, cancellation, and disk-backed tests | **Covered** |
| Completed prediction metadata interprets shape, dtype, spatial orientation and outputs | artifact metadata builder and prediction view | summary-without-map, map orientation, multi-output selector tests | **Partial** — finalizer/open validator does not require all conditionally mandatory metadata (`G12`). |
| Multi-output predictions require explicit output-axis metadata | runtime artifact builder and UI selector | multi-output widget and artifact assertions | **Partial** — malformed persisted completed runs are not rejected (`G12`). |
| Incomplete prediction runs do not hydrate as completed | `_prediction_run_artifacts_complete` rejects `writing` and unknown status values; absent status means complete | cancellation/discard, project reopen, and `test_prediction_run_artifact_write_status_lifecycle_enum` tests | **Covered** — status enum is contract-defined (`writing`/`complete`, absent = complete, unknown rejected). Metadata completeness remains `G12`. |

## Workspace, attachment, and diagnostics acceptance ledger

These requirements consume the persisted graph and therefore belong in the
coverage trace even though they are not HDF5 object schemas.

| Contract area | Implementation evidence | Test evidence | Status |
|---|---|---|---|
| Dataset-only, project-only, and attached viewer modes | viewer startup/open/attach paths and explicit dataset context | `test_project_open.py`, welcome/app tests | **Covered** |
| Project-only mode uses no fake cube/dataset | project-only startup and placeholder canvas | project-open and viewer-shell tests | **Covered** |
| Attach mismatch is classified and user-gated | viewer attach validation | mismatch warning-gate tests | **Covered** |
| Dataset-derived mutations require real attached context | `ProjectState`/EDA guards | ROI creation guard and project-open tests | **Covered** |
| Project browser uses stable IDs and coordinates selection | browser/inspector/viewer coordination | browser-selection viewer-shell test | **Covered** |
| Model workspace renders persisted ROI predictions/residuals and spectral diagnostics | `_model_diagnostics.py`, modeling payload loaders | reopened viewer-shell test plus artifact tests | **Covered** (`G15` closed). |
| Model fit summary and required metric labels are present | modeling workspace UI | modeling-view fit tests and viewer-shell rendering | **Covered** (`G15` closed). |
| Prediction workspace shows compatibility/provenance and lazy persisted maps | prediction workspace and map widget | prediction-view compatibility, summary, cache, orientation and selector tests | **Covered** for current outputs. |
| Long-running jobs expose busy/cancel/error states and avoid stale callbacks | background job helpers and workspace guards | modeling/prediction view tests | **Covered** (`G17` closed). |
| Redraw/navigation preserves workspace state and coalesces refreshes | viewer refresh scheduling and workspace widgets | refresh coalescing and control-height tests | **Covered** (`G17` closed). |
| Headless packages stay Qt-free; workflow services usable without a QApplication | Qt imports confined to `enginir.viewer`; io/project/modeling/prediction services take plain callables | `test_headless_api.py` import probe; service-level model-run and prediction tests run without `qapp` | **Covered**; the end-to-end headless workflow runs in `test_workflow_fits_predicts_and_reopens_identically` and the documented example in `test_docs_examples.py`. |
| Project facade: names unique per kind, ambiguous lookups fail | `enginir._workflow.Project` name checks and lookups | `test_names_are_unique_and_ambiguous_lookups_fail` | **Covered** |
| Project facade: derived dataset/calibration (reuse, DN, precalibrated, references, EVK levels) | `Project.add_dataset`, `_calibration` | `test_a_registered_file_is_reused_and_a_changed_one_refused`, `test_calibration_follows_the_data_and_unit`, `test_reference_spectra_calibrate_and_travel_with_each_roi` | **Covered** |
| Project facade: sample set, recipe and signal domain reused; one signal domain per model | `Project.fit` | `test_refitting_reuses_sample_set_recipe_and_signal_domain`, `test_one_model_refuses_rois_recorded_in_different_signals` | **Covered** |
| Project facade: failed calls write nothing; prediction checked before registering the file | validate-then-write order in `add_roi`, `fit`, `Model.predict` | `test_failed_calls_write_nothing`, `test_prediction_is_refused_before_an_incompatible_file_is_registered` | **Covered** |
| Project facade: `ProjectState` delegates to `Project` and holds no rule of its own | `ProjectState` writes through `Project.save_*`; Explore and Predict use `resolve_dataset`, `make_calibration`, `resolve_calibration`, `resolve_prediction_calibration`, `make_roi` | `test_viewer_writes_projects_only_through_project` (AST boundary check); the viewer suites run through the delegation | **Covered** |

## Explicit gap register

Priority reflects scientific corruption risk first, then maintainability and UI
completeness.

| ID | Priority | Gap | Required closure evidence |
|---|---:|---|---|
| `G01` | P1 | **Closed.** `load_project_store()` is documented as the raw, structurally checked reader; `hydrate_project_store()` is the validating step and the only one that guarantees a contract-conforming workspace. | `test_raw_load_is_structural_and_hydration_validates` (raw load accepts a malformed timestamp that hydration rejects). |
| `G02` | P1 | **Closed.** Hydration validates `project_meta`: non-empty `project_id`, `schema_version` equal to the current version, ISO 8601 `created_at`/`updated_at` (`validate_project_meta`). | `test_hydration_rejects_malformed_project_meta`. |
| `G03` | P1 | **Closed without an API change.** The writers keep taking dicts, but every object payload is validated before it is written, by the same typed-model construction hydration uses (`validate_object_payload`); unknown fields are rejected at write time instead of being stored and ignored later. | `test_writers_refuse_payloads_that_hydration_would_reject`, `test_child_object_writers_validate_too`. |
| `G04` | P0 | **Closed.** `active_calibration_id` is relationship-validated on project open: it must exist and belong to its dataset. | Hydration relationship check plus missing/wrong-owner/missing-dataset tests. |
| `G05` | P0 | **Closed.** Calibration ids are contract-frozen as unique across the whole project; hydration rejects cross-dataset collisions instead of silently overwriting. | Collision corruption test plus the identity rule in `project-schema-contract_v3.md`. |
| `G06` | P0 | **Closed.** Metadata-first open now scans stored ROI spectra/wavelengths/calibration-reference shapes and dtype via HDF5 dataset metadata; `save_training_roi` prevalidates declared `n_pixels`/wavelengths against the supplied spectra. | Shape/dtype scan, writer prevalidation, and corruption tests landed. |
| `G07` | P1 | **Closed.** Every object timestamp (`import_timestamp`, `created_at`, `assigned_at`) must be ISO 8601 (`_timestamp`); text fields use `_text`, booleans `_flag` (rejects non-bool), integers `check_integer`, arrays numeric dtype checks. | `test_object_timestamps_must_be_iso_8601`, `test_hydration_rejects_mistyped_values_and_names_the_location`. |
| `G08` | P0 | **Closed.** Direct writers reject overwriting a stable id with changed content (labels excepted, idempotent re-saves allowed); calibration replacement, assignment updates, and prediction staging are the documented exceptions, codified in `project-schema-contract_v3.md` Object Identity Rules §5. | Immutability policy, writer enforcement, and overwrite tests landed. |
| `G09` | P1 | **Closed.** Policy: frozen-file invariants are enforced by `SampleSet` on construction and hydration (non-empty, unique `roi_ids`; non-empty `dataset_ids`) and by relationship validation (`dataset_ids` cover every live ROI); unlabeled members and incompatible wavelength axes are build/refit-time blockers in `build_sample_set` and `fit_model_run`; deleted ROIs remain allowed under the stale-sample-set policy. | `test_sample_set_rejects_empty_or_repeated_membership`, `test_validate_project_relationships_rejects_sample_set_omitting_roi_dataset`, `test_build_sample_set_needs_at_least_one_labelled_roi`, `test_build_sample_set_rejects_incompatible_wavelength_axes`, `test_fit_refuses_a_stored_sample_set_whose_roi_lost_its_target_value`. |
| `G10` | P3 | **Deprioritized — validate-on-use policy.** Persisted metrics/CV/diagnostics artifacts are written by one service and validated where they are consumed (prediction preflight restores and checks preprocessing state; diagnostics loaders degrade explicitly). An open-time schema validator would re-check our own writes on every open for corruption modes h5py mostly surfaces anyway. Revisit only if real-world artifact corruption is observed. | Only if evidence of real corruption appears. |
| `G11` | P3 | **Deprioritized — validate-on-use policy.** Prediction recipe/signal-domain chain consistency is enforced where it matters: prediction executes the model run's own recipe and signal domain resolved through the sample-set chain, so a divergent persisted id cannot change what runs. A cross-chain open-time check (~one comparison) may be added opportunistically. | Optional one-line relationship check if ever needed. |
| `G12` | P3 | **Deprioritized — validate-on-use policy.** The status enum is contract-defined and enforced (`writing`/`complete`, unknown rejected); the prediction view already degrades to summary-without-map when metadata is incomplete. Exhaustive completed-run metadata validation on open is not worth its maintenance surface. | Only if evidence of real corruption appears. |
| `G13` | P1 | **Closed.** Required-field lists differed between contracts (dataset/calibration contract versus current hydration fields). The dataset, calibration, and signal-domain field tables now match the typed models and `docs/hdf5_format.md`; conditional calibration fields are marked as declared-nullable; the fictional `/predictions/.../runtime/` subgroup was removed from the project-schema layout; `project-schema-contract_v3.md` plus `docs/hdf5_format.md` are declared the canonical layout owners. ROI/target, sample-set, and model-run tables were audited and already matched. | Reconciled at this ledger revision; re-audit on any contract field change. |
| `G14` | P2 | **Closed.** `updated_at` is set only after a successful write; a failed write leaves it unchanged. | `test_updated_at_changes_only_after_a_successful_write` (clock-controlled). |
| `G15` | P2 | **Closed.** A saved PLS and a saved PCR run reopen with every diagnostic drawn: measured vs predicted, residuals, target distribution, scores, coefficients and loadings on the wavelength axis, with clickable ROI points and no leftover empty state. Empty states are covered separately. | `test_reopened_model_run_renders_every_diagnostic_and_selects_rois[PLS/PCR]`, `test_model_diagnostics_show_empty_states_in_the_plots`. |
| `G16` | P2 | **Closed.** Every image format streams into prediction in bounded chunks: EIFF and HDF5 with any frame selection (stepped and reversed), ENVI through a memory map (`iter_envi_frame_chunks`); only point-spectrum formats (CSV, SPC, SPA, JCAMP) are read completely. The temporary full-cube adapter was removed from `prediction-runtime-contract_v3.md`. | `test_image_files_stream_into_the_same_prediction_as_in_memory_data` (fails if an image is read completely or a chunk is misplaced). |
| `G17` | P2 | **Closed.** Switching tabs keeps project, cursor, target, active model and object counts; hidden Model diagnostics never redraw; an unchanged payload is a no-op (this test found and fixed CV outputs being re-read from HDF5, and the figure redrawn, on every refresh); jobs deliver results and errors on the UI thread, exactly once, then finish and release themselves. | `test_workspace_navigation_preserves_context_and_redraws_only_the_visible_workspace`, `tests/test_jobs.py`, existing cancel/failure view tests. |
| `G18` | P2 | **Decision pending (after 1.0).** Prediction does not compare wavelength axes: a scan with the matching band count but a shifted wavelength range is predicted without warning. The check was removed on purpose because not all spectra carry wavelength information (loaders return an empty axis, e.g. ENVI without `wavelength`, some CSV/SPC files), and the prediction-runtime and dataset-calibration contracts forbid wavelength values as a gate. Options: a logged warning when both axes exist and differ by more than `wavelength_tolerance_nm` (already persisted, currently 0 and without effect), or a refusal under the same condition, skipped when either axis is missing. | Contract amendment in both contracts, then a test for differing, matching and missing axes. |

### Contract findings closed by the 1.0 code review

The module-by-module review before 1.0 found and closed these contract
violations, which were not in the register. Each has a test that fails when
the rule is removed.

| Contract | Finding | Fix and evidence |
|---|---|---|
| Headless workflow: no rule only in the viewer | Calibration content deduplication existed only in `ProjectState`; Explore used a second, tolerant definition. | `CalibrationSpec.same_content_as` used by both; `test_calibration_content_identity_ignores_ids_and_timestamps`. |
| Headless workflow; dataset-calibration writer rules | Signal-domain derivation from training data existed only in the Model workspace. | `SignalDomainContract.for_training_data`; `test_signal_domain_for_training_data_follows_the_contract_writer_rules`. |
| ROI/target: floats never truncated | The viewer cast float cubes without a `calibrated` attribute to uint16 (reflectance displayed and stored as zeros); its preview converted differently from training. | dtype decides precalibrated vs raw, preview matches `transform_spectra_to_signal_domain`; `test_float_cubes_are_precalibrated_and_never_truncated`. |
| Project schema §5: idempotent re-saves | An unchanged object holding NaN was refused as a content change. | `equal_nan` comparison; NaN case in `test_resaving_unchanged_objects_is_a_no_op`. |
| Prediction runtime: output-axis and clipping metadata | Runs predicted with an in-memory predictor were stored without `output_names`; persisted clip counts and multi-output axis metadata were untested. | Names read from the saved model; `test_persisted_run_records_clip_counts_output_names_and_a_1d_single_output`, `test_multi_output_predictions_store_their_output_axis`. |
| Prediction runtime: memory budget | Two float32 working chunks were alive while predicting. | One working chunk; measured 2.00 -> 1.00 chunks. |
| Interactive map handoff: preview reads | The preview of the canonical `y_pred`-only layout used h5py point selection (minutes for a full scan). | Block reads; `test_map_preview_subsamples_the_canonical_y_pred_layout`. |
| Sample-set/model-run: MSC/EMSC fitted inside folds | No test caught MSC/EMSC being fitted on all pixels before cross-validation. | `test_fit_model_run_fits_msc_inside_every_cross_validation_fold`. |
| Sample-set/model-run: reproducible validation | `GroupKFold` received the random ROI ids as labels, so which ROIs shared a fold, and the RMSECV and R² of a repeated analysis, changed with every new project (observed R² from 0.87 to 0.94 for the documented example). | ROI positions as labels; `test_cross_validation_does_not_depend_on_the_random_roi_ids`; the documented example output is checked verbatim. |
| Model diagnostics: explicit placeholder states | Empty-state messages were written to widgets never added to a layout. | Messages drawn into the plots; `test_model_diagnostics_show_empty_states_in_the_plots`. |

## Recommended closure order

1. `G04`, `G05`, `G06`, `G08`, `G13`: **done** — identity/calibration/ROI
   corruption guards and field-table reconciliation.
2. `G10`, `G11`, `G12`: **deprioritized** under the validate-on-use policy;
   revisit only on evidence of real artifact corruption.
3. `G01`, `G02`, `G07`, `G09`, `G14`: **done**.
   `G03`: **done** (validated dict writers, no API change).
4. `G15`, `G16`, `G17`: **done**.

## Verification gates for ledger updates

Any change that marks a row **Covered** must include:

- a direct enforcement path, not only a GUI precondition;
- a focused positive test and at least one malformed/corruption test;
- log-before-raise behavior for persisted telemetry;
- updated contract and user/API documentation;
- `uv run pytest`, Ruff, Pyright, and Sphinx warning-as-error builds;
- local SCIREF/Kestreleye smoke checks when the affected path reads project or
  spectral files, without hardcoding those files into repository tests.
