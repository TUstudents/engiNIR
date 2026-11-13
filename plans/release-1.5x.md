# Plan: engiNIR 1.5.x, a minimum project format and bug fixes

Status: decided (2026-11-12): direction, decisions 1–11 and format
decisions F2, F3 and F5 confirmed by the user. Branch `1.5x`; Phase 1 done
(d406d04, b9974a4); Phase 2 done (c38e5ef, 88eface); Phase 3 done (7d41ee5,
06c5cf6, c8254d6, f038094); Phase 4 done (6794057); Phase 5 done (8e87228);
merged to main as PR #11. Checkpoint done (2026-11-13), its fixes on branch
`fix/1.5-open-and-create`; Phase 5b (data files across platforms, decided
2026-11-13) done on the same branch; merged as PR #12. Released as 1.5.1
(user, 2026-11-13) from main, with the kernel engine (PR #13); Phase 6 (GUI
polish) moves to the next release.

## Goal

engiNIR 1.5.x settles on one minimum project file format: every field the
current writers produce is required, and nothing is read "as if" from older
files. On that clean base it fixes the bugs of the GUI review of
2026-11-11. The user then builds a new 1.5.x demo project. The GUI polish
follows later on the same branch, with its own plan.

## Decisions (user, 2026-11-12)

1. **No backward compatibility.** Project files written before the 1.5.x
   format are refused, with a clear message. No migration and no repair of
   old projects. Code and contract rules that exist only to read or refit
   older files are removed.
2. **A minimum 1.5 project format,** settled in this plan before any code
   changes.
3. **Branch `1.5x`** (renamed from `terminology/followup`). One commit per
   phase and one draft PR, as for 1.5.0.
4. **Order:** format and bug fixes first. Then the user builds a new 1.5.x
   demo project. Then the GUI polish joins 1.5.x (plan to follow; it was
   1.6.0 in the 1.5.0 plan).
5. **GroupKFold is an optional fold rule.** `group_kfold` stays as a
   regular rule a user can request (with `group_by = roi`), labeled by ROI
   position as today. It is no longer described as a legacy rule. The
   default stays round-robin, and the fit dialog keeps not offering it.
6. **`scale` and `eiff_raw` are removed, and the code refactored.** The
   `scale` parameter goes from `PLSRegression` and `PCR` (API and stored
   hyperparameters), with PCR's StandardScaler path; scaling is
   `x_scaling` and `y_scaling` only. The `eiff_raw` alias goes from the
   format comparison and the prediction service.
7. **Stored predictions stay optional.** `/predictions` holds runs only when a
   script stores a prediction with the low-level writers (the group itself
   always exists, empty otherwise); the facade and
   the viewer still never write it. Its crash safety is F5.
8. **File metadata** in `/project_meta`, replacing `schema_version` (this
   replaces the earlier proposal F1):

   ```text
   /project_meta
     @project_id          uuid                              set once
     @FileSchemaVersion   "enginir-1.5"                     set once
     @FileCreator         "engiNIR 1.5.1"                   set once
     @FileWriter          "engiNIR 1.5.2"                   set by every write
     @created_at          2026-11-12T09:56:05.852000+00:00  set once
     @updated_at          2026-11-14T16:03:11.204000+00:00  set by every write
   ```

   - `FileSchemaVersion` names the format, not the software. The reader
     requires an exact match. It refuses every other value, and every file
     without it, up front with one message ("this project was written by
     an earlier engiNIR in format …; engiNIR 1.5 reads only enginir-1.5
     projects; create the project again"). It changes only when a field
     becomes required, is removed or changes meaning. A new optional field
     does not change it: the extra-field policy covers that. Later
     releases keep writing `enginir-1.5` while the format is the same.
   - `FileCreator`: the software that created the file. `FileWriter`: the
     software that last wrote it. Both are `"engiNIR <version>"` with the
     version from the package metadata (`enginir.__version__`), never
     typed in. A development build is written as it is (e.g.
     `1.5.1.dev3+g1234abc`). They are informational: shown in the
     Inspector and in error messages, and never gate reading.
   - `FileWriter` and `updated_at` are set together, in the store's one
     after-write step. A failed write changes neither, and reading changes
     nothing.
   - Timestamps (`created_at`, `updated_at` and every object's
     `created_at` and `assigned_at`) are ISO 8601 in UTC with an explicit
     offset (`+00:00`) and microseconds. The reader requires a valid,
     timezone-aware timestamp; today it only checks that it parses. The
     viewer shows local time.
   - The three new names are CamelCase by the user's choice; every other
     stored name stays snake_case.
9. **The ROI field `use` is renamed `job`,** with the same values,
   `calibration` and `prediction`. The API follows, so code and file speak
   alike:
   - `Dataset.add_roi(..., job=...)` and `add_rois`
   - `Project.set_job(roi, job)` and `Project.save_roi_job(roi_id, job)`
   - `TrainingROI.job`
   - the writer `save_roi_job`.
   The viewer's column stays "Set" (Calibration or Prediction), and the
   "Prediction only" checkbox stays. The terminology contract maps `job`
   to the ROI set.
10. **Every model records the software that fitted it**, in a group
    `/model_runs/{id}/software`. A model's numbers depend on these
    versions, so a refit whose metrics differ can name the likely cause
    ("fitted with scikit-learn 1.7.2, refitted with 1.8.0").

    ```text
    /model_runs/{id}/software
      @enginir       "1.5.1"
      @numpy         "2.3.4"
      @scipy         "1.16.2"
      @scikit-learn  "1.7.2"
    ```

    - It is written by every fit and refit, is required in the 1.5
      format, and is part of the model's content: a re-save with different
      versions is changed content. A refit is a new model, so it gets the
      new versions.
    - The versions come from the package metadata of the running
      environment.
    - A refit compares the stored versions with the running ones. When its
      metrics differ from the stored model's, the refit warning lists the
      versions that differ.
    - The Inspector shows the versions under "how the model was fitted".
    - Not in 1.5.x: a write history (who changed which object and when),
      which belongs with the 2.0 method file.
11. **A strict core and tolerant display data** (after the Phase 2 review,
    2026-11-12): do not overspecify, and keep the viewer robust.
    - The reader refuses a file only when a core value is missing or
      malformed: the file metadata, ids and references, datasets and
      reference corrections, ROI spectra, wavelengths, references and `job`, targets
      and target values, every recipe field, input contracts, calibration
      sets, a model's `model_type`, `hyperparameters`, `fit_spec`, `software` and
      executable `prediction_model`, and a stored prediction's references,
      `y_pred`, shapes and dtype.
    - Display data (metrics, the component curve, calibration diagnostics,
      `cv_spec`, `cv_outputs`, prediction telemetry) never stops a project
      from opening; the viewer shows "–" for a missing or undefined value
      and never fails on it.
    - Writers write every field. A value not defined in a case is NaN
      (numbers), an empty string or an empty array, instead of a list of
      conditional fields; `None` is written the same way.
    - Smaller points of the review: the reader does not check that recipe
      fields without effect hold their default; an ROI `@y_value` is an
      ignored extra; a staged write refuses an id already in `/predictions`
      or `/staging`; only `discard_staged_predictions` deletes stale
      staging groups, and that is a write (second review).

## The minimum 1.5 project format

### What stays

The v3 object model and layout stay as they are: datasets with their
reference correction, ROIs, targets and target values, recipes, input
contracts, calibration sets and models, linked by stable ids, with eager
writes and content immutability. The extra-field policy also stays: an
unknown extra field warns and is ignored. That policy is tolerance for
additions, not compatibility with older files.

### What is required (no "absent reads …" rule)

| Object | Required in 1.5 | Today's compatibility rule that goes |
|---|---|---|
| Project | `FileSchemaVersion = "enginir-1.5"`, `FileCreator`, `FileWriter`, timezone-aware timestamps (8) | `schema_version = "viewer-v3"` accepted for every version since 1.0 |
| Dataset | `source_modality` non-empty, `source_format` a current key | `eiff_raw` read as `eiff` (removed, 6) |
| ROI | `job` = `calibration` or `prediction` (9) | `use`; a missing `use` and `training` read as `calibration`; the identity rule treating them as the same content |
| Recipe | every field written, including `sg_edges`, the AsLS fields and the band ranges (F2) | absent `sg_edges` reads `interp`, absent `asls` reads false, absent band ranges select every band |
| Model, `hyperparameters` | `x_scaling`, and `y_scaling` for PLS | `scale` (runs before 1.2), and the StandardScaler path of PCR (removed from the API too, 6) |
| Model, `fit_spec` | every key, including `components/noise_guard` and `components/rule_choice` | legacy runs without `fit_spec` (§7.3: GroupKFold by position, no curve); a `fit_spec` without `components` (1.2); absent `noise_guard` reads `off`; absent `rule_choice` reads `selected` |
| Model, `cv_spec` | `@warnings` (empty string when none) | absent reads "not recorded" |
| Model, `software` | `enginir`, `numpy`, `scipy`, `scikit-learn` versions (10) | none (new) |
| Model, artifacts | the component curve with `rmsee_roi`, `dw_weights`, `noise`, `@structure_ceiling`; the calibration diagnostics (T², DModX, limits, `roi_scores`, CoeffCS, shares above the limits); band importance (`mean_spectrum`, VIP for PLS, selectivity ratio); projection data in `prediction_model`; the coverage metrics and RMSEE where defined | runs before 1.3 and 1.4 without them; "refit" placeholders in the viewer; VIP computed from stored weights for runs before 1.4; prediction without diagnostics for models without projection data |
| Fold rules | `round_robin` (default), `leave_one_group_out`, `group_kfold` (optional, with `group_by = roi`; 5) | `group_kfold` described as "kept only so legacy runs refit" |
| Input contract | `band_mapping_policy = exact` (F3) | `reject` accepted, with the same meaning |
| Predictions | optional: `/predictions` only when a script stores one (7); staged writes as F5 | the legacy dual-array `spatial_map`; the `writing` status in `/predictions` (F5) |

### Format decisions (confirmed 2026-11-12)

- **F1.** Decided as 8. Today every file since 1.0 says `viewer-v3`, so
  an old file fails on whichever field is missing first (as with the empty
  modality in the Keile and Tablets projects). `FileSchemaVersion` lets the
  reader refuse it up front. The contract files keep their `_v3` names
  (internal names), and their status lines say 1.5.
- **F2. Recipes write every field** (confirmed). The reader then has no defaults to
  fill in, and "the same content" is a plain comparison. Recipe ids change,
  which is acceptable without compatibility.
- **F3. `band_mapping_policy` keeps one value, `exact`** (confirmed). The field stays,
  reserved for calibration transfer (2.0).
- **F4.** Decided as 7: stored predictions stay optional.
- **F5. Crash-safe staging of stored predictions** (confirmed). Today a staged write
  creates `/predictions/{id}` marked `writing`. The reader must refuse a
  project with such a run, so a crash or power loss during a script's
  prediction makes the whole project unopenable (as happened with
  `Tablets_10%/demo.h5`).
  - Stage under `/staging/{id}`, a group the format declares as
    "may hold incomplete writes; ignored by the reader". Finalizing moves
    the group to `/predictions/{id}`, a rename inside the file, so
    `/predictions` only ever holds complete runs.
  - A crash leaves a stale staging group. The project still opens, the
    reader logs one warning, and `discard_staged_predictions(path)` removes
    it.
  - `artifact_write_status` goes: a run in `/predictions` is complete by
    construction.

### What the user sees

- Opening any older project gives one message naming its version and what
  to do. The welcome screen keeps it in the recent list (it is the user's
  file).
- Every model shows every diagnostic. No "refit" placeholders, no "not
  recorded".
- The documentation (`docs/hdf5_format.md`, user guide) describes one
  format, without "runs before 1.x" notes.

## Phases

### Phase 1: terminology follow-up (done)

d406d04 and the kernel plan b9974a4. See the CHANGELOG's "Unreleased"
section.

### Phase 2: the minimum format in the contracts

The table above, applied to the contracts, with an independent review.
Every "before 1.x", "legacy" and "absent reads" rule is removed or turned
into a requirement:

- schema: version string, compatibility policy (refuse other versions,
  no migration, no repair), `/predictions` optional (7) and the staging
  group with §7's lifecycle replaced (F5), `/project_meta` with
  `FileSchemaVersion`, `FileCreator`, `FileWriter` and the timestamp rule
  (8), the model's `software` group (10), §5's `use` exception becoming
  `job`, changed only by `save_roi_job` (9)
- dataset reference correction contract: `eiff_raw`, `reject`
- ROI target contract: `job` required, renamed from `use` (9)
- calibration set and model contract: §7.3 removed, §7 required keys, the
  "Runs stored before …" paragraphs, `scale` (6); `group_kfold` described
  as an optional rule (5)
- prediction runtime: low-level persistence kept, its lifecycle section
  rewritten for the staging group (F5); diagnostics always present
- model diagnostics and viewer UI contracts: placeholders for old runs
  removed
- headless contract: the staged-writer list (F5); `add_roi(job=...)`,
  `set_job`, `save_roi_job` (9)
- terminology contract: the `use` reading rule removed; the internal-names
  table maps `job` to the ROI set (9)
- `docs/hdf5_format.md` rewritten for the one format;
  `docs/contract_coverage.md`: gaps that only cover compatibility closed
  as removed, and a new gap G31 for this phase.

### Phase 3: the minimum format in the code

- **Reader.** It requires `FileSchemaVersion = "enginir-1.5"` and every
  core field (decision 11); display data is tolerated. Every writer sets `FileWriter` with `updated_at`;
  creation sets `FileCreator` (8). Fits and refits write `software`, and
  the refit's metric warning names differing versions (10). The `use` → `job`
  rename goes through the models, the store, the facade and the viewer (9). The
  compatibility branches go: in `project/_fit_spec.py` (legacy and partial
  specifications), `project/_models.py` (`use`, `scale`), `io/project_store.py`,
  `models/pls.py` and `models/pcr.py` (`scale` and the StandardScaler path
  removed from the API, 6), `project/_signal_domain.py` and
  `prediction/_service.py` (`eiff_raw`, 6), `modeling/_service.py` (legacy
  refit by position; GroupKFold stays as an optional rule), `prediction/_service.py`
  and `viewer/_prediction.py` (models without projection data),
  `viewer/_model_diagnostics.py` and `viewer/_modeling.py` (placeholders,
  VIP from weights), and `preprocessing/recipe.py` (defaults). The precise
  list is made at the start of the phase.
- **Writers.** They write every field (F2). The staged prediction writer
  works under `/staging/` and finalizes by moving the group (F5).
  Fault-injected test: a write interrupted between initialize and finalize
  leaves a project that opens, with one warning, and no run in
  `/predictions`.
- **Tests.**
  - Tests that build legacy runs or old files are removed.
  - New: the reader refuses an older version string with the message;
    refuses a 1.5 file with any core field removed (one test per core
    field, generated from a list, so no field is forgotten); opens a file
    with display data removed, and the viewer pages show "–" for it;
    reads and refits a complete file.
  - Fault-injected: a reader that accepts a missing `job` fails the test;
    a write that does not update `FileWriter` fails the test; a refit
    whose metrics differ under a changed (mocked) scikit-learn version
    names that version in its warning, and fails without the comparison.
- **The four gates.**

### Phase 4: bug fixes

From the GUI review (2026-11-11). With the minimum format, B3 and B4 (old
and half-written projects) are no longer bugs: they are refused by design.

1. **Fit Again restores the signal unit (B1).**
   - `apply_model_run` sets the signal unit from the run's input contract.
   - Test: an Absorbance model refits on Absorbance while Explore shows raw
     counts. Fault-injected without the fix.
2. **The full seed range (B2).**
   - The seed field accepts 0 ≤ seed < 2⁶³.
   - Test: seed 2⁴⁰ is preset unchanged and refits with the same checksum.
3. **ROI names stay visible (B5).**
   - The Dataset column shows the file name without its folder, with the
     full label as tooltip; rows stay grouped by data file.
   - No column is pinned (user decision 2026-11-13; the pinned Dataset and
     ROI columns of Phase 4 were removed).
4. **Fold warnings in the fit summary (B6).**
   - The summary shows the run's fold warnings in words, which every 1.5
     run has.
   - The Validation line wraps.
   - Test, fault-injected: a run with `one_level_per_file` shows it.

### Phase 5: wording and small display fixes

1. **Viewer strings:**
   - "Smoothing and derivatives:" and "Scatter correction:" in the fit
     dialog
   - "the model is fitted with this"
   - "Exclude pixels above DCrit (DModX)"
   - "more pixels above DCrit than allowed"
   - ("Cross-track" on the sensor frame stays, by the user's decision of
     2026-11-12)
   - PC1, PC2, … in PCR titles
   - "engiNIR", without "EDA", on the welcome screen
   - the dataset selector shows a file opened without a project.
2. **Every Explore y axis** uses the signal's shown name ("Raw counts
   (DN)"). Test with a raw-counts dataset.
3. **The library's fold-warning texts** in the agreed terms ("the ROIs of
   one dataset share one value").
4. **The Distance page notes** in one corner, without overlap. Test: their
   bounding boxes don't overlap (Agg renderer).
5. **The terminology scan:**
   - `train…` and `outlying` are added, as are the exact labels "Filter:"
     and "Normalization:".
   - A library-message scan (`log_and_raise`, `logger.*`, `raise …(...)`
     outside the viewer) starts with an inventory. If it finds more than
     about 20 strings, it moves to the GUI polish phase and only the
     strings shown in the viewer are fixed now.

### Checkpoint: the 1.5.x demo project

The user builds a new demo project with the branch (several datasets, ROI
grids, calibration and prediction ROIs, a model) and checks it in the
viewer: Fit Again, the ROI table after Predict ROIs, the fit summary. Before
going on, the findings go into this plan.

Findings (2026-11-13), fixed on branch `fix/1.5-open-and-create`:

- a project of an earlier format was read as a dataset ("has no dataset
  'spectra'"); it is now refused as a project, naming its format
- creating a project in a folder the user cannot write to showed nothing
  (traceback in the log); it is now reported in a dialog
- a target without values was refused by its id; now by its name
- a canceled band proposal or prediction was logged as a failed job with a
  traceback; now as info
- ROI table: no column is pinned (the pinned Dataset and ROI columns of
  Phase 4 were removed); the Dataset column shows the file name without its
  folder, the full label as tooltip, rows grouped by data file

Not fixed: a Qt `commitData` warning while editing the ROI table (no lost
edit seen); one full test run crashed once and did not recur in three runs.

### Phase 5b: data files across platforms

The project is used from Linux (WSL, data under `/mnt/z/…`) and Windows
(the same data under `Z:\…`). The project opens on both, since the ROI
spectra are stored in it, but every data file is "missing" on the other
platform: one Locate per dataset, and each Locate rewrites `source_path`,
so switching platforms rewrites the paths back and forth.

Decisions (user, 2026-11-13):

- **Find the data file at open time; never rewrite the project for it.**
  When the stored path does not exist, candidates are tried in a fixed
  order; the first that exists and passes the identity check is used for
  this session. Only `Locate File…` writes a path.
- **A. Relative first.** Order: path relative to the project file →
  stored absolute `source_path` → drive mapping. A project folder copied
  with its data opens the copy next to it, not the originals on the share.
- **B. The relative path is always stored when it can be computed.** It
  also covers `projects/demo.h5` next to `Imaging_data/…`
  (`../Imaging_data/…`). On Windows there is no relative path between
  drives: it is then stored empty (undefined value). A relative path that
  crosses mounts on Linux is stored too; on the other platform it is not
  found and the next candidate is tried.
- **C. The drive mapping always applies** (not only in WSL): `/mnt/<x>/…`
  ↔ `<X>:\…`, with the WSL mount root read from `/etc/wsl.conf`
  (`[automount] root`, default `/mnt/`). A miss costs one `exists()`.
- Path mappings in the user settings (`\\172.16.0.61\z` → `/mnt/z`, macOS
  `/Volumes/…`) are not part of this phase.

Work:

1. **Contracts.**
   - `project-dataset-attachment-contract_v3.md`, Missing and moved files:
     the candidates and their order, the identity check, nothing written on
     open; the viewer's tooltip shows only the path the file is read from
     (user, 2026-11-13: how it was found and the stored path are too much
     for the user), how it was found is logged at info.
   - `project-schema-contract_v3.md` and `docs/hdf5_format.md`: dataset
     attribute `relative_path` (POSIX separators, relative to the project
     file's folder; display data: tolerated when missing, empty when not
     computable). Written by `add_dataset` and `Locate`, never on open.
2. **Stored paths are parsed by their own syntax.** Today
   `_native_path` (`_workflow.py`) swaps separators and calls `abspath`, so
   `Z:\NonKF\x.h5` read on Linux becomes a path inside the current folder.
   A drive letter or UNC prefix means a Windows path, a leading `/` a POSIX
   path; only then are they mapped. Fixed and tested on its own.
3. **One resolver in the library**, e.g. `resolve_data_file(project_path,
   dataset) -> (path, how) | None`, used everywhere a stored data-file path
   is opened or compared: `Project.dataset(path)` and `Project.datasets`
   (`_workflow.py`, matched by `_same_path`), showing a dataset, the
   missing-file marks of the data file selector, the reopen binding check
   (`_viewer.py`). Headless and viewer behave the same.
4. **Identity check.** Done as: a candidate other than the stored path
   must have the format, shape, modality and wavelengths and the pixels of
   the dataset's smallest ROI (a few frames, cheap over the share); the
   stored path itself is used as before. Locate keeps the full check of
   every ROI. (Changed from the proposal of a full check at first show:
   the smallest ROI already tells two `stream_hsi.h5` files apart.)
5. **Batch Locate.** After one successful Locate, offer the same folder
   change for the other missing data files ("4 more files found under the
   new folder"); each is checked, and each accepted one writes its paths.
6. **Tests**, each fault-injected: a project written with `/mnt/z/…` paths
   opened with the data under another root (relative), a Windows path read
   on Linux (parsing), the drive mapping both ways (pure path functions, no
   real drives), a candidate that exists but is another file (refused), an
   open that writes nothing (`updated_at` unchanged), batch Locate.
7. Docs: user guide (moving projects between Windows and Linux), API
   reference for the resolver, coverage map (new rule, tests).

Not affected: dark and white references are stored as arrays (their
`*_ref_source` is provenance only, never read again); prediction records
keep their `data_path` as a record.

### Phase 6: GUI polish

Plan to follow, written after the checkpoint so it can use the demo
project. The 1.5.0 plan and the review of 2026-11-11 list the candidates:

- the workspace tabs at the top, out of the closable Controls dock (a
  viewer-contract requirement)
- a context bar and a next-step hint
- empty and loading states instead of 0–1 axes
- a compact Controls dock
- the ROI table's width in Model and Predict
- an Inspector that does not repeat the fit summary
- plot titles that are not truncated, and a readable Spectral page
- the model health summary (share of pixels above DCrit against the
  expected 1 %, unusual ROIs).

### Phase 7: release

Released as 1.5.1 from main (user, 2026-11-13), with the kernel engine of
`kernel-pls.md` and without Phase 6. CHANGELOG: one
section with a prominent **Breaking** list: the new project format with
older files refused, the CSV columns, the layer names, the PCR names,
"canceled", the `scale` parameter of `PLSRegression` and `PCR`
removed (use `x_scaling` and `y_scaling`), and the ROI `use` renamed
`job` in the file and the API (`set_job`, `save_roi_job`, `add_roi(job=)`). Four gates. Then, on the
user's go: merge, tag, CI on the tag, GitHub release (Zenodo, PyPI).

## Related plans

- `kernel-pls.md` reads absent `algorithm` keys as `nipals` and `svd`
  for older models. Under decision 1, it must be updated before it starts:
  the format requires `algorithm`, and older files do not exist.

## Out of scope

- Repair or migration of older projects (decision 1).
- The kernel PLS engine (its own plan, after 1.5.x).
- 2.0 features: the method file, calibration transfer, OPLS, EPO, OSC and
  classification.
