# Plan: project format enginir-1.7, "calibration" renamed to "reference correction"

Status: decided (user, 2026-11-18): D1 to D8 below. In progress on branch
`format-1.7`: Phase 0 (05e8b63), Phase 1 (29f096a), Phases 2 and 3 in one
commit (the store and the API share every name).

## Goal

In the project file and the API, the word *calibration* means two things:

- the dark/white **reference correction** (raw counts to reflectance);
- model **calibration** (ASTM E1655): the calibration set, calibration ROIs
  and the calibration range.

The user-facing words have kept the two apart since 1.5
(`contracts/terminology_v3.md`): the viewer, the docs and the library
messages say "reference correction". The stored names and the Python API
still say `calibration`.

This is a problem for anyone who reads a project file or an audit export
without the code. A prediction run stores `@calibration_id`, which looks
like the model's calibration but names the reference correction applied to
the predicted scan.

The new format `enginir-1.7`, released as engiNIR 1.7.0, therefore renames every stored name that means
the reference correction. The public API gets the same names. A migration
tool converts `enginir-1.5` projects.

The rename was deferred to 2.0 in `plans/terminology.md` ("Deferred to
2.0"). This plan brings it forward to 1.7.0 and decides the questions that
plan left open.

## Scope (measured 2026-11-18, on `5f2b16b` plus the uncommitted G11 change)

Occurrences of the reference-correction names (`CalibrationSpec`,
`calibration_id`, `calibrations`, `calibration_refs`, `active_calibration*`,
`required_calibration_mode`, `calibrated_input_allowed`, `precalibrated*`,
`*_calibration`, `.calibration`, and the shared rules):

| Where | Occurrences | Files |
|---|---:|---:|
| `src/enginir` | 537 | 25 |
| `tests` | 375 | 22 |
| `docs` | 61 | 10 |
| `contracts` | 103 | 8 |

The most frequent names in the source are `calibrations` (143),
`calibration_id` (120), `CalibrationSpec` (73), `active_calibration` (48)
and `precalibrated` (37).

These names stay, because they already mean model calibration:

- the calibration set (`SampleSet`, by its term);
- calibration ROIs (`job = "calibration"`);
- `calibration_scores`, `roi_scores` and the calibration range;
- the metric keys.

Also out of scope: the `calibrated` attribute that *input* data files carry
(an external format, not the project file).

## Decisions (user, 2026-11-18)

**D1. Release and format name.** engiNIR **1.7.0**, format
**`enginir-1.7`**: a format change is not a patch, and the format is named
after the release that introduces it. `FileCreator` and `FileWriter` keep
recording the software version.

**D2. File and API together.** Rename the stored names and the public API
in the same release. Otherwise the code translates between two vocabularies
forever.

**D3. The new names.** Use the contract term in full; the names are long
but cannot be misread:

| Now (file / API) | New |
|---|---|
| group `datasets/{id}/calibrations/{id}/` | `datasets/{id}/reference_corrections/{id}/` |
| dataset `@active_calibration_id` | `@active_reference_correction_id` |
| ROI `@calibration_id` | `@reference_correction_id` |
| ROI group `calibration_refs/{dark,white}` | `reference_spectra/{dark,white}` |
| prediction run `@calibration_id` | `@reference_correction_id` |
| input contract `@required_calibration_mode` | `@required_correction_mode` |
| input contract `@calibrated_input_allowed` | `@corrected_input_allowed` |
| mode value `precalibrated_import` | `already_corrected` (the glossary's term) |
| mode values `raw_dn`, `dark_white_reference` | unchanged (unambiguous) |
| `CalibrationSpec` | `ReferenceCorrection` |
| `CALIBRATION_MODES` | `CORRECTION_MODES` |
| `calibration_id` (fields, keyword arguments) | `reference_correction_id` |
| `Dataset.calibration`, `Project.save_calibration` | `Dataset.reference_correction`, `Project.save_reference_correction` |
| `make_calibration`, `resolve_calibration`, `resolve_prediction_calibration` | `make_reference_correction`, `resolve_reference_correction`, `resolve_prediction_reference_correction` |
| `check_calibration_signal_domain_compatibility` | `check_reference_correction_compatibility` |
| store `save_calibration_spec`, `iter_roi_calibration_refs` | `save_reference_correction`, `iter_roi_reference_spectra` |
| `contracts/dataset-calibration-contract_v3.md` | `contracts/dataset-reference-correction-contract_v3.md` |
| `CalibrationStats` (conversion counters) | `ReferenceCorrectionStats` |
| progress phase `calibrating` | `correcting` |
| prediction provenance `calibration_id`, `calibration_mode` | `reference_correction_id`, `correction_mode` |

Internal names (private functions, the viewer's `apply_shown_calibration`
and similar) follow the same rule. The guard test of Phase 5 decides where
the rule applies.

Not chosen: a shorter `correction` / `correction_id`; it is vague next to
the other corrections (baseline, scatter).

**D4. Projects in format `enginir-1.5`.** Add a migration tool. The reader
stays strict and refuses every other format with the one message it gives
today, which then names the tool.

Projects written by 1.5.1 to 1.6.1 are in public use (PyPI). Refusing them
alone would strand their ROIs, values and models. The 1.5 decision "no
backward compatibility" was about not *reading* old shapes, and the reader
keeps that rule. A one-time conversion to a new file does not break it.

Not chosen: refusal only (strands users), and a compatibility reader (it
brings back the "read as if" rules that 1.5 removed).

**D5. How the migration works: a command, not the viewer.**

- `enginir.io.migrate_project(source, target)` and a console command
  `enginir-migrate SOURCE TARGET`.
- It copies the file, then renames groups, attributes and the mode value in
  the copy. It never changes the source and refuses an existing target.
- It accepts only `enginir-1.5`. It sets `FileSchemaVersion =
  "enginir-1.7"` and `FileWriter`, and keeps `project_id`, `FileCreator`,
  `created_at` and every object id and value. Ids are UUIDs with no prefix,
  so no id changes.
- It ends by opening the result with the normal strict reader. If that
  fails, it deletes the target and reports why.
- It writes one log entry with source, target and the counts of renamed
  objects.
- The viewer does not convert. Its refusal message for an `enginir-1.5`
  project names the command, with the file's path filled in.

**D6. No deprecated aliases.** The old public names are removed in 1.7.0:
the class, the constant, the facade properties and methods, the shared
rules and the `calibration_id=` keyword arguments. Scripts that use them
fail with `AttributeError` or `TypeError`. The CHANGELOG lists every old name
with its replacement.

**D7. One branch, one phase per commit.** Use branch `format-1.7` and a
draft PR, as for 1.5.x. Feature work does not share the release; the
rename is large enough on its own.

**D8. A test project in format `enginir-1.5`, not shipped.** Make a
minimal synthetic example project with **1.6.1, before any rename**:

- two small scans;
- ROIs with values;
- one dataset whose references come from files, so that ROI reference
  spectra are stored;
- a fitted PLS model.

Commit it as `tests/data/enginir-1.5-project.h5`, with the prediction map it
gives as `.npy`. Keep it as small as the builder allows (target: under
1 MB). This is the only way to test the migration once the old writer is
gone.

It must not reach PyPI. Checked on 2026-11-18: the sdist built by
`uv_build` contains no `tests/` files, and the wheel packages only
`src/enginir`. Phase 6 checks this again on the release archives.

## Phases

### Phase 0: freeze the old format (before any rename)

- Make the D8 project with the released code from the synthetic builders
  (`tests/_scans.py`). Record in `tests/data/README.md` the command, the
  version and the seed.
- Test: the project opens with the current reader, and its model predicts
  the stored map.

### Phase 1: contracts and file documentation

Update the following to the D3 names, with format `enginir-1.7`:

- `project-schema-contract_v3.md`: layout, object identity and the
  compatibility policy, with the migration as the only route from 1.5;
- the renamed dataset-reference-correction contract, plus
  `roi-target`, `prediction-runtime` and the attachment contracts;
- `docs/hdf5_format.md`;
- `terminology_v3.md`: the table "Internal name → Term" shrinks, because the
  names now *are* the term;
- `docs/glossary.rst`.

Add a new gap `G36` (stored names that contradict the terminology) to
`contracts/coverage.md`, open until Phase 5.

### Phase 2: the store and the typed layer

- `FILE_SCHEMA_VERSION = "enginir-1.7"`. Rename the group and attribute
  names in `project_store.py`, `_hdf5_groups.py`, `_hydration.py` and the
  relationship validation, and rename the typed class and fields.
- The refusal message for `enginir-1.5` names `enginir-migrate` with the
  file's path; the viewer shows that message.
- Tests:
  - the per-core-field test (`tests/test_project_file.py`) runs with the
    new names;
  - fault-injected: a file that still has `calibrations/` or `@calibration_id`
    under the new `FileSchemaVersion` is refused, naming the field.

### Phase 3: the public API

- New names (D3) everywhere in `enginir`, `enginir.project` and
  `enginir.io`. No aliases (D6).
- Tests:
  - the documented examples (`docs/examples/`) run with the new names;
  - `docs/api/` documents every new name, checked by the Sphinx `-W`
    build.

### Phase 4: migration

- `migrate_project` and the console command `enginir-migrate` in
  `pyproject.toml` (D5), documented in the user guide (I/O, "Projects from
  engiNIR 1.5 and 1.6").
- Tests on the D8 project:
  - the migrated file opens strictly, keeps every id and value, and its
    model predicts the stored map exactly;
  - the source is unchanged (checksum);
  - an existing target is refused;
  - a non-1.5 source is refused;
  - an interrupted migration leaves no target;
  - fault-injected: a missed rename, which the strict reader must catch.
- Test that the viewer's refusal of the D8 project names the command.

### Phase 5: everything else, and the guard

- Rename the internal code, the viewer, the tests, the benchmarks and the
  docs (`docs/user_guide/`, `docs/api/` with sections for the new names,
  `viewer_architecture.md`).
- Guard test (`tests/test_terminology.py`): no stored key and no public name
  says `calibration` in the reference-correction sense. It checks:
  - the store's group and attribute names, read from a freshly written
    project;
  - the public names, from `dir()` of the public modules.

  It has an allow-list of the model-calibration names with reasons.
  Fault-injected: putting back `calibration_id` in the writer fails it.
- Close `G36`.

### Phase 6: release 1.7.0

- CHANGELOG: format `enginir-1.7`, the migration command, and a table of
  every renamed name (old to new). State that the old Python names are gone.
- Build the sdist and the wheel and check that neither contains `tests/` or
  an `.h5` file (D8).
- All four gates and the full CI run, including macOS and the Windows
  install.
- Release as in `AGENTS.md`.
- The example project (`_example.py`) is written in the new format by
  construction.

## Risks

- **A missed rename in a rarely used path,** for example stored
  predictions or ROI reference spectra. Counter-measures:
  - the strict reader refuses unknown and missing fields;
  - the guard test checks the written file;
  - the fixture exercises reference spectra (one dataset with references
    from files).
- **Users' scripts break.** This is accepted (D6). Counter-measures: the
  CHANGELOG table of old to new names, and the error message for old
  projects that names the migration command.
- **The size of the change.** One phase per commit, each passing the four
  gates. Phases 2 and 3 are mechanical renames, done with a checked script
  and then reviewed by hand. The script is not committed.

## Out of scope

- Model-calibration names, the `raw_dn` and `dark_white_reference` mode
  values, and input-file attributes.
- Any other format change. If one is wanted for `enginir-1.7`, it gets its
  own decision here before Phase 1.
