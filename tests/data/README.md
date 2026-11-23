# Test data

Committed inputs of tests. Nothing here is packaged: the sdist and the wheel
hold only `src/enginir`.

## `simca_reference/`

The SIMCA 18 reference project (tables, export scripts); see its README.

## `enginir_1_5/`

A project in format `enginir-1.5`, the format before the reference
correction was renamed:

- `project.h5` holds two datasets: one with dark and white references from
  files, whose ROIs store reference spectra, and one already corrected
  float dataset with a prediction ROI.
- It also holds a target, calibration ROIs with values, a fitted PLS model
  and one stored prediction run.
- `scan.h5`, `dark.h5`, `white.h5` and `corrected.h5` are its data files,
  found next to the project (`relative_path`).
- `expected_map.npy` is the model's prediction of `scan.h5`.

The files were made on 2026-11-18 with engiNIR 1.6.1, before the rename, by:

```bash
uv run python -m tests.data.enginir_1_5.make_project
```

The script builds the project in `/tmp/enginir-1.5`, so the stored absolute
paths name that folder and not the machine it was made on. It uses the names
of format `enginir-1.5` and does not run with 1.7 or later. Do not regenerate
the files: they stand for projects that users already have.
