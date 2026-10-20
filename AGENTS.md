# AGENTS.md

Instructions for AI coding agents working in this repository. Follow them as
written; where a rule has a reason, the reason decides edge cases.

## Commands

Always run Python through `uv run` (never bare `python`, `python3` or
`pytest`), so the project environment is used.

```bash
uv sync --all-extras                                   # install everything
uv run pytest                                          # full suite (60 s per-test timeout)
uv run pytest tests/test_workflow.py -q                # one file
uv run ruff check src/enginir tests docs/examples      # lint
uv run pyright src/enginir                             # type check
uv run python -m sphinx -W -E -b html docs /tmp/docs   # docs; warnings are errors
```

Viewer tests need `QT_QPA_PLATFORM=offscreen` on machines without a display.

A change is done only when all four gates are clean: pytest, ruff, pyright
and Sphinx. Build Sphinx with `-E` into a fresh directory; a cached build can
hide warnings.

## Sources of truth

- `contracts/*_v3.md` freeze persistence, runtime and workflow behaviour. When
  code, docs or this file disagree with a contract, the contract wins; change
  a contract only deliberately and in the same change as the code.
- `docs/contract_coverage.md` maps every contract rule to its implementation
  and tests, with the gap register (`G01`–`G18`). Update it when you close or
  open a gap.
- `docs/hdf5_format.md` documents the project file layout.

## Architecture

engiNIR is a chemometrics library for NIR spectra and hyperspectral images
with a desktop viewer. Spectra are `(n_samples, n_bands)`, image cubes
`(n_frames, height, n_bands)` (push-broom; ENVI cubes are `(lines, samples,
bands)`).

| Package | Contents |
|---|---|
| `enginir.Project` (`_workflow.py`) | the workflow facade: datasets, targets, ROIs, `fit`, `predict`, addressed by name; plus the shared rules `resolve_dataset`, `make_calibration`, `resolve_calibration`, `resolve_prediction_calibration`, `make_roi` |
| `io/` | file readers (`load_spectra` dispatches by extension; every loader returns `SpectralData`), project store (`project_store.py`, groups in `_hdf5_groups.py`) |
| `project/` | typed project objects, hydration, relationship validation, signal-domain conversion |
| `preprocessing/` | SNV, MSC, EMSC, Savitzky-Golay, Detrend, Smoother; `PreprocessingRecipe` |
| `models/` | PLS, PCR, SIMCA, LDA, SVR, random forest, MLP (scikit-learn API) |
| `validation/` | cross-validation, Kennard-Stone, figures of merit, outliers |
| `modeling/`, `prediction/` | sample sets and model fitting; chunked prediction |
| `visualization/` | matplotlib plots |
| `viewer/` | PyQt6 + pyqtgraph viewer; see `docs/viewer_architecture.md` |
| `_checks.py`, `_logging.py` | validators with `log_and_raise`; Loguru setup |

Boundaries:

- Qt is imported only in `viewer/`. Everything else, including
  `import enginir`, stays Qt-free (`tests/test_headless_api.py`).
- The viewer holds no workflow rule. `ProjectState` mirrors `Project` for the
  widgets and writes only through `Project.save_*`; Explore and Predict use
  the shared rules above. The only direct writer call in the viewer is the
  prediction worker storing its job's new dataset and calibration (checked by
  `test_viewer_writes_projects_only_through_project`).
- Services take plain callables for progress and cancellation; the viewer
  adapts them to Qt signals in `viewer/_jobs.py`.

## Frozen scientific semantics

These are contract decisions; do not change them without a contract change.

- Pixels are observations; targets, grouping, cross-validation and reported
  metrics are per ROI (ROI mean). Never split one ROI's pixels across folds.
- Stateful preprocessing (MSC, EMSC) is fitted inside every cross-validation
  fold.
- Training and prediction convert spectra with the same function
  (`transform_spectra_to_signal_domain`).
- Calibrations belong to their dataset; stable ids, not names, define
  identity; stored objects are never overwritten with different content
  (names are mutable labels).
- Raw ROI spectra are stored as recorded (uint16 counts, or float32 for
  precalibrated data); working arrays are float32. Do not upcast to float64
  or add copies without a measured reason.
- Wavelength values are metadata, not a prediction gate (open decision G18).

## Code conventions

- Log before raising: use `log_and_raise(...)` from `enginir._checks`, or
  `logger.error(...)` immediately before `raise`. Saved logs are the error
  report users send.
- Loguru calls use `{}` placeholders, never f-strings or `%s`.
- Every module that does work has `logger = get_logger(__name__)` from
  `enginir._logging`. The library is muted until `enable_logging()`; the
  viewer logs to `~/.enginir/enginir.log` (or `$ENGINIR_LOG_FILE`).
- No `assert` in production code (removed by `python -O`).
- Division guards use `_EPS = 1e-10` with a `logger.warning`.
- Compare wavelength axes with `np.allclose(..., atol=1e-6, rtol=0.0)`.
- Do not cast classification labels to float; detect classifiers with
  `is_classifier`.
- PLS `coef_` is `(n_bands,)` or `(n_bands, n_targets)` (transposed from
  scikit-learn). `metrics_report` takes calibration and test vectors
  separately.
- Narrow optional Qt/HDF5 values at boundaries instead of suppressing pyright.
- Python 3.13, src layout (`src/enginir/`), ruff target `py313`.

## Tests

- Test behaviour that matters, not bookkeeping: no tests that only restate an
  implementation or check trivial attributes.
- When a test is meant to catch a fault, check that it fails when the fault
  is injected.
- Documented examples in `docs/examples/` run as tests
  (`tests/test_docs_examples.py`).

## Docs

- User docs: `docs/user_guide/`; API reference: `docs/api/` (every exported
  function has its own section; use `automodule` with `:no-members:` when
  functions are listed separately).
- Keep `docs/api/viewer.rst`, `docs/user_guide/viewer.rst` and
  `docs/viewer_architecture.md` in sync when the viewer changes.
- Public docs contain no notebooks, internal URLs or local paths.

## Repository and release

- The repository is https://github.com/TUstudents/engiNIR (Zenodo concept
  DOI 10.5281/zenodo.23235907); use plain `git` against `origin`.
- Commit, tag or push only when asked.
- The version lives only in `pyproject.toml`; `enginir.__version__` and the
  docs read it from the package metadata.
- Release `X.Y.Z`: set the version in `pyproject.toml`, add the
  `CHANGELOG.md` entry, run all four gates, commit, tag `vX.Y.Z`, push the
  tag, wait for CI on the tag, and publish a GitHub release from it (Zenodo
  archives it and mints the version DOI; `.github/workflows/publish.yml`
  uploads it to PyPI with Trusted Publishing, after checking that the tag
  matches the version, the pre-release flag matches the version and CI
  passed for the commit). Release candidates are `X.Y.ZrcN`, tagged
  `vX.Y.ZrcN` and published as GitHub pre-releases. A PyPI version can be
  uploaded only once, even after deletion. Check with `git rev-list -n 1 vX.Y.Z` and
  `git ls-remote --tags origin vX.Y.Z`.
