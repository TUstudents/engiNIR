# Plan: engiNIR 1.7.3

Status: decided (user, 2026-11-19): D1 to D8 below. In progress.

1.7.3 takes the small fixes and the model and storage items of the 1.7.2
backlog (`plans/backlog-1.7.2.md`). The ROI workflow (items 2, 4, 8) and the
domain model doc (item 5) stay in the backlog for a later release; item 4
needs its "which grid" answered first.

## Decisions (user, 2026-11-19)

**D1. Scope.** Small fixes (items 3, 7, 9, 10, and the wavelength warning)
and model and storage (items 1 and 6).

**D2. Blosc is the default for project files.** Every array a project file
stores (ROI pixels, references, model and prediction artifacts, maps) is
written with Blosc. `save_hdf5` keeps gzip by default.

**D3. Blosc codec: Zstandard, level 5, byte shuffle** (`hdf5plugin.Blosc(
cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)`).

**D4. The component curve goes up to 10 components, or the count asked.**
`A_max = max(n, min(10, max(1, G − 2), M))` (today 15). Auto chooses within
the curve, so within 10 unless more were asked for.

**D5. The Log panel uses the log level of Tools > Preferences**
("Log file level", default INFO) and follows a change at once. The log file
follows from the next start, as now.

**D6. Glossary.** Frame and Row get their own entries, and the glossary adds
the terms the viewer and the guides use but lack (list in Phase 3).

**D7. Release directly** when CI is green, as 1.7.1 and 1.7.2.

**D8. Format.** Compression is not a format change: `FileSchemaVersion`
stays `enginir-1.7`, and a reader decodes gzip and Blosc alike. Old and new
arrays mix in one file (a project written before keeps its gzip arrays).

## What is built today

- `project_store` and `_hdf5_groups` write every array with
  `compression="gzip"` (11 places); `docs/hdf5_format.md` says "Default
  compression for dataset writes is `gzip`".
- Reading Blosc needs `hdf5plugin`'s filters registered in the process. Only
  the scan reader registers them (`hdf5_io._ensure_hdf5_filters`, a lazy
  `import hdf5plugin`); the project reader does not. A Blosc project read by
  engiNIR ≤ 1.7.2 opens (the metadata are attributes) and fails on its first
  array.
- `hdf5plugin>=7.0.0` is a runtime dependency already.
- The curve: `modeling/_components.default_max`,
  `contracts/sample-set-model-run-contract_v3.md` (`components/max`).
- The Log panel: `add_log_sink(..., level="DEBUG")` in
  `HSIViewer._build_log_panel`; the file level in `_preferences.Preferences.
  log_level` and `enable_viewer_logging`.
- Reset View: `FalseColourCanvas.reset_view` (`autoRange(padding=0.02)`);
  showing an image: `set_image` (`autoRange()`, pyqtgraph's padding, possibly
  before the page has its size).
- The wavelength warning: `hdf5_io._warn_missing_wavelengths` on every read
  of a file without wavelengths, also when its dataset got a table.

## Phases

### Phase 1: Blosc project files (D2, D3, D8)

*Done (2026-11-19).* Measured (Linux, warm): the example project 1.51 MB
gzip, 1.34 MB Blosc, the same speed; the four ROIs of a real tray
(`Tablets_10%`, uint16, 8.8 MB raw): gzip 6.28 MB, write 0.17 s, read
0.039 s; Blosc 4.99 MB, write 0.10 s, read 0.008 s. Importing `hdf5plugin`
takes about 0.35 s, in `import enginir.io` (the warm-up's background job).

- One place decides a project array's compression (`_hdf5_groups`):
  Blosc-Zstandard 5 with shuffle; the 11 gzip places use it. Chunking stays
  as now. `save_hdf5` keeps `compression="gzip"` and accepts `"blosc"`.
- Every open of a project file registers the filters first
  (`_open_project`, `load_project_store`, the lazy loaders, the migration,
  the viewer's direct reads): a project of 1.7.3 opens wherever 1.7.3 runs,
  also on a reader thread.
- Size and speed, measured once on the example project and a real tray
  project (Linux, warm): file size, save and open time, gzip against Blosc.
  Recorded in the plan; no benchmark.
- Docs: `docs/hdf5_format.md` (the default, how to open a project in
  HDFView/MATLAB/h5py: `import hdf5plugin`), the schema contract (the
  compression is the writer's choice, not part of the format), CHANGELOG
  (a project written by 1.7.3 needs engiNIR 1.7.3 or `hdf5plugin`).
- Tests: a new project's arrays use Blosc (read the filter id of each array);
  a project with gzip arrays and Blosc arrays opens and predicts the same;
  a subprocess that imports only `enginir.io.project_store` opens a Blosc
  project (the filters are registered by the reader itself). Fault-injected:
  one write left on gzip; the reader without the registration.

### Phase 2: the component curve up to 10 (D4)

- `default_max`: 15 becomes 10, as `max(n, min(10, max(1, G − 2), M))`.
- Contract `components/max` and coverage; the tests that pin the curve's
  length; the viewer's curve plot needs nothing (it draws what is stored).
- Existing models keep their stored curves (up to 15); only new fits change.
- Tests: a fit with G = 30 groups gives a curve of 10; `n_components=12`
  gives 12; Auto stays within the curve. Fault-injected: 15 back.

### Phase 3: small fixes (D5, D6, items 3, 9, wavelength warning)

- **About box (item 3):** the citation as in the README (with the concept
  DOI 10.5281/zenodo.23235907) and the GitHub link
  (https://github.com/TUstudents/engiNIR), as selectable text and links.
- **Log panel (D5):** its sink uses the Preferences level; applying
  Preferences replaces the sink with the new level. Test: at INFO no DEBUG
  line reaches the panel; after a change to DEBUG one does. Fault-injected:
  the fixed DEBUG level.
- **Reset View as the default (item 9):** one function fits the image to its
  view with `padding=0.02`; `set_image` uses it, and again when the view
  first gets its size and after a maximize or restore. The Sensor frame and
  the Spatial profile follow the shared row axis. Test: after showing an
  image in a shown window, the view range equals that of `reset_view`.
- **No wavelength warning for a scan with a table:** the warning of a file
  without wavelengths is logged once per file and session, and not at all
  when the file is read as a registered dataset whose table gave it
  wavelengths. Test: showing a table dataset logs no such warning.
- **Glossary (D6, item 10):** own entries for Frame and Row; new entries:
  sensor frame, spatial image, band image, spatial profile, false-color
  image, dark and white reference, detector levels, SNV, MSC, EMSC,
  Savitzky-Golay, detrend, R², Q², RMSEC, RMSECV, RMSEP, bias, VIP, scores,
  loadings, coefficients, component curve, check and verdict, wavelength
  table. Checked against `contracts/terminology_v3.md` (which fixes the
  terms; a new term goes into both).

### Phase 4: release

CHANGELOG 1.7.3, the version, all four gates, tag `v1.7.3`, CI on the tag,
the GitHub release with the installer, PyPI, Zenodo (D7).

## Risks

- **Older readers and other tools (D2).** engiNIR ≤ 1.7.2, HDFView without
  the plugin and MATLAB cannot read the arrays of a 1.7.3 project. Users
  sharing projects must update; the guide says how to open one in Python
  (`import hdf5plugin`). If 1.7.2 is not released yet when Phase 1 starts,
  its project reader could already register the filters (a small change),
  so 1.7.2 reads 1.7.3 projects.
- **Blosc in threads.** Reading projects in background jobs (Project ROIs
  tiles, prediction checks) must find the filters registered: the reader
  registers them itself, not the caller.
- **Curve length.** A user who relied on curves up to 15 asks for the count
  explicitly; Auto never needed more than 10 in the projects seen so far.

## Out of scope (backlog, later)

The ROI workflow (items 2, 4, 8), the domain model doc (item 5).
