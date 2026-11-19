# Plan: engiNIR 1.7.1, a fast start that never waits for a network drive

Status: decided (user, 2026-11-19): D1 to D8 below. Phases 1, 2, 3 and 5 done;
Phase 4 measured, waiting for a decision (see there).

## Goal

The welcome screen appears quickly, opening a project does not wait for
the scientific libraries, and no step of the start or of opening a project
waits for the file system of a data path on the UI thread. That matters most
for a network drive (Z:) that is slow or unreachable. Measured by the user
on Windows (2026-11-19): about 20 s from the click to the welcome screen and
another 10 s to an empty project. The same work takes 5.2 s in a warm
process, so the rest is a cold start (Defender scanning about 2,200 files),
other load on the machine, and the network drive. The fix is to do less
before the first window and to do nothing synchronously against data paths.

## What the start does today (measured 2026-11-19)

User's Windows machine, warm, in-process (`-X importtime` and a step
script):

| Step | Time |
|---|---:|
| `import enginir.viewer` (the launcher needs it for `__version__` and `open_project`) | 3.34 s |
| of it: `enginir.modeling._bands` → `sklearn.model_selection` (with `scipy.stats` 0.96 s) | 1.86 s |
| of it: `enginir.io.csv_io` → pandas | 0.57 s |
| of it: `enginir.preprocessing.derivatives` → `scipy.signal` | 0.17 s |
| of it: loguru (with asyncio), h5py | 0.17 s, 0.11 s |
| logging, Qt app and theme | 0.14 s |
| the recent-files list: `os.path.exists` and an h5py open per `.h5` (5 of 10 on Z:) | 0.39 s |
| welcome window | 0.12 s |
| the main viewer's modules (matplotlib, pyqtgraph, pages) | 1.14 s |
| **total** | **5.16 s** |

Linux, warm: 1.4 s to the welcome window, 0.8 s more for an empty project.

On the UI thread and against data paths:

- **Recent files:** the welcome window calls `os.path.exists` on every
  recent path and opens every `.h5` with h5py to tell a project from a
  dataset (`_welcome._populate_recent`). With Z: online that cost 0.4 s; with
  Z: unreachable, each call waits for the network timeout.
- **Opening a project:** `HSIViewer.refresh_checks` calls
  `project.checks(predict=False)`, which looks up every dataset's data file
  (`Project.find_data_file`, several candidate paths each, `_data_paths`).
  The viewer asks `ProjectState.data_file` and `find_data_file` again in
  about 20 places (the browser, the Explore page, Predict, Locate), and
  nothing caches the answer.
- **Install:** `scripts/install-enginir-for-windows.ps1` runs
  `uv tool install` without `--compile-bytecode`, so the first start after
  an install or upgrade compiles about 2,200 modules.

## Decisions (user, 2026-11-19)

**D1. Release.** engiNIR **1.7.1**, a bugfix release, with no format or API
change. Branch `startup-speed`, one commit per phase, a draft PR.

**D2. A light `import enginir`.** `enginir/__init__.py` and
`enginir/viewer/__init__.py` resolve their public names on first use (PEP
562 `__getattr__`, typed with `TYPE_CHECKING` imports). `import enginir` and
the launcher then import only the standard library, loguru and Qt. Scripts
keep `from enginir import Project` unchanged; only the moment of the import
moves. The launcher (`viewer/__main__.py`) imports `_app`, `_welcome`,
`_jobs`, `_loading` and `_recent` directly, and nothing that reaches
`_workflow`, scikit-learn, SciPy, pandas, h5py, matplotlib or pyqtgraph.
`__version__` comes from the package metadata as now.

**D3. Warm up while the welcome screen waits.** When the welcome screen is
painted, a background thread imports the scientific stack: `enginir._workflow`
(which brings scikit-learn, SciPy and pandas), h5py, and `enginir.io`. The
viewer's Qt modules (`_viewer`, matplotlib's Qt backend, pyqtgraph) are
imported on the UI thread in small steps from an idle timer, because Qt
modules are only safe to import on the main thread. By the time the user
picks a file, opening it finds everything loaded; a pick before the warm-up
ends waits only for what is left.

**D4. Recent files without file access at startup.** Each entry in
`recent.json` stores its kind (`project` or `dataset`), set when the entry is
added, so the list is drawn from the file alone, with no `exists` and no
h5py open. A background job checks which entries exist, one path at a time,
and grays out the missing ones when it is done; an unreachable Z: delays
only that, never the window. Entries written by 1.7.0 and earlier have no
kind: the job determines it once and stores it.

**D5. Data-file lookups once per project, in the background.**
- `ProjectState` caches `find_data_file` per dataset for the session. The
  cache is cleared by Locate, by adding or relocating a file, and by
  **Refresh** (or reopening).
- Opening a project shows its datasets at once as "checking…" and starts a
  background job that finds every data file, then fills the cache and
  updates the browser, the Explore page and the checks.
- `refresh_checks` calls `project.checks(files=False, predict=False)` on the
  UI thread. The file check `data.files_missing` arrives from the job, like
  the prediction check (one job for both, one generation counter).
- Showing a dataset whose file is not yet checked waits for that one
  lookup, with the busy indicator, never for all.


**D6. Bytecode at install.** The installer adds `--compile-bytecode` to
`uv tool install`, so the first start after an install or upgrade does not
compile. The `windows-install` CI job checks that the `.pyc` files exist.

**D7. Measuring the start in the log (no benchmark).** The first log line
states the time since the process started, read from the operating system
(Windows `GetProcessTimes` through ctypes; Linux and macOS `/proc` or
`ps`). It also logs the time to the welcome window shown, to the warm-up
done, and to the main window shown. A user's log then shows the whole start,
including the time before Python ran. No startup benchmark.

**D8. The Model and Predict pages are built on first show.**
`HSIViewer._modeling_view` and `._prediction_view` become properties that
build the page on first access. The window is then built without them. Their
docks (the Fit settings, the Prediction settings, the Models panel) and
their signal connections are set up when the page is built. Today
construction touches them in many places (71 uses in `_viewer.py`, 9
files), so the phase first moves every construction-time use behind the
first build. A Project-only window that never opens Model or Predict never
builds them.

## Phases

### Phase 1: a light import (D2), with a guard test

- Lazy names in `enginir/__init__.py` and `enginir/viewer/__init__.py`. The
  launcher imports only its own modules.
- Guard test (`tests/test_headless_api.py` or a new
  `tests/test_startup.py`): in a subprocess, the launcher path up to a
  constructed `WelcomeWindow` imports none of `sklearn`, `scipy`, `pandas`,
  `h5py`, `matplotlib`, `pyqtgraph`, `enginir._workflow`. Fault-injected: one
  eager import put back fails it.
- `from enginir import Project` and every public name still work (a test
  that resolves every name in `__all__`). pyright still sees the types.

### Phase 2: warm-up (D3)

- The welcome window starts the warm-up after its first paint. Main-thread
  steps run from a `QTimer` with a short interval; the background thread
  imports the non-Qt stack.
- Tests:
  - after the warm-up, opening a project imports nothing new (compare
    `sys.modules`);
  - a pick during the warm-up opens correctly;
  - the warm-up thread has ended before the process exits
    (`wait_for_running_jobs`).

### Phase 3: recent files and data files off the UI thread (D4, D5)

*Done (2026-11-19).* As built: the session cache of D5 already existed
(`Project.find_data_file` keeps each answer, Locate sets it), so
`ProjectState` only records which datasets were looked up
(`data_file_known`) and the window leaves the others to one background job
(`look_up_data_files`, signal `data_files_checked`). The data-file selector
says "(checking…)", the checks leave out the file check, and the last shown
file waits for the job. The Open Recent menu checks in the background too
and grays out missing files, like the welcome window.

- `recent.json` with the kind per entry; the background existence check.
- The data-file cache and the background lookup at project open; checks
  with `files=False` on the UI thread, the file check from the job; the
  "checking…" state in the browser and the Explore page.
- Tests:
  - with a fake slow `os.path.exists` (a monkeypatched delay, or a path that
    blocks), the welcome window and an opened project appear without
    waiting; the missing state arrives later;
  - the cache is used (count lookups) and cleared by Locate and Refresh;
  - the contract's Missing and moved files rules still hold
    (`tests/test_data_files.py`, `tests/test_project_open.py`);
  - fault-injected: a lookup on the UI thread at open, and no cache.

### Phase 4: the Model and Predict pages on first show (D8)

*Measured (2026-11-19, Linux, warm, an empty project).* The window takes
0.41 s to build. The Model page takes about 0.05 s of that, nearly all of
it the embedded Fit settings form (`FitModelRunDialog`, `embed_fit_settings`),
and the Predict page 0.01 s. The rest is the Explore page: the controls
panel with its plot panels (0.14 s) and the docks (0.05 s). Lazy pages
would save about 15 % for a restructuring of 71 uses in 9 files, so the
phase waits for a decision (user).

- Measure first: the time to build `HSIViewer` for an empty project, and the
  share of `ModelingView` and `PredictionView` in it (Linux, warm).
- The two pages become lazily built properties. Construction-time uses
  (dock contents, signal connections, the workflow strip, the Models
  panel, the checks' fixes that open a page) move to the first build or
  look at the page only once it exists.
- Tests:
  - an empty project's window has built neither page; showing Model or
    Predict builds it once;
  - every workflow fix that opens a page works from a window where it was
    not built;
  - the existing window, modeling and prediction tests pass unchanged in
    behavior;
  - fault-injected: a page built during construction fails the first test.

### Phase 5: install, measurement, docs (D6, D7)

*Done (2026-11-19).* The installer passes `--compile-bytecode`;
`windows-install` checks the `.pyc` files of modules that nothing imports
before the check. `_startup_clock` gives the time since the process start;
the `Startup timing` lines at INFO state it for the launcher, the welcome
window, the warm-up and the first main window.

- `--compile-bytecode` in the installer, and the check in `windows-install`.
- The process-start time and the step times in the log.
- Docs:
  - `docs/viewer_architecture.md` (Startup: what is imported when; the
    warm-up; no file access on the UI thread);
  - `docs/user_guide/installation.rst` (the first start after an install is
    no longer slow);
  - the attachment contract (Missing and moved files: the lookup is
    asynchronous and the state shows "checking…");
  - CHANGELOG 1.7.1.

## Acceptance

- Linux, warm: the welcome window is painted within about 0.5 s of the
  process start (now 1.4 s), and an empty project opens within about
  0.3 s of the pick after the warm-up (now 0.8 s), measured once by hand
  (no benchmark, D7).
- The guard test and each fault injection hold.
- With Z: unreachable (simulated in tests), neither the welcome window nor
  opening a project waits for it.
- The user measures on Windows, cold and warm, with the log lines of D7:
  the time to the welcome window and to the empty project, before and
  after.

## Risks

- **Importing in a background thread.** Python's import lock makes
  concurrent imports safe. Qt modules are imported only on the main thread
  (D3). A module that is half-imported by the warm-up when the UI thread
  needs it makes the UI thread wait for it, which is correct and no slower
  than today.
- **Lazy public names hide import errors until use.** The test that
  resolves every name in `__all__` catches them in CI.
- **Missing-file states arriving later.** A user may see "checking…"
  briefly. That is better than a frozen window, and the state is
  explicit.

## Out of scope

- Making scikit-learn, SciPy or pandas themselves faster to import.
- Antivirus exclusions (a site's IT decision; the guide can mention that
  Defender scanning slows the first start).
- Building other pages lazily (Explore stays built: it is the first page
  of a project).
