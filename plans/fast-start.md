# Plan: the welcome window at once, the main window without the modeling libraries

Status: built 2026-11-23 on branch `fast-start` for engiNIR 1.7.8 (decisions D1,
D2); to be measured on Windows. Follows
`plans/cold-start.md` (1.7.6), which measured where a cold start's time goes;
this plan changes what the user waits for.

## The problem (log of 2026-11-23, Windows, cold start, Open Example Project)

| step | time |
|---|---|
| Python running before the launcher's code (266 modules) | 3.82 s |
| welcome window shown | 5.71 s |
| warm-up: NumPy | 2.26 s |
| warm-up: **SciPy** (627 modules, 104 native, 67 MB) | **15.99 s** |
| warm-up: scikit-learn | 3.12 s |
| warm-up: h5py | 1.23 s |
| warm-up: matplotlib | 3.04 s |
| warm-up: engiNIR (mostly scikit-learn's models and hdf5plugin) | 5.43 s |
| warm-up: pyqtgraph | 2.76 s |
| **main window shown** | **41.41 s** |

The main window waits for the whole warm-up, because nearly every module it
imports pulls in scikit-learn and with it most of SciPy (measured on Linux,
SciPy modules / native files loaded by one import):

| import | SciPy |
|---|---|
| `enginir`, `enginir.io.project_store` | 0 / 0 |
| `enginir.project` | 534 / 88 |
| `enginir._workflow`, `.preprocessing`, `.prediction`, `.modeling` | 534 / 88 |
| `enginir._guidance`, `enginir.validation.outliers` | 493 / 82 |
| `sklearn.base` alone | 493 / 82 |

SciPy cannot be left out or made smaller: scikit-learn requires it and
`sklearn.base` loads most of it; engiNIR's own SciPy imports add 41 modules.
What can change is that the window waits for it. The causes are few and
mostly incidental:

- `project/_fit_spec.py` and `_guidance.py` import `validation.outliers` for
  the constant `LIMIT_LEVEL`; `validation/__init__` then loads
  cross-validation, scikit-learn and SciPy.
- `preprocessing/recipe.py` imports every transformer class (all
  `sklearn.base` estimators) at the top, though the recipe dataclass, its
  checks and `format_ranges` need none of them until a pipeline is built.
- The package `__init__`s of `preprocessing`, `validation`, `models`,
  `modeling` and `prediction` import all their modules; `import enginir`
  already loads names on first use (PEP 562).
- In the viewer, `_model_diagnostics.py` imports `scipy.stats` and
  `validation` at the top; `_modeling.py`, `_state.py`, `_prediction.py`,
  `_band_ranges.py`, `_project_rois.py`, `_inspector.py` import from
  `modeling`, `prediction`, `preprocessing` and `validation`.

## Goal

1. **The welcome window at once.** Nothing before it that it does not need.
2. **The main window after the libraries it shows with**: NumPy, h5py (with
   hdf5plugin), matplotlib, pyqtgraph and engiNIR's light modules. Not
   SciPy, not scikit-learn. From the log: about 12 s of imports instead of
   36 s, so the main window at about 18–20 s instead of 41 s on this cold
   machine, and at about 1 s warm.
3. **The modeling libraries load in the background while the window is
   shown.** What needs them (Fit, Predict, the model's diagnostics and
   checks) says that it is loading and runs when they are there.

## Phase 1: the welcome window

- `viewer/__main__.py` imports `enginir.__version__` (and with it
  `importlib.metadata`, `email`, `zipfile`, `json`) at the top only for
  `--version`: read it on use.
- Logging: `enable_viewer_logging()` imports loguru, which imports `asyncio`,
  `multiprocessing` and `concurrent` (about 150 modules, about 1 s cold at
  the measured 6 ms per module). Every module creates its logger at import
  (`get_logger(__name__)`), so loguru is imported with the first engiNIR
  module. D1: `get_logger` returns a small proxy that imports loguru at its
  first message; the launcher keeps its first lines (start, log file, the
  start clock) and writes them right after the welcome window is shown,
  with the sink. A crash before that still reaches the log (the crash handler
  adds the sink and writes the kept lines) and the message box. Test: the
  launcher's imports up to the welcome window load no `loguru` and no
  `asyncio` (`tests/test_startup.py`, fault-injected).
- Log the launcher's steps at INFO (QApplication, theme, welcome window
  built, shown), so the next Windows log shows what of the 1.9 s in the
  launcher is Qt's own DLLs.

Gain: about 1 to 1.5 s cold. What remains is Python's own start and Qt's DLLs
(Qt6Core, Gui, Widgets, the platform plugin); a window before Python runs
needs a native splash, which needs a packed build (not in this plan).

## Phase 2: engiNIR's light modules stay light

- `LIMIT_LEVEL` and the other constants the light modules import move to
  `enginir/_levels.py` (it exists); `validation.outliers` imports them from
  there (same public names).
- `preprocessing/recipe.py` imports the transformer classes where a pipeline
  is built, not at the top; `TransformerMixin` only for type checking.
- The package `__init__`s of `preprocessing`, `validation`, `models`,
  `modeling`, `prediction` load their names on first use, as `enginir` does
  (`__getattr__`, `__all__` and `__dir__` unchanged, type checkers see the
  names through `TYPE_CHECKING`).
- Test (`tests/test_startup.py`, a subprocess as for NumPy and pandas):
  importing `enginir.project`, `enginir._workflow`, `enginir._guidance`,
  `enginir.preprocessing.recipe` and `enginir.io` loads no `scipy` and no
  `sklearn`. Fault-injected: put one top import back.

## Phase 3: the main window without SciPy and scikit-learn

- The viewer modules import modeling code where it is used (fit, predict,
  diagnostics, band proposal, checks with limits), not at the top.
- `_warmup.py` in two stages:
  1. **window stage** (background, then the Qt modules on the UI thread as
     now): NumPy, h5py, hdf5plugin, matplotlib, engiNIR's light modules,
     pyqtgraph, `_open`, `_viewer`. A pick on the welcome window waits only
     for this stage.
  2. **modeling stage** (background, started when the window stage ends):
     SciPy, scikit-learn, `enginir.modeling`, `enginir.prediction`,
     `enginir.validation`, `_model_diagnostics`.
- One rule in `_warmup.py`: `when_modeling_ready(callback)` runs `callback`
  on the UI thread at once when the modeling stage is done, or when it
  ends; until then the main window's status line says "Loading the modeling
  libraries…". Every action that needs them goes through it (D2: the click
  is accepted and runs when loading is done).
- No two threads import the same packages (the 1.7.6 deadlock): while the
  modeling stage runs, the UI thread imports nothing of it, since every
  path to it goes through `when_modeling_ready`. Test: the window opens a
  project with a model, a calibration set and a stored prediction in a
  subprocess where SciPy and scikit-learn cannot be imported (a meta-path
  finder that refuses them); the window, the Project tree, Properties, the
  ROI table and the spectra work; Fit and Predict say they are loading.
  Fault-injected: one top import of `scipy.stats` back in a viewer module.
- The example project has a model: its Model page shows the model's stored
  metrics at once (they are read from the file); plots and checks that need
  SciPy fill in when the stage is done.
- The log adds one line: `Startup timing: modeling libraries ready … after
  the process started`.

## Decisions (2026-11-23)

- **D1** The log sink is added right after the welcome window is shown; the
  launcher's first lines are kept and written then.
- **D2** Fit, Predict and the model checks work while the modeling
  libraries load: the click is accepted, the status line says "Loading the
  modeling libraries…", and the action runs by itself when loading is done.

## Not in this plan

- A packed build (PyInstaller) or a native splash screen.
- An antivirus exclusion in the installer (needs admin rights, weakens the
  machine's protection; a hint in the installation guide at most).
- Leaving out matplotlib: the viewer's plot panels use it at the top; 3 s
  cold. A later plan if the next log shows it matters.

## Docs and contracts

`docs/viewer_architecture.md` (startup), `docs/user_guide/viewer.rst` (what
the status line says while the libraries load), viewer contract (startup
rule), CHANGELOG. The measured Windows times after the change go into this
plan.

## Built (2026-11-23)

As planned, with these findings on the way:

- The light set needed two more small modules: `modeling/_stored.py` (model
  type, stored hyperparameters, fold warning words) and
  `modeling/_sample_set.py` (`plan_sample_set`); both are re-exported from
  `modeling._service` and `enginir.modeling` under the same names.
- The library's computing modules import SciPy where they compute
  (`validation.outliers`, `prediction._uncertainty`, `preprocessing._pipeline`,
  the diagnostics' ellipse); modules that define scikit-learn estimators
  import it at the top as before.
- An import gate (`_warmup._ModelingGate`, `builtins.__import__` during the
  modeling stage only) makes any other thread wait for SciPy and
  scikit-learn; the lazy package names import through it. Worker jobs (ROI
  spectra, the band proposal) need nothing else.
- `from enginir.viewer import _warmup` loaded `_open`: the viewer package sent
  every unknown name there. Private submodules now import directly.
- Before the welcome window (Linux, counted): 164 modules instead of about 270.

## First Windows measurement (2026-11-23, cold) and what changed

Welcome window at 2.99 s (was 5.71 s), main window at 24.3 s (was 41.4 s).
SciPy then took 21 s in the background; scikit-learn had not finished when
the user looked at the model, whose plots waited for both ("models are never
shown"). Changed:

- The model's plots are drawn at once; only the ROIs' T² ellipse and the
  F-test's critical value need SciPy and are added when it is loaded.
- The gate covered only the modules the stage names. A thread importing
  another module that imports scikit-learn (`modeling._components`,
  `_exclusion`, `_gram_fit`, `models.*`, `_estimator`) would hold that
  module's lock while waiting, and the stage could wait for it: a deadlock
  until the 300 s timeout. The gate now covers every such module; a test
  recomputes the list from the source.
- A stage that has not ended after 90 s logs every thread's stack.

## Second Windows run (2026-11-23) and what changed

Welcome window at 3.54 s, main window at 26.8 s. Then the ROIs' spectra did not
show and choosing a dataset for the map froze the window; the stall report
showed why:

- The project's prediction checks (a worker, at the project's opening) began
  importing scikit-learn 0.6 s before the modeling stage installed its gate.
  The worker held scikit-learn's import lock, then waited at the gate; the
  stage waited for that lock. The gate is now installed when the warm-up
  starts; before the stage starts only the UI thread passes (it starts the
  stage). The window stage's job and timer belong to the application, so the
  stage always follows.
- Showing a dataset ran `PreprocessingRecipe.build_steps` on the UI thread,
  which imported every step's module even for an empty recipe, and waited at
  the gate. It now imports only the steps a recipe uses; with a recipe, the
  map shows the frame without it while the libraries load and redraws it after.
- The ROIs' spectra waited for scikit-learn though they need neither library:
  `roi_spectrum_summaries` and `roi_to_signal_domain` moved to the light
  `modeling._roi_spectra` (same public names).
