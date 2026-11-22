# Plan: a faster cold start on Windows

Status: phases 0 to 2 built on branch `cold-start` (2026-11-22); measured on a
second Windows machine (below); a start after a reboot on the work laptop is
open.

## The problem (log of 2026-11-22, Windows, installed with the 1.7.4 installer)

| | warm (20:44) | cold (22:40) |
|---|---|---|
| Python running before engiNIR's launcher code | 0.5 s | **17.6 s** |
| Welcome window shown | 0.8 s | 19.3 s |
| Warm-up (the scientific stack, in the background) | 3.7 s | **31.3 s** |
| Main window after picking the example | — | 50.6 s |

The same code runs ten times slower cold: the time is reading files from a cold
disk while the antivirus scans them, not computing. The installer already
compiles bytecode (`--compile-bytecode`), so nothing is compiled at the start.
A pick on the welcome window waits for the warm-up (`wait_for_library`), so the
31 s are felt in full.

What the warm-up loads (measured on Linux, same packages): 2,125 modules, about
1,800 `.pyc` files and 241 compiled extensions (144 MB), plus about 60 MB of
bundled math libraries (`numpy.libs`, `scipy.libs`). By package: SciPy 534,
pandas 293, scikit-learn 278, pyqtgraph 145, NumPy 144, engiNIR 101,
matplotlib 100, narwhals 56, h5py 40.

Which share costs most on Windows (many small files or large native files) is
not known. That decides whether packing files (a zip, a PyInstaller build) can
help at all, since native files cannot be packed. This plan measures it and
takes the one cheap cut that is certain.

## Phase 0 (done, not committed): no NumPy before the welcome window

The launcher imported NumPy through `viewer/_theme.py` (one `np.linspace`) and
through `viewer/_windows_shell.py` (`log_and_raise` from `enginir._checks`,
which imports NumPy). Both import it on use now; the launcher loads 264 modules
instead of 349. `tests/test_startup.py` lists `numpy` in `HEAVY` (fails when
the import is put back; checked).

## Phase 1: pandas no longer a dependency

**Finding:** making `io/csv_io.py` import pandas lazily saves nothing:
scikit-learn imports pandas whenever it is installed
(`sklearn/utils/fixes.py`, `try: import pandas`). pandas is out of the start
only if it is not installed. Nothing else needs it (`uv tree --invert
--package pandas`: only engiNIR).

Uses of pandas in engiNIR:

- `load_csv`: `pd.read_csv` and `pd.read_excel`.
- `save_csv`: `pd.DataFrame(...).to_csv`.
- `metrics_report(..., as_frame=True)`: returns a one-row DataFrame (already a
  lazy import).

Steps:

1. `load_csv` reads CSV with the standard `csv` module and Excel with
   `openpyxl` (already a dependency; pandas uses it for `.xlsx` anyway) into
   the same table of header and rows; the rest of the function (sample column,
   text columns, numbers) works on that table. Same results, same messages.
   Number parsing as pandas did it for our files: empty cell gives NaN, a cell
   that is not a number refuses with the existing message.
2. `save_csv` writes with `csv`: the same header and the same number format as
   now (checked byte for byte against the file pandas writes, in the test).
3. `as_frame=True` stays, needs pandas, and says so when it is missing
   (`ImportError` naming `pip install pandas`, logged). Documented in the user
   guide (validation) and the API docs.
4. `pyproject.toml`: pandas leaves `dependencies`; an extra `pandas`
   (`enginir[pandas]`), and in the dev group, so the tests of `as_frame` run.
5. Tests: the existing CSV and Excel tests in `tests/test_io.py` keep passing
   unchanged (they are the oracle); one round-trip test against a file written
   by pandas (`pytest.importorskip("pandas")`), fault-injected. `test_startup`
   adds `pandas` to the modules not loaded by the warm-up of an install without
   pandas (a subprocess with pandas blocked by a meta-path finder).
6. Docs: installation (the extra), CHANGELOG, `viewer_architecture.md`
   (startup section).

Gain: 293 pandas modules and its native files (49 MB) leave every start, about
14 % of the files the warm-up opens. narwhals stays (scikit-learn needs it).

Risk: users who relied on pandas arriving with engiNIR. The CHANGELOG says so.

## Phase 2: the warm-up logs where its time goes

The warm-up imports the packages one by one in dependency order before the
engiNIR modules, and logs each at INFO:

    Startup timing: warm-up imported scipy in 9.12s (534 modules, 61 native, 41 MB native)

Order: `numpy`, `scipy`, `sklearn`, `h5py`, `matplotlib`, then the existing
`BACKGROUND_MODULES` and `UI_MODULES` (`pyqtgraph`, Qt backends, the viewer).
Each line counts the modules it added to `sys.modules`, how many of them are
native (`.pyd`/`.so`), and the size of those files. The order makes each time
exclusive (SciPy's time does not include NumPy). The same order as today's
imports, so the total does not change.

The launcher also logs how many modules were loaded when it starts
(`mark_start`), so the 17.6 s before it can be compared with the modules that
cause them.

Tests: one test that a warm-up logs a line per package with a module count
(the format is not pinned). No timing assertions.

Then on Windows: one cold start (after a reboot) and one warm start; record the
lines in `plans/backlog.md` item 17. From them:

- most time in packages with many small files and few native ones (SciPy,
  scikit-learn): a packed build (PyInstaller one-dir) is worth a plan, code
  signing first;
- most time in the native files (NumPy, SciPy's libraries, Qt): no packing
  helps; the answer is an antivirus exclusion of the uv folders and a local
  disk, which goes into the installation guide.

## Measured on Windows (2026-11-22, a second machine, branch `cold-start`)

A fresh install (`uv tool install --link-mode copy --no-cache`: files never
read or scanned before), then two more starts. Not a start after a reboot; the
work laptop's cold start (1.7.4) was slower again: 17.6 s before the launcher,
31.3 s warm-up.

| step | modules | native | native MB | fresh | warm |
|---|---|---|---|---|---|
| launcher (before the welcome window) | 269 | | | 1.53 s | 0.65 s |
| numpy | 84 | 2 | 4 | 1.00 s | 0.06 s |
| scipy | 627 | 104 | 67 | **12.23 s** | 0.80 s |
| scikit-learn | 208 | 8 | 1 | 1.35 s | 0.14 s |
| h5py | 41 | 25 | 3 | 0.56 s | 0.05 s |
| matplotlib | 176 | 10 | 8 | 1.28 s | 0.30 s |
| engiNIR | 251 | 46 | 7 | 2.37 s | 0.45 s |
| pyqtgraph and the viewer | 212 | 4 | 1 | 1.22 s | 0.16 s |
| **warm-up total** | | | | **20.1 s** | **2.0 s** |

Reading: steps of mostly small files cost about 6 ms per module fresh
(scikit-learn 1.35 s for 208, pyqtgraph 1.06 s for 179). With that rate,
SciPy's 627 modules explain about 4 s of its 12.2 s; the other 8 s are its
67 MB of native files (about 0.12 s per MB). Over the whole warm-up, roughly
half the time is small files (about 1,600 modules) and half native files
(about 90 MB, three quarters of it SciPy's).

So a packed build (PyInstaller) could save at most about half; SciPy's native
files remain, and scikit-learn imports `scipy.stats` (most of SciPy) itself.
The rest is the antivirus and the disk.

The first try of this measurement found a bug: Open Example Project, clicked
during the warm-up, imported the library in a second thread at the same time
(it did not call `wait_for_library`); fixed, with a test for every welcome
pick.

## Not in this plan

- A zip of `site-packages` by the installer: matplotlib, scikit-learn and
  pyqtgraph read data files by path; fragile.
- A PyInstaller or Nuitka build: waits for phase 2's numbers.
- Lazy scikit-learn: the workflow and the checks need it to open a project.

## Order

Phase 0 and 2 first (one change, no dependency change), then the Windows
measurement, then phase 1 (a dependency change for users; its own CHANGELOG
entry).
