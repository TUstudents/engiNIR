# Plan: engiNIR 1.5.2, the viewer on Windows

Status: approved (user, 2026-11-13) with W1–W5 as recommended; the script
is named `install-enginir-for-windows.ps1`. Phase 1 done (suite green on
Linux and Windows; 35c1d46). Phase 2 written (Windows CI job); it runs
first on the PR.
Branch `windows-1.5.2` from main (1.5.1, 561983b); one commit per phase,
one PR, then release 1.5.2.

## Goal

A PAT expert without a Python background installs and starts the viewer
on Windows the way they would any desktop program: one command copied from
the docs (or later a downloaded file), then a Start menu entry with the
engiNIR icon. No console window, no virtual environment, no `pip`.
Updating takes one command. GitHub CI tests every change on Windows, so a
release never breaks the Windows install unnoticed.

`uv tool install enginir` already works today: PyQt6 is a base dependency,
and `enginir-viewer` is a `[project.scripts]` entry. What's missing is
a launcher without a console, shortcuts, a one-line installer, a docs page
and Windows CI.

## Finding: the Windows test failures are real but almost all are test bugs

The full suite was re-run on Windows from a clean copy of main 561983b
(`git archive`, `uv sync --all-extras`, `QT_QPA_PLATFORM=offscreen`):
**6 failed, 1045 passed, 7 skipped**, the same six tests as before.
They are not stale. However, five of them are bugs in the tests, not
in the library:

| # | Test | Cause | Kind | Fix |
|---|---|---|---|---|
| 1 | `test_hdf5_project::test_current_project_ignores_legacy_extras_at_root_and_roi` | asserts `"/" in message` to mean "at the root"; the message has the project path, which has `\` on Windows | test | assert on `"top-level"` in the message instead of `/` |
| 2, 3 | `test_headless_api::test_viewer_writes_projects_only_through_project`, `…never_addresses_objects_by_name` | `path.read_text()` without encoding reads the sources (with `→`, `²`) as cp1252 | test | `read_text(encoding="utf-8")` |
| 4 | `test_logging::test_default_log_file_falls_back_to_tmp_when_home_is_unwritable` | the fake home `/nonexistent…` is `C:\nonexistent…` on Windows, and an admin may create it, so the home isn't unwritable | test **and** code | the code: the fallback `/tmp/enginir/enginir.log` becomes `C:\tmp\enginir\…` on Windows; use `tempfile.gettempdir()` (still `/tmp` on Linux unless `TMPDIR` is set). The test: make the home unwritable with a monkeypatched `mkdir`/sink failure instead of a fake path, and expect `gettempdir()` |
| 5 | `test_project_rois::test_the_scan_layout_keeps_positions_and_stacks_data_files` | the test builds its own file names with `source_path.split("/")`; the viewer uses `data_file_labels` (fixed in 1.5.1 via `stored_path().parts`) and shows the right names | test | take the names from `project_state.data_file_labels` (or `stored_path(...).name`) |
| 6 | `test_roi_table::test_enter_stores_and_moves_to_the_next_row` | on Windows the Return key reaches the editor one event round later; after one `processEvents()` the 0 ms timer that opens the next editor hasn't run yet. With more rounds the value is stored, the current cell is row 2 and the editor is open, so the behavior is correct | test | wait until `EditingState` with a deadline (loop `processEvents` up to 1 s) instead of one `processEvents()` |

So the only library fix is the log fallback (4). The library already passes
`encoding=` on every text read (checked with grep), and no `split("/")`
on paths remains outside `_data_paths.py` (where it splits the stored
POSIX-style `relative_path`, as intended).

Also found while checking (on Linux as well, not Windows-specific): kernel
PLS `_y_scores` divides by `q·q`, which is 0 for a zero-filled exhausted
component. That gives NaN Y scores and a `RuntimeWarning` (`pls.py:244`, seen
in `test_prediction_runs`, `test_viewer_shell`). It's a 1.5.1 regression of
the kernel engine. The fix: guard with `_EPS`, set U = 0 for such a
component, and log a warning (the conventions' division guard). Include it
in 1.5.2.

## Decisions (user, 2026-11-13: as recommended)

- **W1 Launcher names.** `enginir-viewer` becomes a
  `[project.gui-scripts]` entry, so on Windows it starts without a
  console window (on Linux and macOS, gui-scripts behave like scripts). A new
  `[project.scripts] enginir-viewer-console` keeps a console for
  diagnosis and `--help`. *Recommended.* Alternative: keep
  `enginir-viewer` as a console script and add a new windowless name;
  shortcuts would then point to the new name.
- **W2 Shortcuts without a new dependency.** `enginir-viewer
  --install-shortcut` / `--remove-shortcut` creates or removes
  "engiNIR Viewer" in the Start menu (and, with `--desktop`, on the
  desktop). It uses `IShellLinkW` + `IPropertyStore` through `ctypes`
  (about 100 lines in `viewer/_windows_shell.py`). This sets the target,
  the icon (`_resources/*.ico`) and the AppUserModelID
  `TUstudents.engiNIR.viewer` that the app sets at runtime; without
  the matching id, pinning the running window to the taskbar would pin
  `pythonw.exe`. *Recommended.* Alternative: `pywin32` as a
  Windows-only dependency (large; for this one use).
- **W3 Where the installer script lives.** `install-enginir-for-windows.ps1` is
  attached to each GitHub release as an asset, and the docs use
  `…/releases/latest/download/install-enginir-for-windows.ps1` (the script then always
  matches a published release, never an unreleased main). *Recommended.*
  Alternative: raw file from main.
- **W4 Python for the tool.** The script runs `uv tool install --python
  3.13 enginir`, so uv fetches a Python 3.13 that the user never sees
  and isn't affected by other Pythons on the machine. The Windows CI job
  tests 3.13. *Recommended.* Alternative: let uv choose (newest, now
  3.14).
- **W5 The Windows CI job is required from the start.** Phase 1 makes
  the suite green on Windows first, so the job doesn't need
  `continue-on-error`. Note that `publish.yml` accepts a release only if
  the whole CI run passed, Windows included. *Recommended.*

## Phases

### Phase 1: the suite is green on Windows

- Fix tests 1, 2, 3, 5 and 6 as in the table; for 4, the fix in
  `_logging.default_log_files` (`tempfile.gettempdir()`), its test, the
  docstring and `docs/user_guide/logging.rst`.
- The kernel PLS `_y_scores` guard, with a test (an exhausted component
  gives finite U = 0 and a warning; fails with the guard removed).
- Check: the full suite on Linux and on Windows (the copy in
  `%TEMP%\enginir-152`), plus the four checks on Linux.

### Phase 2: Windows CI

- `.github/workflows/ci.yml`: new job `windows` on `windows-latest`, Python
  3.13, `timeout-minutes: 30`, `QT_QPA_PLATFORM: offscreen`; steps
  checkout, setup-uv (same pinned version), `uv sync --all-extras`,
  `uv run python -X faulthandler -m pytest -q` (no `apt-get`, no GNU
  `timeout`; pytest-timeout and the job timeout cover hangs). Ruff,
  pyright and Sphinx stay on Linux only (same result everywhere).
- AGENTS.md: the CI line names the Windows job.
- Check: the job passes on the PR; time it (expected 6–10 min, in
  parallel with Linux).

### Phase 3: launcher and shortcuts (W1, W2)

- `pyproject.toml`: `[project.gui-scripts] enginir-viewer`,
  `[project.scripts] enginir-viewer-console` (both
  `enginir.viewer.__main__:main`).
- No console means `sys.stderr` is `None` under `pythonw`, and
  `enable_logging` adds a stderr sink. Only add it when `sys.stderr` is not
  `None`; the log file still gets everything. A crash at start then shows
  a message box with the log file path (the viewer already logs uncaught
  errors with `catch_uncaught=True`; the launcher's `except` in
  `main` also needs the message box, because nobody sees the exit code).
- `--version` (prints the version; in the GUI launcher a message box).
- `viewer/_windows_shell.py`: create, read and remove a `.lnk` with
  target, arguments, icon and AppUserModelID; Start menu
  (`FOLDERID_Programs`) and desktop (`FOLDERID_Desktop`) via
  `SHGetKnownFolderPath`. The target is the `enginir-viewer.exe` next to
  `sys.executable`'s tool environment, found with `shutil.which` and checked
  to exist. On other platforms the options exit with a clear message (a
  Linux `.desktop` entry stays a later item).
- Tests (Windows-only, `skipif` elsewhere): a shortcut written to a
  temp folder reads back with the same target, icon and AppUserModelID;
  remove deletes only our shortcut. Both fail when the id is not written.
  A Linux test that the options refuse cleanly off Windows.
- Docs: `docs/api/viewer.rst`, `docs/user_guide/viewer.rst`,
  `docs/viewer_architecture.md` (kept in sync).

### Phase 4: one-line installer and docs page (W3, W4)

- `scripts/install-enginir-for-windows.ps1`:
  1. installs uv with Astral's official installer if `uv` is missing;
  2. `uv tool install --python 3.13 --upgrade enginir`;
  3. `uv tool update-shell` (PATH for `enginir-viewer-console`);
  4. `enginir-viewer --install-shortcut`;
  5. prints what it did and how to start, update and uninstall.

  It changes nothing outside the user profile and needs no admin rights.
- `docs/user_guide/installation.rst`: a section "Install the viewer on
  Windows" first, before the developer installs. It covers the one-liner
  (`powershell -ExecutionPolicy ByPass -c "irm
  https://github.com/TUstudents/engiNIR/releases/latest/download/install-enginir-for-windows.ps1
  | iex"`), what it does, the manual equivalent (three commands), updating
  (`uv tool upgrade enginir`), uninstalling (`enginir-viewer
  --remove-shortcut`, then `uv tool uninstall enginir`), and where the log
  file is. README: a short pointer.
- The release step in AGENTS.md gains: attach
  `scripts/install-enginir-for-windows.ps1` to the GitHub release.

### Phase 5: install smoke test in CI

- In the `windows` job, after the tests:
  1. `uv build`;
  2. install the wheel as a tool into a temporary `UV_TOOL_DIR` /
     `UV_TOOL_BIN_DIR`;
  3. `enginir-viewer-console --version` must print the version;
  4. `--install-shortcut` creates the Start menu `.lnk` (the runner is a
     throwaway VM) and `--remove-shortcut` removes it;
  5. run `install-enginir-for-windows.ps1` against the local wheel (a `-Wheel`
     parameter for testing) to check the script itself.
- This catches a broken entry point, a missing icon in the wheel or a
  broken script before the release.

### Phase 6: release 1.5.2

- `pyproject.toml` 1.5.2, `CHANGELOG.md` entry, the four checks, PR,
  merge, tag `v1.5.2`, CI on the tag (Linux and Windows), GitHub release
  with `install-enginir-for-windows.ps1` attached, PyPI via publish.yml, Zenodo DOI.
- After the release, on the user's Windows machine: run the one-liner
  from the docs as a novice would, start from the Start menu, pin to the
  taskbar, check the icon, update, uninstall.

## Out of scope (later)

- A signed `.msi`/`.exe` installer (needs a code-signing certificate;
  unsigned ones trigger SmartScreen, which is worse for novices than a
  PowerShell line).
- Opening `.h5` project files by double-click (file association).
- Linux `.desktop` entry and icon for WSLg.
- Phase 6 GUI polish of the 1.5.x plan, and the kernel leftovers (MSC/EMSC on
  the sums, one diagnostics pass, fit progress naming its phase).
