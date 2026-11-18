# Plan: viewer windows are deleted when they close (engiNIR 1.7.0rc2)

Status: decided (user, 2026-11-18): D1 to D5 below. Implemented on branch
`window-lifetime` (df1e7db code and tests, 8a02ebf docs and contract);
Phase 2 needed no code change (see the commit). Not yet released.

## Goal

A viewer window that the user closes is gone: Qt deletes it at a safe point
of the event loop, it no longer holds data, and nothing calls it any more.
This fixes a memory leak, wasted project reloads, and crashes that happen at
random moments. There is no change to the format or the API; it ships with
the format change of 1.7.0, as its second release candidate.

## What happens today (measured 2026-11-18, on `011d641`)

No window sets `WA_DeleteOnClose`, so closing a window only hides it. The
window and everything it owns live on for as long as Python holds a
reference to it:

1. **A closed child window keeps its data.** **Open Data...** in a window
   without a project opens a new window with `open_viewer` and keeps it in
   the parent's `_child_viewers`
   (`viewer/_viewer.py`, `_track_child_viewer`). The window is removed from
   that list only on `destroyed`, which never fires for a window that was
   merely closed. Each closed child keeps its cube, possibly gigabytes, until
   the parent is freed.
2. **A closed window keeps reloading its project.** Every `ProjectState`
   joins `_OPEN_STATES` (`viewer/_state.py`), a `WeakSet` that windows on
   the same project use to refresh each other after a write. A closed but
   live window stays in it, so every write in another window starts a full
   `reload()` of a window nobody sees.
3. **Windows are freed at random moments.** When the last Python reference
   goes away, the window's signal connections form reference cycles, so
   Python's cyclic garbage collector frees it whenever it runs. That can be
   while Qt paints another window. One example of a lost reference: the
   app's single `_viewer_ref` in `viewer/__main__.py`. CI crashed now and
   then that way (segmentation fault in `QPainter::restore`); the test
   fixture `_windows_are_destroyed_after_the_test` (`tests/conftest.py`)
   works around it in the tests only.
4. **The welcome window is never deleted either.** It is closed after the
   hand-off to the first viewer (`__main__.py`) and stays alive, together
   with the relay of its last background job.
5. **Deferred calls can reach a deleted object.** The macOS teardown error of
   2026-11-18 ("wrapped C/C++ object of type QLabel has been deleted", in a
   Qt slot) is this kind of call. The candidates are:
   - four `QTimer.singleShot` calls (`_browser.py`, `_app.py` with a
     lambda, and two in `_viewer.py`);
   - signals of long-lived objects (the theme, the preferences, another
     window's `ProjectState`) connected to a window's methods or to lambdas
     that hold a window.

Already safe: background jobs (`viewer/_jobs.py`). Their callback relay is a
child of the window, so Qt disconnects it when the window is deleted, and
`_RUNNING_JOBS` keeps the thread alive until it finishes.

Opening a project inside a window replaces that window's content
(`_open_project_btn_clicked_with_path`); no new window is involved.

## Decisions (user, 2026-11-18)

**D1. Release.** Part of engiNIR 1.7.0: released as **`1.7.0rc2`**, so the
final 1.7.0 has the format change and the window lifetime together, both
tested by users in a release candidate. Branch `window-lifetime` from
`main`, one commit per phase, and a draft PR, as for format 1.7.

**D2. Which windows.** `HSIViewer` and `WelcomeWindow`, the two top-level
main windows. Dialogs and docks have a parent, so they are deleted with it;
the dialogs that call `deleteLater` today keep doing so.

**D3. Keeping windows alive.** One app-level registry (`viewer/_windows.py`,
a `set` of open top-level windows). Every window joins it when it is shown
and leaves it on `destroyed`. It replaces `__main__`'s `_viewer_ref` and
each window's `_child_viewers`; ownership by a parent window was never
meant (closing a parent does not close its children).

**D4. The test fixture.** Keep `_windows_are_destroyed_after_the_test`, with
its docstring saying that it is a safety net now. It still frees widgets
that a test creates without showing or closing them.

**D5. Contract.** One rule in `viewer-workspace-ui-contract_v3.md` (window
lifetime): a closed window is deleted, keeps no data and receives no
refresh, and a job that finishes after its window closed calls nothing.
Gap `G37` opens in `contracts/coverage.md` and closes with Phase 4.

## Phases

### Phase 0: tests that show the faults (red first)

Each test fails on today's code, for the reason given in its name:

- a closed window is freed: a weak reference to the `HSIViewer` is dead
  after `QApplication.sendPostedEvents(None, DeferredDelete)`;
- a closed child window releases its cube: a weak reference to the cube
  array is dead after the child closes (the parent stays open);
- a closed window stops reloading: after it closes, a write in another
  window on the same project does not call its `reload` (count calls);
- the welcome window is freed after the hand-off.

### Phase 1: delete on close, with cleanup

- `HSIViewer.__init__` and `WelcomeWindow.__init__` set
  `Qt.WidgetAttribute.WA_DeleteOnClose`.
- `HSIViewer.closeEvent`, before deletion, in this order:
  1. `ProjectState.close()` (new) leaves `_OPEN_STATES` and stops its
     reload timer;
  2. the viewer state drops the cube and its memory map or file handle;
  3. the log sink is removed, as it is today.

  Every step is idempotent, because `closeEvent` can run again on an
  application quit.
- The registry of D3 replaces `_viewer_ref` and `_child_viewers`.

### Phase 2: no call after deletion

- `QTimer.singleShot`: use a context object, so that Qt drops the call when
  the window is gone. If PyQt6 does not take the bound method's object as the
  context, use a single-shot `QTimer` that is a child of the window.
- Audit `.connect(` on long-lived senders (`_theme`, `_preferences`, other
  windows' `ProjectState`, `QApplication`): connect bound methods of the
  window, not lambdas that hold it, or disconnect them in `closeEvent`.
- Test: a window closed while a job runs gets no callback, raises no
  `RuntimeError` in a Qt slot (the conftest check fails the test on one),
  and the job's thread still finishes.

### Phase 3: adapt the tests that use a closed window

Tests that close a window and then read from it are changed to read before
closing, or to check that the window is gone. A test that closes a window to
reach a state (for example a second window on the same project) keeps
working through the registry.

### Phase 4: docs, contract, release

- `docs/viewer_architecture.md`: a "Window lifetime" section (registry,
  delete on close, the cleanup order, no deferred call to a closed window).
  `docs/user_guide/viewer.rst` and `docs/api/viewer.rst` change only if
  they mention closing windows.
- The contract rule of D5; close `G37` in the ledger.
- CHANGELOG `1.7.0rc2` (Fixed): memory released when a window closes, no
  reloads in closed windows, no crash from windows freed at random moments.
  The final 1.7.0 entry later merges rc1 and rc2.
- All four gates, and the full CI run including macOS and the Windows
  install check. Then version `1.7.0rc2`, tag `v1.7.0rc2` on `main` and a
  GitHub pre-release, as `AGENTS.md` describes for release candidates.

## Acceptance

- The four Phase 0 tests pass. When `WA_DeleteOnClose`, the
  `_OPEN_STATES` removal or the cube release is taken out again, the test
  for that fault fails (fault injection).
- The Phase 2 test passes, and fails with a lambda put back on a long-lived
  signal.
- 50 consecutive full suite runs in CI (a manual workflow dispatch on the
  branch) without a segmentation fault or hang, compared with the
  occasional crash before.

## Risks

- **pyqtgraph-owned widgets.** Menu widgets belong to a `QWidgetAction`.
  Deleting the window deletes them through Qt's ownership, which is correct;
  Python code must not delete them as well (the reason for the conftest's
  `sip.ispyowned` rule). Phase 0 and the 50-run check cover it.
- **Hidden uses of a closed window.** Code that keeps a window to reuse it
  after close (for example "show again"). The audit in Phase 2 and the
  Phase 3 test changes find these.
- **Quit order.** With the last window deleted, the application quits as
  today (`quitOnLastWindowClosed`). `wait_for_running_jobs` still runs
  before the interpreter exits.

## Out of scope

- The intermittent test hang in garbage collection (Python 3.12, once; and
  once locally on 2026-11-18). Random-moment frees may explain it; if it
  persists after this plan, it gets its own investigation.
- Any change to how many windows exist or what opens one.
