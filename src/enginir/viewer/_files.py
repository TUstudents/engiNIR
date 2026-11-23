"""Telling a project file from a data file, without loading the viewer or the library."""

from __future__ import annotations

from pathlib import Path


def _is_project_file(path: str | Path | None) -> bool:
    """Return True when *path* is an engiNIR project file of any format.

    A project in an earlier format goes to the project path, which refuses it
    with a message naming its format; it is never opened as a dataset.
    """
    if path is None:
        return False
    candidate = Path(path)
    if candidate.suffix.lower() not in {".h5", ".hdf5"} or not candidate.exists():
        return False
    # The warm-up may be importing h5py in the background: two threads importing the same
    # package at once can deadlock (the recent-files check is a job; a double click on
    # the welcome window runs here on the UI thread).
    from ._warmup import wait_for_library

    wait_for_library()
    try:
        import h5py  # h5py alone: enginir.io (pandas and every reader) is not needed here

        with h5py.File(candidate, "r") as handle:
            # as enginir.io.project_store.has_project_meta: every project has /project_meta
            return isinstance(handle.get("project_meta"), h5py.Group)
    except OSError:
        return False
