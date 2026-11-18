"""Converting a project of the previous format ``enginir-1.5`` into ``enginir-1.7``.

In ``enginir-1.5`` the stored names of the reference correction said
"calibration" (``calibrations/``, ``calibration_id``, ``calibration_refs``,
``precalibrated_import``, ...), which clashed with model calibration. The
conversion renames them and changes nothing else (project-schema contract,
Compatibility policy, "Renamed in format enginir-1.7"):

- it writes a new file and never changes the source;
- it keeps ``project_id``, ``FileCreator``, ``created_at`` and every object
  id and value, and sets ``FileSchemaVersion``, ``FileWriter`` and
  ``updated_at``;
- it ends by reading the new file with the strict reader and removes it if
  that fails.

Command line: ``enginir-migrate SOURCE TARGET``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import h5py

from .._checks import log_and_raise
from .._logging import get_logger
from ._hdf5_groups import decode
from .project_store import (
    FILE_SCHEMA_VERSION,
    PREVIOUS_FILE_SCHEMA_VERSION,
    _now_iso,
    file_writer_label,
)

logger = get_logger(__name__)

#: Groups renamed wherever they occur (``datasets/{id}/calibrations``, ``rois/{id}/calibration_refs``).
RENAMED_GROUPS = {
    "calibrations": "reference_corrections",
    "calibration_refs": "reference_spectra",
}
#: Attributes renamed wherever they occur.
RENAMED_ATTRIBUTES = {
    "calibration_id": "reference_correction_id",
    "active_calibration_id": "active_reference_correction_id",
    "required_calibration_mode": "required_correction_mode",
    "calibrated_input_allowed": "corrected_input_allowed",
    "calibration_mode": "correction_mode",
}
#: Attribute values renamed (the reference correction mode and the input contract's required mode).
RENAMED_VALUES = {"precalibrated_import": "already_corrected"}


def migrate_project(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> Path:
    """Convert an ``enginir-1.5`` project into a new ``enginir-1.7`` file.

    Parameters
    ----------
    source : str or path-like
        A project written by engiNIR 1.5.1 to 1.6.1 (format ``enginir-1.5``).
        It is only read.
    target : str or path-like
        The new project file; it must not exist. Keep it in the source's
        folder (or move the data files with it): data files are found next
        to the project.

    Returns
    -------
    pathlib.Path
        ``target``.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    FileExistsError
        If ``target`` exists (or is ``source``).
    ValueError
        If ``source`` is not an ``enginir-1.5`` project, or the converted
        file does not pass the reader; nothing is left at ``target`` then.
    """
    source, target = Path(source), Path(target)
    if not source.is_file():
        log_and_raise("Not converted: {} does not exist", source, error=FileNotFoundError)
    if target.exists() or target.resolve() == source.resolve():
        log_and_raise("Not converted: {} exists; choose a new file name", target,
                      error=FileExistsError)
    found = _file_format(source)
    if found != PREVIOUS_FILE_SCHEMA_VERSION:
        log_and_raise("Not converted: {} is in format {!r}; only {!r} projects are converted to {!r}",
                      source, found, PREVIOUS_FILE_SCHEMA_VERSION, FILE_SCHEMA_VERSION)
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        shutil.copyfile(source, partial)
        counts = _rename(partial)
        _check(partial)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        logger.error("Conversion of {} to {} failed; nothing was written", source, target)
        raise
    logger.info("Converted {} ({}) to {} ({}): {} group(s), {} attribute(s), {} value(s) renamed",
                source, PREVIOUS_FILE_SCHEMA_VERSION, target, FILE_SCHEMA_VERSION,
                counts["groups"], counts["attributes"], counts["values"])
    return target


def _file_format(path: Path) -> str:
    try:
        with h5py.File(path, "r") as f:
            meta = f.get("project_meta")
            if not isinstance(meta, h5py.Group):
                return "none (not an engiNIR project)"
            attrs = meta.attrs
            return decode(attrs.get("FileSchemaVersion", attrs.get("schema_version", ""))) or "unknown"
    except OSError as exc:
        log_and_raise("Not converted: {} is not an HDF5 file ({})", path, exc)


def _rename(path: Path) -> Counter[str]:
    """Rename groups, attributes and values in place in ``path`` (the copy)."""
    counts: Counter[str] = Counter()
    with h5py.File(path, "r+") as f:
        names: list[str] = []
        f.visit(names.append)
        # Deepest first, so a renamed group never changes a path still to visit.
        for name in sorted(names, key=lambda n: n.count("/"), reverse=True):
            obj = f[name]
            for old, new in RENAMED_ATTRIBUTES.items():
                if old in obj.attrs:
                    obj.attrs[new] = obj.attrs[old]
                    del obj.attrs[old]
                    counts["attributes"] += 1
            for key, value in obj.attrs.items():
                if isinstance(value, (str, bytes)) and decode(value) in RENAMED_VALUES:
                    obj.attrs[key] = RENAMED_VALUES[decode(value)]
                    counts["values"] += 1
            parent, _, last = name.rpartition("/")
            owner = f[parent or "/"]
            if last in RENAMED_GROUPS and isinstance(obj, h5py.Group) and isinstance(owner, h5py.Group):
                owner.move(last, RENAMED_GROUPS[last])
                counts["groups"] += 1
        meta = f["project_meta"].attrs
        meta["FileSchemaVersion"] = FILE_SCHEMA_VERSION
        meta["FileWriter"] = file_writer_label()
        meta["updated_at"] = _now_iso()  # as every write
    return counts


def _check(path: Path) -> None:
    """Read the converted file as a project is opened: strict on the core."""
    from ..project import hydrate_project_store
    from .project_store import load_project_store

    hydrate_project_store(load_project_store(path))


def main(argv: list[str] | None = None) -> int:
    """``enginir-migrate SOURCE TARGET``: convert an engiNIR 1.5/1.6 project; 0 on success."""
    parser = argparse.ArgumentParser(
        prog="enginir-migrate",
        description=f"Convert an engiNIR project of format {PREVIOUS_FILE_SCHEMA_VERSION} "
                    f"(engiNIR 1.5.1 to 1.6.1) into a new file of format {FILE_SCHEMA_VERSION}. "
                    "The source is not changed.")
    parser.add_argument("source", help="the project to convert")
    parser.add_argument("target", help="the new project file (must not exist)")
    args = parser.parse_args(argv)
    try:
        migrate_project(args.source, args.target)
    except (OSError, ValueError) as exc:
        print(f"enginir-migrate: {exc}", file=sys.stderr)
        return 1
    print(f"Converted {args.source} to {args.target} (format {FILE_SCHEMA_VERSION}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
