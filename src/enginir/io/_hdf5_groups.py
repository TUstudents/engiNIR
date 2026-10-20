"""Conversion between nested Python mappings and HDF5 groups.

Project objects are stored as one HDF5 group each: scalars and short numeric
vectors as attributes, arrays and string lists as datasets, nested mappings as
subgroups. ``None`` values of mapping keys are omitted; ``None`` inside a list
of strings is written as :data:`NONE_SENTINEL`. Lists that mix numbers with
``None`` are rejected. Empty lists are read back as empty arrays.
"""
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false
# h5py's stubs do not describe Group/Dataset item access.

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import h5py
import numpy as np

from enginir._checks import log_and_raise

NONE_SENTINEL = "__enginir_none__"
MAX_ATTRIBUTE_VECTOR = 16


def decode(value: Any) -> Any:
    """Convert values read from HDF5 to plain Python values.

    Byte strings become ``str``, string arrays become (nested) lists of
    ``str``, NumPy scalars become Python scalars and :data:`NONE_SENTINEL`
    becomes ``None``. Numeric arrays are returned unchanged.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SO":
            return np.vectorize(_decode_text, otypes=[object])(value).tolist()
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return None if isinstance(value, str) and value == NONE_SENTINEL else value


def _decode_text(item: Any) -> Any:
    text = item.decode("utf-8") if isinstance(item, bytes) else item
    return None if text == NONE_SENTINEL else text


def is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bool, int, float, np.generic))


def write_mapping(group: h5py.Group, payload: dict[str, Any], compression: str) -> None:
    """Write ``payload`` into ``group`` (see the module docstring for the layout)."""
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            write_mapping(group.create_group(key), value, compression)
        elif is_scalar_like(value):
            group.attrs[key] = value
        else:
            array = np.asarray(value)
            if array.ndim == 0:
                item = array.item()
                group.attrs[key] = NONE_SENTINEL if item is None else item
            elif array.dtype.kind in "UO":
                write_strings(group, key, _string_items(array, key))
            elif array.ndim == 1 and array.size <= MAX_ATTRIBUTE_VECTOR:
                group.attrs[key] = array
            else:
                group.create_dataset(key, data=array, compression=compression)


def _string_items(array: np.ndarray, key: str) -> list[str]:
    items = array.ravel().tolist()
    if not all(item is None or isinstance(item, str) for item in items):
        log_and_raise("Field '{}' mixes strings or None with other values; store numbers as a "
                      "numeric array", key, error=TypeError, depth=3)
    return [NONE_SENTINEL if item is None else item for item in items]


def write_strings(group: h5py.Group, name: str, values: list[str]) -> None:
    """Write ``values`` as a variable-length UTF-8 string dataset."""
    group.create_dataset(name, data=np.array(values, dtype=h5py.string_dtype()))


def read_mapping(
    group: h5py.Group,
    *,
    keys: set[str] | None = None,
    skip_datasets: tuple[str, ...] = (),
    metadata_only_groups: tuple[str, ...] = (),
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Read a group written by :func:`write_mapping` back into a dict.

    Parameters
    ----------
    group:
        Group to read.
    keys:
        If given, read only these top-level attributes, datasets and subgroups.
    skip_datasets:
        Dataset names not to read, at any depth.
    metadata_only_groups:
        Subgroups (by name, at any depth) of which only scalar attributes are
        read, so that large arrays stay on disk.
    metadata_only:
        Read only scalar attributes of ``group`` itself.
    """
    result: dict[str, Any] = {}
    for key, value in group.attrs.items():
        if keys is not None and key not in keys:
            continue
        decoded = decode(value)
        if metadata_only and isinstance(decoded, (np.ndarray, list)):
            continue
        result[key] = decoded
    for key, item in group.items():
        if (keys is not None and key not in keys) or key in skip_datasets:
            continue
        if isinstance(item, h5py.Group):
            result[key] = read_mapping(
                item,
                skip_datasets=skip_datasets,
                metadata_only_groups=metadata_only_groups,
                metadata_only=metadata_only or key in metadata_only_groups,
            )
        elif not metadata_only:
            result[key] = decode(item[()])
    return result


def values_equal(existing: Any, incoming: Any) -> bool:
    """Compare a stored payload with an incoming one by value (arrays, strings, nesting)."""
    if isinstance(existing, dict) or isinstance(incoming, dict):
        return (
            isinstance(existing, dict)
            and isinstance(incoming, dict)
            and existing.keys() == incoming.keys()
            and all(values_equal(existing[key], incoming[key]) for key in existing)
        )
    if existing is None or incoming is None:
        return existing is None and incoming is None
    existing_array, incoming_array = np.asarray(existing), np.asarray(incoming)
    if existing_array.shape != incoming_array.shape:
        return False
    if existing_array.dtype.kind in "USO" or incoming_array.dtype.kind in "USO":
        return bool(np.array_equal(existing_array.astype(str), incoming_array.astype(str)))
    # NaN is a stored value like any other: re-saving it unchanged is not a change.
    floating = existing_array.dtype.kind in "fc" and incoming_array.dtype.kind in "fc"
    return bool(np.array_equal(existing_array, incoming_array, equal_nan=floating))


@contextmanager
def replacement_group(parent: h5py.Group, name: str) -> Iterator[h5py.Group]:
    """Yield a temporary group that replaces ``parent[name]`` only if the block succeeds.

    A failed write therefore never leaves a half-written object behind.
    """
    file = parent.file
    temporary_name = f"__enginir_write_{uuid.uuid4().hex}"
    temporary = file.create_group(temporary_name)
    try:
        yield temporary
    except BaseException:
        del file[temporary_name]
        raise
    if name in parent:
        del parent[name]
    file.move(f"/{temporary_name}", f"{(parent.name or '').rstrip('/')}/{name}")


def child_group(parent: h5py.Group, name: str, *, create: bool = False) -> h5py.Group:
    """Return ``parent[name]`` as a group, creating it if ``create`` is set."""
    item = parent.get(name)
    if item is None and create:
        return parent.create_group(name)
    if not isinstance(item, h5py.Group):
        log_and_raise("Expected HDF5 group '{}' under {}", name, parent.name, error=KeyError,
                      depth=2)
    return item


__all__ = [
    "MAX_ATTRIBUTE_VECTOR",
    "NONE_SENTINEL",
    "child_group",
    "decode",
    "is_scalar_like",
    "read_mapping",
    "replacement_group",
    "values_equal",
    "write_mapping",
    "write_strings",
]
