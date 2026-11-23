"""Project files store their arrays with Blosc."""

from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np
import pytest

BLOSC = "32001"  # the HDF5 filter id of Blosc


def _arrays(path) -> dict[str, h5py.Dataset]:
    found: dict[str, tuple[dict, np.dtype]] = {}

    def visit(name: str, obj: object) -> None:  # returns None: visititems goes on
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind not in "OSU":
            found[name] = (dict(obj._filters), obj.dtype)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found


@pytest.fixture()
def example(tmp_path):
    from enginir._example import create_example_project

    return create_example_project(tmp_path)


def test_a_new_project_stores_every_array_with_blosc(example):
    arrays = _arrays(example)
    assert len(arrays) > 50, "pixels, references, model artifacts, prediction ROIs"
    other = {name: filters for name, (filters, _dtype) in arrays.items() if BLOSC not in filters}
    assert other == {}, f"not Blosc: {sorted(other)[:5]}"


def test_a_project_with_gzip_arrays_opens_and_predicts_the_same(example):
    """Projects written before 1.7.3 hold gzip arrays; old and new arrays mix in a file."""
    from enginir import Project

    def predicted(path):
        model = next(iter(Project.open(path).models.values()))
        return [roi.mean for roi in model.predict_rois().rois]

    before = predicted(example)
    with h5py.File(example, "r+") as f:
        names: list[str] = []
        f.visititems(lambda name, obj: names.append(name)
                     if isinstance(obj, h5py.Dataset) and obj.dtype.kind not in "OSU" else None)
        for name in names[::2]:  # every second array back to gzip
            data, attrs = f[name][()], dict(f[name].attrs)
            del f[name]
            f.create_dataset(name, data=data, compression="gzip")
            f[name].attrs.update(attrs)
    kinds = {BLOSC in filters for filters, _dtype in _arrays(example).values()}
    assert kinds == {True, False}, "both kinds in one file"
    assert predicted(example) == pytest.approx(before)


def test_the_project_reader_alone_decodes_blosc(example):
    """No caller must import hdf5plugin first: the project reader registers the filters."""
    script = ("import sys\nfrom enginir.io.project_store import load_roi_spectra_batch\n"
              "import h5py\n"
              "with h5py.File(sys.argv[1], 'r') as f:\n"
              "    key = '/rois/' + next(iter(f['rois'])) + '/spectra'\n"
              "print(load_roi_spectra_batch(sys.argv[1], [key])[key].shape)")
    result = subprocess.run([sys.executable, "-c", script, str(example)], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().startswith("(")


def test_save_hdf5_keeps_gzip_and_offers_blosc(tmp_path):
    from enginir.io import load_hdf5, save_hdf5

    cube = np.arange(2 * 3 * 5, dtype=np.uint16).reshape(2, 3, 5)
    for compression, expected in (("gzip", "gzip"), ("blosc", BLOSC)):
        path = tmp_path / f"{compression}.h5"
        if compression == "gzip":
            save_hdf5(str(path), cube, np.linspace(1000, 1100, 5))  # the default
        else:
            save_hdf5(str(path), cube, np.linspace(1000, 1100, 5), compression=compression)
        with h5py.File(path, "r") as f:
            assert expected in dict(f["spectra"]._filters)
        np.testing.assert_array_equal(load_hdf5(str(path)).X, cube)
