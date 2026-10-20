"""Contract tests for enginir.io readers and writers."""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from ._project_payloads import roi_provenance


def test_csv_round_trip_preserves_sample_names_and_dispatch(tmp_path):
    from enginir.io import SpectralData, load_spectra, save_csv

    rng = np.random.default_rng(1)
    X = rng.random((4, 5))
    wavelengths = np.linspace(1000.0, 1200.0, 5)
    path = tmp_path / "spectra.csv"

    save_csv(str(path), X, wavelengths, sample_names=["a", "b", "c", "d"])
    data = load_spectra(str(path), sample_col="sample_id")

    assert isinstance(data, SpectralData)
    np.testing.assert_allclose(data.X, X)
    np.testing.assert_allclose(data.wavelengths, wavelengths)
    assert data.meta["sample_names"] == ["a", "b", "c", "d"]


def test_load_csv_rejects_unknown_sample_column(tmp_path):
    from enginir.io import load_csv

    path = tmp_path / "spectra.csv"
    path.write_text("sample_id,1000,1001\na,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample_col 'missing' is not a column"):
        load_csv(path, sample_col="missing")


def test_load_spectra_rejects_unknown_extension(tmp_path):
    from enginir.io import load_spectra

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_spectra(str(tmp_path / "data.xyz"))


def test_hdf5_round_trip_and_metadata(tmp_path):
    from enginir.io import SpectralData, load_hdf5, save_hdf5

    X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    wavelengths = np.linspace(900.0, 1200.0, 4)
    path = tmp_path / "spectra.h5"

    save_hdf5(str(path), X, wavelengths, meta={"instrument": "alpha", "ids": ["a", "b"]})
    data = load_hdf5(str(path))

    assert isinstance(data, SpectralData)
    np.testing.assert_allclose(data.X, X)
    np.testing.assert_allclose(data.wavelengths, wavelengths)
    assert data.meta["instrument"] == "alpha"
    assert list(data.meta["ids"]) == ["a", "b"]
    assert data.meta["hdf5_schema"] == "enginir_simple_spectral_hdf5"
    assert data.meta["hdf5_spectra_key"] == "spectra"
    assert data.meta["x_units"] == "nm"


def test_load_hdf5_supports_frame_slices_and_header_only(tmp_path):
    from enginir.io import load_hdf5, save_hdf5

    X = np.arange(60, dtype=np.uint16).reshape(5, 3, 4)
    wavelengths = np.linspace(900.0, 1200.0, 4)
    path = tmp_path / "chunkable.h5"

    save_hdf5(str(path), X, wavelengths)

    sliced = load_hdf5(str(path), frames=slice(1, 4))
    header = load_hdf5(str(path), frames=slice(1, 4), header_only=True)

    np.testing.assert_array_equal(sliced.X, X[1:4])
    assert header.X.shape == (0, 3, 4)
    assert header.meta["source_shape"] == (3, 3, 4)
    assert header.meta["frames_loaded"] == 0
    assert header.meta["frames_total"] == 5


def test_load_hdf5_missing_wavelengths_falls_back_to_band_indices(tmp_path):
    import h5py

    from enginir.io import load_hdf5

    cube = np.ones((5, 10, 20), dtype=np.uint16)
    path = tmp_path / "no_wavelengths.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=cube)

    data = load_hdf5(str(path))

    np.testing.assert_array_equal(data.wavelengths, np.arange(20, dtype=float))
    assert data.meta["hdf5_schema"] == "kestreleye_raw_hdf5"
    assert data.meta["hdf5_spectra_key"] == "data"
    assert data.meta["x_units"] == "band_index"


def test_load_hdf5_supports_blosc_compressed_data_dataset(tmp_path):
    import h5py
    import hdf5plugin

    from enginir.io import load_spectra

    cube = np.arange(5 * 10 * 20, dtype=np.uint16).reshape(5, 10, 20)
    path = tmp_path / "kestreleye_like.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data",
            data=cube,
            chunks=(1, 10, 20),
            **hdf5plugin.Blosc(),
        )

    data = load_spectra(str(path), frames=slice(1, 3))

    np.testing.assert_array_equal(data.X, cube[1:3])
    np.testing.assert_array_equal(data.wavelengths, np.arange(20, dtype=float))
    assert data.meta["hdf5_schema"] == "kestreleye_raw_hdf5"
    assert data.meta["hdf5_spectra_key"] == "data"
    assert data.meta["x_units"] == "band_index"


def test_project_store_round_trip_keeps_v3_workspace_objects(tmp_path):
    from enginir.io import (
        create_project,
        load_project_store,
        load_roi_spectra,
        save_calibration_spec,
        save_dataset_ref,
        save_training_roi,
    )

    path = tmp_path / "project.h5"
    wavelengths = np.linspace(900.0, 1700.0, 6, dtype=np.float32)
    spectra = np.arange(24, dtype=np.uint16).reshape(4, 6)
    create_project(str(path))
    save_dataset_ref(
        str(path),
        "dataset_1",
        {
            "source_path": "/tmp/sample.eiff",
            "source_format": "eiff",
            "source_modality": "nir_hsi",
            "cube_shape": (2, 2, 6),
            "n_bands": 6,
            "wavelengths": wavelengths,
            "loader_metadata": {"instrument": "alpha"},
            "import_timestamp": "2026-01-01T00:00:00",
        },
    )
    save_calibration_spec(
        str(path),
        "dataset_1",
        "cal_1",
        {
            "mode": "raw_dn",
            "dark_ref_source": None,
            "white_ref_source": None,
            "dark_ref_data": None,
            "white_ref_data": None,
            "black_level": None,
            "white_level": None,
            "output_signal_unit": "DN",
            "created_at": "2026-01-01T00:00:00",
        },
    )
    save_training_roi(
        str(path),
        "roi_a",
        {
            "dataset_id": "dataset_1",
            "calibration_id": "cal_1",
            "name": "roi_a",
            "wavelengths": wavelengths,
            "source_file": "/tmp/sample.eiff",
            "spatial_bounds": (0, 1, 0, 1),
            "n_pixels": 4,
            "created_at": "2026-01-01T00:00:00",
            "provenance": roi_provenance(),
        },
        spectra,
    )

    store = load_project_store(str(path))
    loaded_spectra = load_roi_spectra(str(path), "/rois/roi_a/spectra")

    assert store["project_meta"]["schema_version"] == "viewer-v3"
    assert set(store["datasets"]) == {"dataset_1"}
    assert set(store["rois"]) == {"roi_a"}
    assert "spectra" not in store["rois"]["roi_a"]
    np.testing.assert_array_equal(loaded_spectra, spectra)


def _pad10(name: str) -> bytes:
    return name.encode("ascii").ljust(10)


def _chunk(name: str, payload: bytes) -> bytes:
    return _pad10(name) + struct.pack(">Q", len(payload)) + payload


def _group(keyword: str, content_type: str, children: bytes) -> bytes:
    inner = _pad10(content_type) + children
    return _pad10(keyword) + struct.pack(">Q", len(inner)) + inner


def _compress_envelope(data: bytes) -> bytes:
    return struct.pack("<I", 2) + struct.pack("<I", len(data)) + b"\x01" + zlib.compress(data)


def _build_stnrd(width: int, height: int, count: int, compression: int = 2) -> bytes:
    buf = bytearray(54)
    struct.pack_into(">H", buf, 0, 0x0100)
    buf[2:6] = b"\xff\xff\xff\xf8"
    struct.pack_into(">H", buf, 8, width)
    struct.pack_into(">H", buf, 12, height)
    struct.pack_into(">H", buf, 14, 2)
    struct.pack_into(">H", buf, 16, compression)
    struct.pack_into(">H", buf, 18, 12)
    struct.pack_into(">H", buf, 20, 1)
    struct.pack_into(">H", buf, 22, 1)
    struct.pack_into(">H", buf, 28, count)
    struct.pack_into(">H", buf, 30, 2025)
    buf[32] = 6
    buf[33] = 3
    return bytes(buf)


def _build_wavtb(wavelengths_nm: np.ndarray) -> bytes:
    header = struct.pack("<4H", 1, 1, 0, len(wavelengths_nm))
    values = (wavelengths_nm / 0.025).astype(np.uint32).tobytes()
    return header + values


def _build_eiff_file(
    width: int,
    height: int,
    frames: list[np.ndarray],
    wavelengths_nm: np.ndarray | None = None,
) -> bytes:
    prop_stnrd = _group(
        "PROP", "EVKSTNRD", _chunk("EVKSTNRD", _build_stnrd(width, height, len(frames)))
    )
    prop_wavtb = b""
    if wavelengths_nm is not None:
        propinfo = _chunk("PROPINFO", b"\x01\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00")
        wavtb_chunk = _chunk("EVKWAVTB", _compress_envelope(_build_wavtb(wavelengths_nm)))
        prop_wavtb = _group("PROP", "EVKWAVTB", propinfo + wavtb_chunk)

    frame_groups = b""
    for frame in frames:
        body = _compress_envelope(frame.astype("<u2").tobytes())
        frame_groups += _group("FORM", "SPEHLIOS", _chunk("BODY", body))
    return _group("LIST", "SPEHLIOS", prop_stnrd + prop_wavtb + frame_groups)


@pytest.fixture()
def synthetic_eiff(tmp_path):
    rng = np.random.default_rng(42)
    width, height, n_frames = 8, 4, 3
    frames = [rng.integers(0, 4096, size=(height, width), dtype=np.uint16) for _ in range(n_frames)]
    wavelengths = np.linspace(1000, 1700, width)
    path = tmp_path / "test.eiff"
    path.write_bytes(_build_eiff_file(width, height, frames, wavelengths))
    return path, frames, wavelengths


def test_load_eiff_supports_header_only_and_full_cube(synthetic_eiff):
    from enginir.io.eiff_io import load_eiff

    path, frames, wavelengths = synthetic_eiff
    header_only = load_eiff(path, header_only=True)
    full = load_eiff(path)

    assert header_only.X.shape == (0, 4, 8)
    assert header_only.meta["count"] == 3
    assert header_only.meta["source_shape"] == (3, 4, 8)
    np.testing.assert_allclose(header_only.wavelengths, wavelengths, atol=0.03)
    np.testing.assert_array_equal(full.X[0], frames[0])


def test_load_eiff_parallel_frame_workers_match_sequential(synthetic_eiff, monkeypatch):
    import enginir.io.eiff_io as eiff_io

    path, frames, _ = synthetic_eiff
    monkeypatch.setattr(eiff_io, "_PARALLEL_FRAME_THRESHOLD", 1)

    sequential = eiff_io.load_eiff(path, frame_workers=1)
    parallel = eiff_io.load_eiff(path, frame_workers=2)

    np.testing.assert_array_equal(parallel.X, sequential.X)
    np.testing.assert_array_equal(parallel.X[-1], frames[-1])


def test_load_eiff_reports_frame_progress(synthetic_eiff, monkeypatch):
    import enginir.io.eiff_io as eiff_io

    path, _, _ = synthetic_eiff
    monkeypatch.setattr(eiff_io, "_PARALLEL_FRAME_THRESHOLD", 1)
    sequential_progress: list[tuple[int, int]] = []
    parallel_progress: list[tuple[int, int]] = []

    eiff_io.load_eiff(
        path,
        frame_workers=1,
        progress_callback=lambda done, total: sequential_progress.append((done, total)),
    )
    eiff_io.load_eiff(
        path,
        frame_workers=2,
        progress_callback=lambda done, total: parallel_progress.append((done, total)),
    )

    assert sequential_progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert parallel_progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_load_eiff_throttles_large_frame_progress(tmp_path, monkeypatch):
    import enginir.io.eiff_io as eiff_io

    width, height, n_frames = 4, 2, 20
    frames = [
        np.full((height, width), frame_idx, dtype=np.uint16) for frame_idx in range(n_frames)
    ]
    path = tmp_path / "large_progress.eiff"
    path.write_bytes(_build_eiff_file(width, height, frames, np.linspace(1000, 1300, width)))
    monkeypatch.setattr(eiff_io, "_PROGRESS_UPDATE_STEPS", 5)
    progress: list[tuple[int, int]] = []

    eiff_io.load_eiff(
        path,
        frame_workers=1,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(0, 20), (4, 20), (8, 20), (12, 20), (16, 20), (20, 20)]


@pytest.mark.parametrize(
    ("filename", "payload", "error_text"),
    [
        ("tiny.eiff", b"\x00" * 10, "at least"),
        (
            "not_eiff.eiff",
            _pad10("RIFF") + struct.pack(">Q", 200) + b"\x00" * 200,
            "not an EVK HELIOS EIFF file",
        ),
    ],
)
def test_load_eiff_rejects_invalid_files(tmp_path, filename, payload, error_text):
    from enginir.io.eiff_io import EIFFError, load_eiff

    path = tmp_path / filename
    path.write_bytes(payload)
    with pytest.raises(EIFFError, match=error_text):
        load_eiff(path)


def test_load_eiff_enforces_memory_limit(synthetic_eiff):
    from enginir.io.eiff_io import EIFFMemoryError, load_eiff

    path, _, _ = synthetic_eiff
    with pytest.raises(EIFFMemoryError, match="max_bytes"):
        load_eiff(path, max_bytes=10)


def _build_spc_header(
    fnpts: int,
    ffirst: float,
    flast: float,
    fnsub: int = 1,
    ftflgs: int = 0,
    fexp: int = 0x80,
    fxtype: int = 3,
    fytype: int = 2,
) -> bytes:
    buf = bytearray(512)
    buf[0] = ftflgs
    buf[1] = 0x4B
    buf[3] = fexp
    struct.pack_into("<I", buf, 4, fnpts)
    struct.pack_into("<d", buf, 8, ffirst)
    struct.pack_into("<d", buf, 16, flast)
    struct.pack_into("<I", buf, 24, fnsub)
    buf[28] = fxtype
    buf[29] = fytype
    return bytes(buf)


def _build_spc_sub_header(index: int = 0, subnpts: int = 0) -> bytes:
    buf = bytearray(32)
    buf[0] = 0x80
    struct.pack_into("<H", buf, 2, index)
    struct.pack_into("<I", buf, 16, subnpts)
    return bytes(buf)


def _build_spc_file(wavelengths: np.ndarray, spectra: np.ndarray) -> bytes:
    n_spectra, n_points = spectra.shape
    header = _build_spc_header(
        fnpts=n_points,
        ffirst=float(wavelengths[0]),
        flast=float(wavelengths[-1]),
        fnsub=n_spectra,
        ftflgs=0x04 if n_spectra > 1 else 0x00,
    )
    body = b""
    for index in range(n_spectra):
        body += _build_spc_sub_header(index) + spectra[index].astype("<f4").tobytes()
    return header + body


@pytest.fixture()
def synthetic_spc(tmp_path):
    wavelengths = np.linspace(1000, 2000, 128)
    spectra = np.vstack([np.sin(wavelengths / 100), np.cos(wavelengths / 90)]).astype(np.float32)
    path = tmp_path / "test.spc"
    path.write_bytes(_build_spc_file(wavelengths, spectra))
    return path, wavelengths, spectra


def test_load_spc_reads_multi_spectrum_payload(synthetic_spc):
    from enginir.io.spc_io import load_spc

    path, wavelengths, spectra = synthetic_spc
    data = load_spc(path)

    np.testing.assert_allclose(data.X, spectra, atol=1e-6)
    np.testing.assert_allclose(data.wavelengths, wavelengths, atol=0.01)
    assert data.meta["n_spectra"] == 2


def test_load_spc_rejects_subspectrum_length_mismatch(tmp_path):
    from enginir.io.spc_io import SPCError, load_spc

    header = _build_spc_header(100, 1000.0, 2000.0, fnsub=2, ftflgs=0x04)
    y = np.ones(100, dtype="<f4")
    body = _build_spc_sub_header(0, subnpts=100) + y.tobytes()
    body += _build_spc_sub_header(1, subnpts=50) + y.tobytes()
    path = tmp_path / "bad.spc"
    path.write_bytes(header + body)

    with pytest.raises(SPCError, match="50 points"):
        load_spc(path)


def _build_spa_file(
    npts: int,
    x_first: float,
    x_last: float,
    y_values: np.ndarray,
    title: str = "Test Spectrum",
    data_type_code: int = 13,
) -> bytes:
    data_offset = 580
    buf = bytearray(data_offset)
    buf[0:30] = title.encode("latin-1")[:30].ljust(30, b"\x00")
    buf[32:60] = b"Thu Jan 08 10:15:00 2026".ljust(28, b"\x00")
    struct.pack_into("<H", buf, 60, data_type_code)
    struct.pack_into("<f", buf, 564, x_first)
    struct.pack_into("<f", buf, 568, x_last)
    struct.pack_into("<I", buf, 572, npts)
    struct.pack_into("<I", buf, 576, data_offset)
    return bytes(buf) + y_values.astype("<f4").tobytes()


@pytest.fixture()
def synthetic_spa(tmp_path):
    npts = 200
    x_first, x_last = 4000.0, 400.0
    rng = np.random.default_rng(5)
    y = rng.random(npts).astype(np.float32)
    path = tmp_path / "test.spa"
    path.write_bytes(_build_spa_file(npts, x_first, x_last, y, title="NIR Test"))
    return path, x_first, x_last, y


def test_load_spa_reads_values_wavelengths_and_metadata(synthetic_spa):
    from enginir.io.spa_io import load_spa

    path, x_first, x_last, y = synthetic_spa
    data = load_spa(str(path))

    assert data.X.shape == (1, y.shape[0])
    assert data.wavelengths[0] == pytest.approx(x_first, rel=1e-4)
    assert data.wavelengths[-1] == pytest.approx(x_last, rel=1e-4)
    np.testing.assert_allclose(data.X[0], y, atol=1e-6)
    assert data.meta["title"] == "NIR Test"
    assert data.meta["date"] == "Thu Jan 08 10:15:00 2026"  # not run into the type code
    assert data.meta["data_type"] == "absorbance"


def test_load_spa_rejects_truncated_file(tmp_path):
    from enginir.io.spa_io import SPAError, load_spa

    path = tmp_path / "tiny.spa"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(SPAError, match="at least"):
        load_spa(str(path))


def _build_jcamp_xpluspy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    title: str = "Test",
    yfactor: float = 1.0,
) -> str:
    lines = [
        f"##TITLE= {title}",
        "##JCAMP-DX= 4.24",
        "##DATA TYPE= INFRARED SPECTRUM",
        "##XUNITS= 1/CM",
        "##YUNITS= ABSORBANCE",
        f"##NPOINTS= {len(x_values)}",
        f"##FIRSTX= {x_values[0]:.4f}",
        f"##LASTX= {x_values[-1]:.4f}",
        f"##FIRSTY= {y_values[0]:.6f}",
        f"##YFACTOR= {yfactor}",
        "##XYDATA= (X++(Y..Y))",
    ]
    for start in range(0, len(x_values), 10):
        chunk_y = y_values[start : start + 10]
        row = f"{x_values[start]:.4f}  " + "  ".join(f"{value / yfactor:.6f}" for value in chunk_y)
        lines.append(row)
    lines.append("##END=")
    return "\n".join(lines) + "\n"


def _build_jcamp_xyxy(x_values: np.ndarray, y_values: np.ndarray, *, title: str = "XY Test") -> str:
    lines = [
        f"##TITLE= {title}",
        "##JCAMP-DX= 5.01",
        "##DATA TYPE= INFRARED SPECTRUM",
        "##XUNITS= NANOMETERS",
        "##YUNITS= ABSORBANCE",
        f"##NPOINTS= {len(x_values)}",
        f"##FIRSTX= {x_values[0]:.4f}",
        f"##LASTX= {x_values[-1]:.4f}",
        "##XYPOINTS= (XY..XY)",
    ]
    for xi, yi in zip(x_values, y_values):
        lines.append(f"{xi:.4f}  {yi:.6f}")
    lines.append("##END=")
    return "\n".join(lines) + "\n"


def test_load_jcamp_reads_xy_variants_multiblock_and_yfactor(tmp_path):
    from enginir.io.jcamp_io import load_jcamp

    x = np.linspace(4000, 400, 40)
    y1 = np.linspace(0.1, 0.9, 40)
    y2 = np.linspace(0.9, 0.1, 40)
    compound = (
        _build_jcamp_xpluspy(x, y1, title="Block 1")
        + "\n"
        + _build_jcamp_xpluspy(x, y2, title="Block 2")
    )
    compound_path = tmp_path / "compound.dx"
    compound_path.write_text(compound, encoding="latin-1")
    compound_data = load_jcamp(str(compound_path))

    assert compound_data.X.shape == (2, 40)
    np.testing.assert_allclose(compound_data.X[0], y1, atol=1e-5)
    np.testing.assert_allclose(compound_data.X[1], y2, atol=1e-5)

    xyxy_path = tmp_path / "xyxy.jdx"
    xyxy_path.write_text(
        _build_jcamp_xyxy(np.linspace(1000, 2500, 20), np.linspace(0.2, 0.8, 20)),
        encoding="latin-1",
    )
    xyxy_data = load_jcamp(str(xyxy_path))
    assert xyxy_data.X.shape == (1, 20)

    yfactor_path = tmp_path / "yfactor.dx"
    yfactor_path.write_text(_build_jcamp_xpluspy(x, y1, yfactor=0.001), encoding="latin-1")
    yfactor_data = load_jcamp(str(yfactor_path))
    np.testing.assert_allclose(yfactor_data.X[0], y1, atol=1e-5)


def _write_envi_pair(
    tmp_path, X: np.ndarray, wavelengths: np.ndarray | None = None, *, name: str = "cube"
):
    from enginir.io.envi_io import save_envi

    base = str(tmp_path / name)
    save_envi(base, X, wavelengths if wavelengths is not None else np.array([]))
    return base + ".hdr", base + ".img"


def test_envi_round_trip_flatten_and_binary_path_variants(tmp_path):
    from enginir.io import load_spectra
    from enginir.io.envi_io import load_envi

    rng = np.random.default_rng(30)
    cube = rng.random((4, 5, 6)).astype(np.float32)
    wavelengths = np.linspace(1000, 1500, 6)
    hdr, img = _write_envi_pair(tmp_path, cube, wavelengths)

    data = load_envi(hdr)
    flat = load_envi(hdr, flatten=True)
    np.testing.assert_allclose(data.X, cube, rtol=1e-5)
    np.testing.assert_allclose(data.wavelengths, wavelengths, rtol=1e-5)
    assert flat.X.shape == (4 * 5, 6)

    dat = tmp_path / "cube.dat"
    (tmp_path / "cube.img").rename(dat)
    from_dat = load_envi(str(dat))
    from_dispatch = load_spectra(str(dat))
    np.testing.assert_allclose(from_dat.X, cube, rtol=1e-5)
    np.testing.assert_allclose(from_dispatch.X, cube, rtol=1e-5)
    assert img.endswith(".img")


def test_envi_rejects_missing_companion_and_wavelength_mismatch(tmp_path):
    from enginir.io.envi_io import load_envi, save_envi

    with pytest.raises(ValueError, match="X must be 2-D or 3-D"):
        save_envi(str(tmp_path / "bad_shape"), np.ones(10), np.linspace(1000, 2000, 10))

    with pytest.raises(ValueError, match="wavelengths"):
        save_envi(str(tmp_path / "bad_cube"), np.ones((4, 6, 10)), np.linspace(1000, 2000, 7))

    hdr = tmp_path / "orphan.hdr"
    hdr.write_text(
        "ENVI\nsamples = 4\nlines = 3\nbands = 2\ndata type = 4\ninterleave = bsq\nbyte order = 0\n"
    )
    with pytest.raises(FileNotFoundError):
        load_envi(str(hdr))


# ---------------------------------------------------------------------------
# Shared reader API
# ---------------------------------------------------------------------------


def test_all_format_errors_share_one_value_error_base():
    from enginir.io import (
        EIFFError,
        EIFFMemoryError,
        EIFFTruncatedError,
        ENVIError,
        EnginirIOError,
        JCAMPError,
        SPAError,
        SPCError,
    )

    for error in (EIFFError, EIFFMemoryError, EIFFTruncatedError, ENVIError, JCAMPError,
                  SPAError, SPCError):
        assert issubclass(error, EnginirIOError) and issubclass(error, ValueError)


def test_every_loader_reports_source_format_and_x_units(tmp_path, synthetic_eiff, synthetic_spc,
                                                        synthetic_spa):
    from enginir.io import load_spectra

    csv_path = tmp_path / "spectra.csv"
    csv_path.write_text("sample,1000,1001\na,1,2\n")
    jcamp_path = tmp_path / "spectrum.jdx"
    jcamp_path.write_text(_build_jcamp_xyxy(np.linspace(1000, 1100, 5), np.ones(5)))
    hdr, _ = _write_envi_pair(tmp_path, np.ones((2, 2, 3), np.float32), np.array([1.0, 2, 3]))
    paths = {
        "csv": (csv_path, "nm"),
        "jcamp": (jcamp_path, "nm"),
        "envi": (hdr, "nm"),
        "eiff": (synthetic_eiff[0], "nm"),
        "spc": (synthetic_spc[0], "nm"),
        "spa": (synthetic_spa[0], "cm-1"),
    }
    for file_format, (path, x_units) in paths.items():
        meta = load_spectra(path).meta
        assert meta["format"] == file_format
        assert meta["x_units"] == x_units
        assert meta["source_path"] == str(Path(path).resolve())


def test_load_spectra_dispatches_every_documented_extension():
    """The extension table in the module docstring is the one load_spectra uses."""
    import enginir.io as io_mod

    table = io_mod.__doc__.split("Project files")[0]
    documented = set(re.findall(r"``(\.[a-z0-9]+)``", table))
    assert documented == set(io_mod._READERS)
    for suffix, (file_format, _) in io_mod._READERS.items():
        assert io_mod.file_format(f"x{suffix.upper()}") == file_format


def test_load_csv_rejects_non_numeric_cells_and_save_accepts_paths(tmp_path):
    from enginir.io import load_csv, save_csv

    path = tmp_path / "spectra.csv"
    save_csv(path, np.array([[1.0, 2.0]]), np.array([1000.0, 1001.0]), ["a"])
    assert load_csv(path).meta["sample_names"] == ["a"]

    path.write_text("sample,1000,1001\na,1,oops\n")
    with pytest.raises(ValueError, match="non-numeric spectral values"):
        load_csv(path)


# ---------------------------------------------------------------------------
# JCAMP-DX
# ---------------------------------------------------------------------------


def test_jcamp_xfactor_scales_table_values_but_not_firstx_lastx(tmp_path):
    from enginir.io import load_jcamp

    x = np.linspace(4000.0, 3991.0, 10)
    text = _build_jcamp_xpluspy(x, np.ones(10)).replace(
        "##XYDATA=", "##XFACTOR= 0.5\n##XYDATA="
    )
    lines = text.splitlines()
    table_start = lines.index("##XYDATA= (X++(Y..Y))") + 1
    first, *rest = lines[table_start].split()
    lines[table_start] = " ".join([f"{float(first) / 0.5:.4f}", *rest])  # X check in table units
    path = tmp_path / "xfactor.dx"
    path.write_text("\n".join(lines) + "\n")

    data = load_jcamp(path)

    np.testing.assert_allclose(data.wavelengths, x)
    assert data.meta["x_units"] == "cm-1"


def test_jcamp_ignores_comment_lines_and_inline_comments(tmp_path):
    from enginir.io import load_jcamp

    x = np.linspace(1000.0, 1009.0, 10)
    text = _build_jcamp_xpluspy(x, np.arange(10.0)).replace(
        "##XYDATA= (X++(Y..Y))\n", "##XYDATA= (X++(Y..Y))  $$ table follows\n$$ a comment line\n"
    )
    text = text.replace("##END=", "$$ trailing comment\n##END=")
    path = tmp_path / "comments.dx"
    path.write_text(text)

    np.testing.assert_allclose(load_jcamp(path).X[0], np.arange(10.0))


def test_jcamp_rejects_blocks_with_different_axes_or_lengths(tmp_path):
    from enginir.io import JCAMPError, load_jcamp

    y = np.ones(10)
    shifted = _build_jcamp_xpluspy(np.linspace(1000, 1009, 10), y) + _build_jcamp_xpluspy(
        np.linspace(2000, 2009, 10), y
    )
    shorter = _build_jcamp_xpluspy(np.linspace(1000, 1009, 10), y) + _build_jcamp_xpluspy(
        np.linspace(1000, 1008, 9), y[:9]
    )
    for name, text in (("shifted.dx", shifted), ("shorter.dx", shorter)):
        path = tmp_path / name
        path.write_text(text)
        with pytest.raises(JCAMPError, match="different X axis"):
            load_jcamp(path)


def test_jcamp_reports_compressed_tables_as_unsupported(tmp_path):
    from enginir.io import JCAMPError, load_jcamp

    text = _build_jcamp_xpluspy(np.linspace(1000, 1009, 10), np.ones(10))
    text = text.replace("##END=", "1010 A0J1\n##END=")
    path = tmp_path / "compressed.dx"
    path.write_text(text)

    with pytest.raises(JCAMPError, match="compressed"):
        load_jcamp(path)


# ---------------------------------------------------------------------------
# SPC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "exponent", "dtype", "scale"),
    [(0x00, 32, "<i4", 1.0), (0x00, 30, "<i4", 0.25), (0x01, 16, "<i2", 1.0),
     (0x01, 18, "<i2", 4.0)],
)
def test_spc_decodes_scaled_integer_y_values(tmp_path, flags, exponent, dtype, scale):
    from enginir.io import load_spc

    values = np.array([1, -2, 300, 4000], dtype=dtype)
    header = _build_spc_header(4, 1000.0, 1003.0, ftflgs=flags, fexp=exponent)
    path = tmp_path / "integer.spc"
    path.write_bytes(header + _build_spc_sub_header() + values.tobytes())

    np.testing.assert_allclose(load_spc(path).X[0], values * scale)


def test_spc_reports_raman_shift_axis_and_rejects_per_spectrum_x_arrays(tmp_path):
    from enginir.io import SPCError, load_spc

    y = np.ones(4, dtype="<f4").tobytes()
    raman = tmp_path / "raman.spc"
    raman.write_bytes(_build_spc_header(4, 100.0, 400.0, fxtype=13) + _build_spc_sub_header() + y)
    meta = load_spc(raman).meta
    assert (meta["x_label"], meta["x_units"]) == ("raman shift", "cm-1")
    assert meta["timestamp"] is None

    xyxy = tmp_path / "xyxy.spc"
    xyxy.write_bytes(_build_spc_header(4, 1.0, 4.0, ftflgs=0x40 | 0x04) + _build_spc_sub_header()
                     + y)
    with pytest.raises(SPCError, match="TXYXYS"):
        load_spc(xyxy)


# ---------------------------------------------------------------------------
# ENVI
# ---------------------------------------------------------------------------


def _write_envi_raw(tmp_path, data: bytes, **fields) -> Path:
    header = {"samples": 3, "lines": 2, "bands": 4, "data type": 12, "interleave": "bsq",
              "byte order": 0, "header offset": 0, **fields}
    hdr = tmp_path / "raw.hdr"
    hdr.write_text("ENVI\n" + "\n".join(f"{key} = {value}" for key, value in header.items()))
    (tmp_path / "raw.img").write_bytes(data)
    return hdr


@pytest.mark.parametrize("interleave", ["bsq", "bil", "bip"])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_envi_reads_every_interleave_byte_order_and_header_offset(tmp_path, interleave,
                                                                  byte_order):
    from enginir.io import load_envi

    cube = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)  # (lines, samples, bands)
    layout = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    stored = cube.transpose(layout).astype(">u2" if byte_order else "<u2")
    hdr = _write_envi_raw(tmp_path, b"\x00" * 16 + stored.tobytes(), interleave=interleave,
                          **{"byte order": byte_order, "header offset": 16})

    X = load_envi(hdr).X
    np.testing.assert_array_equal(X, cube)
    assert X.dtype.isnative and X.flags.c_contiguous  # big-endian files are converted


def test_envi_converts_micrometers_and_validates_wavelengths(tmp_path):
    from enginir.io import ENVIError, load_envi

    data = np.zeros(24, "<u2").tobytes()
    hdr = _write_envi_raw(tmp_path, data, wavelength="{1.0, 1.1, 1.2, 1.3}",
                          **{"wavelength units": "Micrometers"})
    np.testing.assert_allclose(load_envi(hdr).wavelengths, [1000.0, 1100.0, 1200.0, 1300.0])

    hdr = _write_envi_raw(tmp_path, data, wavelength="{1000, 1100}")
    with pytest.raises(ENVIError, match="2 wavelengths for 4 bands"):
        load_envi(hdr)

    hdr = _write_envi_raw(tmp_path, data, wavelength="{1, 2, 3, 4}",
                          **{"wavelength units": "Furlongs"})
    with pytest.raises(ENVIError, match="units"):
        load_envi(hdr)


def test_envi_checks_header_values_and_data_size(tmp_path):
    from enginir.io import ENVIError, load_envi

    with pytest.raises(ENVIError, match="not an integer"):
        load_envi(_write_envi_raw(tmp_path, np.zeros(24, "<u2").tobytes(), samples="abc"))
    with pytest.raises(ENVIError, match="holds 23 values"):
        load_envi(_write_envi_raw(tmp_path, np.zeros(23, "<u2").tobytes()))

    records: list = []
    sink_id = logger.add(lambda msg: records.append(msg.record), level="WARNING")
    try:
        cube = load_envi(_write_envi_raw(tmp_path, np.zeros(30, "<u2").tobytes())).X
    finally:
        logger.remove(sink_id)
    assert cube.shape == (2, 3, 4)
    assert any("ignoring 6 values" in record["message"] for record in records)


def test_save_envi_keeps_dotted_base_names_and_writes_lists_in_braces(tmp_path):
    from enginir.io import load_envi, save_envi

    save_envi(tmp_path / "scan_v1.5", np.ones((2, 2, 3), np.float32), np.array([1.0, 2.0, 3.0]),
              meta={"fwhm": [5.0, 5.0, 5.0]})

    assert (tmp_path / "scan_v1.5.hdr").is_file() and (tmp_path / "scan_v1.5.img").is_file()
    assert load_envi(tmp_path / "scan_v1.5.hdr").meta["fwhm"] == "5.0, 5.0, 5.0"


# ---------------------------------------------------------------------------
# EIFF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frames", [slice(0, 2), slice(None, None, 2), slice(None, None, -1),
                                    slice(2, 0, -1)])
@pytest.mark.parametrize("workers", [1, 4])
def test_eiff_frame_slices_match_numpy_slicing(tmp_path, monkeypatch, frames, workers):
    import enginir.io.eiff_io as eiff_io

    monkeypatch.setattr(eiff_io, "_PARALLEL_FRAME_THRESHOLD", 1)
    rng = np.random.default_rng(1)
    cube = rng.integers(0, 4096, size=(5, 4, 8), dtype=np.uint16)
    path = tmp_path / "frames.eiff"
    path.write_bytes(_build_eiff_file(8, 4, list(cube), np.linspace(1000, 1700, 8)))

    data = eiff_io.load_eiff(path, frames=frames, frame_workers=workers)

    np.testing.assert_array_equal(data.X, cube[frames])
    assert data.meta["source_shape"] == cube[frames].shape


def test_eiff_truncated_file_returns_complete_frames_with_flag(tmp_path):
    from enginir.io import load_eiff

    rng = np.random.default_rng(2)
    cube = rng.integers(0, 4096, size=(3, 4, 8), dtype=np.uint16)
    path = tmp_path / "truncated.eiff"
    path.write_bytes(_build_eiff_file(8, 4, list(cube), np.linspace(1000, 1700, 8))[:-10])

    records: list = []
    sink_id = logger.add(lambda msg: records.append(msg.record), level="WARNING")
    try:
        data = load_eiff(path)
    finally:
        logger.remove(sink_id)

    np.testing.assert_array_equal(data.X, cube[:2])
    assert data.meta["truncated"] is True and data.meta["frames_loaded"] == 2
    assert any("truncated" in record["message"] for record in records)


def test_eiff_keeps_the_full_wavelength_table_and_warns_on_band_mismatch(tmp_path):
    from enginir.io import load_eiff

    frames = [np.zeros((4, 8), np.uint16)]
    path = tmp_path / "wavelengths.eiff"
    path.write_bytes(_build_eiff_file(8, 4, frames, np.linspace(1000, 1700, 6)))

    records: list = []
    sink_id = logger.add(lambda msg: records.append(msg.record), level="WARNING")
    try:
        data = load_eiff(path, header_only=True)
    finally:
        logger.remove(sink_id)
    assert data.wavelengths.size == 6
    assert any("6 entries for 8 bands" in record["message"] for record in records)


def test_load_csv_reads_xlsx_workbooks_and_selects_sheets(tmp_path):
    import pandas as pd

    from enginir.io import load_spectra

    path = tmp_path / "spectra.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"sample": ["a"], 1000.0: [1.0], 1001.0: [2.0]}).to_excel(
            writer, sheet_name="first", index=False
        )
        pd.DataFrame({"sample": ["b", "c"], 1000.0: [3.0, 5.0], 1001.0: [4.0, 6.0]}).to_excel(
            writer, sheet_name="Cal set", index=False
        )

    first = load_spectra(path)
    calibration = load_spectra(path, sheet_name="Cal set")

    np.testing.assert_array_equal(first.X, [[1.0, 2.0]])
    np.testing.assert_array_equal(calibration.X, [[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(calibration.wavelengths, [1000.0, 1001.0])
    assert calibration.meta["sample_names"] == ["b", "c"]


def test_eiff_frame_chunks_stream_the_same_frames_as_load_eiff(tmp_path):
    from enginir.io import iter_eiff_frame_chunks, load_eiff

    cube = np.arange(5 * 4 * 8, dtype=np.uint16).reshape(5, 4, 8)
    path = tmp_path / "stream.eiff"
    path.write_bytes(_build_eiff_file(8, 4, list(cube), np.linspace(1000, 1700, 8)))

    chunks = list(iter_eiff_frame_chunks(path, chunk_size=2, frames=slice(1, 5)))
    assert [offset for offset, _ in chunks] == [0, 2]
    np.testing.assert_array_equal(np.concatenate([block for _, block in chunks]),
                                  load_eiff(path, frames=slice(1, 5)).X)


def test_eiff_frame_chunks_raise_on_truncation(tmp_path):
    from enginir.io import EIFFTruncatedError, iter_eiff_frame_chunks

    cube = np.ones((3, 4, 8), dtype=np.uint16)
    path = tmp_path / "truncated.eiff"
    path.write_bytes(_build_eiff_file(8, 4, list(cube), np.linspace(1000, 1700, 8))[:-10])

    with pytest.raises(EIFFTruncatedError, match="frame 2 is missing"):
        list(iter_eiff_frame_chunks(path, chunk_size=2))


def test_eiff_frame_without_body_is_an_error_not_a_skipped_frame(tmp_path):
    from enginir.io import EIFFError, load_eiff

    width, height = 8, 4
    frame = np.ones((height, width), dtype=np.uint16)
    good = _group("FORM", "SPEHLIOS", _chunk("BODY", _compress_envelope(frame.tobytes())))
    bad = _group("FORM", "SPEHLIOS", _chunk("NOBODY", b"0" * 8))
    props = _group("PROP", "EVKSTNRD", _chunk("EVKSTNRD", _build_stnrd(width, height, 3)))
    path = tmp_path / "missing_body.eiff"
    path.write_bytes(_group("LIST", "SPEHLIOS", props + good + bad + good))

    with pytest.raises(EIFFError, match="frame 1 has no valid BODY"):
        load_eiff(path)


def test_eiff_root_chunk_may_declare_more_than_four_gib(tmp_path):
    from enginir.io import load_eiff

    cube = np.ones((2, 4, 8), dtype=np.uint16)
    data = bytearray(_build_eiff_file(8, 4, list(cube), np.linspace(1000, 1700, 8)))
    data[10:18] = struct.pack(">Q", 5 * 1024**3)  # a long scan; the file itself is short
    path = tmp_path / "large_root.eiff"
    path.write_bytes(bytes(data))

    assert load_eiff(path).X.shape == (2, 4, 8)


def test_envi_without_wavelength_units_keeps_values_as_unknown(tmp_path):
    from enginir.io import load_envi

    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    cube.transpose(2, 0, 1).tofile(tmp_path / "cube.raw")
    (tmp_path / "cube.raw.hdr").write_text(
        "ENVI\nsamples = 3\nlines = 2\nbands = 4\ndata type = 4\ninterleave = bsq\n"
        "byte order = 0\nwavelength units = Unknown\nwavelength = {1.0, 1.1, 1.2, 1.3}\n"
    )

    data = load_envi(tmp_path / "cube.raw")
    np.testing.assert_array_equal(data.X, cube)
    np.testing.assert_allclose(data.wavelengths, [1.0, 1.1, 1.2, 1.3])
    assert data.meta["x_units"] == "unknown"


def test_hdf5_reads_negative_frame_steps_and_keeps_its_own_identity(tmp_path):
    from enginir.io import load_hdf5, save_hdf5

    cube = np.arange(5 * 2 * 3, dtype=np.uint16).reshape(5, 2, 3)
    path = tmp_path / "cube.h5"
    save_hdf5(path, cube, np.array([1000.0, 1100.0, 1200.0]),
              meta={"source_path": "/elsewhere/cube.eiff", "format": "eiff", "timestamp": None,
                    "device": {"serial": "A1"}, "labels": ["a", None]})

    data = load_hdf5(path, frames=slice(None, None, -2))
    np.testing.assert_array_equal(data.X, cube[::-2])
    assert data.meta["format"] == "hdf5"
    assert data.meta["source_path"] == str(path.resolve())
    assert data.meta["device"] == {"serial": "A1"}
    assert data.meta["labels"] == ["a", None]
    assert "timestamp" not in data.meta


def test_save_hdf5_leaves_an_existing_file_intact_on_failure(tmp_path):
    from enginir.io import load_hdf5, save_hdf5

    path = tmp_path / "cube.h5"
    save_hdf5(path, np.ones((2, 3)), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError):
        save_hdf5(path, np.zeros((2, 3)), np.array([1.0, 2.0, 3.0]), meta={"mixed": [1.0, None]})

    np.testing.assert_array_equal(load_hdf5(path).X, 1.0)
    assert [item.name for item in tmp_path.iterdir()] == ["cube.h5"]


def test_roi_storage_rejects_integers_outside_uint16():
    from enginir.io.project_store import as_roi_storage_array

    assert as_roi_storage_array(np.array([[0, 65535]], dtype=np.int32)).dtype == np.uint16
    with pytest.raises(ValueError, match="fit into uint16"):
        as_roi_storage_array(np.array([[-1, 70000]], dtype=np.int32))


def test_eiff_device_info_reads_nul_terminated_utf16_fields_at_fixed_offsets():
    from enginir.io.eiff_io import _parse_device_info

    payload = bytearray(600)
    payload[54:54 + 10] = "SN-42".encode("utf-16le")  # followed by NULs
    payload[120:120 + 24] = "AlphaVision\x00junk".encode("utf-16le")[:24]  # text after a NUL
    payload[460:460 + 6] = "EVK".encode("utf-16le")
    assert _parse_device_info(bytes(payload)) == {
        "serial": "SN-42", "camera_model": "AlphaVision", "manufacturer": "EVK"
    }
    assert _parse_device_info(bytes(100)) == {}  # short payload: absent fields are omitted
