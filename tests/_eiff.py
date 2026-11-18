"""EIFF files built in memory, chunk by chunk, for reader tests."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def pad10(name: str) -> bytes:
    return name.encode("ascii").ljust(10)


def chunk(name: str, payload: bytes) -> bytes:
    return pad10(name) + struct.pack(">Q", len(payload)) + payload


def group(keyword: str, content_type: str, children: bytes) -> bytes:
    inner = pad10(content_type) + children
    return pad10(keyword) + struct.pack(">Q", len(inner)) + inner


def compress_envelope(data: bytes) -> bytes:
    return struct.pack("<I", 2) + struct.pack("<I", len(data)) + b"\x01" + zlib.compress(data)


def build_stnrd(width: int, height: int, count: int, compression: int = 2) -> bytes:
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


def build_wavtb(wavelengths_nm: np.ndarray) -> bytes:
    header = struct.pack("<4H", 1, 1, 0, len(wavelengths_nm))
    values = (wavelengths_nm / 0.025).astype(np.uint32).tobytes()
    return header + values


def eiff_bytes(
    width: int,
    height: int,
    frames: list[np.ndarray],
    wavelengths_nm: np.ndarray | None = None,
) -> bytes:
    prop_stnrd = group(
        "PROP", "EVKSTNRD", chunk("EVKSTNRD", build_stnrd(width, height, len(frames)))
    )
    prop_wavtb = b""
    if wavelengths_nm is not None:
        propinfo = chunk("PROPINFO", b"\x01\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00")
        wavtb_chunk = chunk("EVKWAVTB", compress_envelope(build_wavtb(wavelengths_nm)))
        prop_wavtb = group("PROP", "EVKWAVTB", propinfo + wavtb_chunk)

    frame_groups = b""
    for frame in frames:
        body = compress_envelope(frame.astype("<u2").tobytes())
        frame_groups += group("FORM", "SPEHLIOS", chunk("BODY", body))
    return group("LIST", "SPEHLIOS", prop_stnrd + prop_wavtb + frame_groups)
