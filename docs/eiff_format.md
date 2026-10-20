# EIFF Binary Format Reference

EIFF (EVK Interchangeable File Format) is a chunk-based container format for
storing spectral image sequences from HELIOS hyperspectral NIR cameras. It
derives from the IFF (Interchange File Format) concept but extends identifiers
to 10 bytes and size fields to 64 bits.

---

## 1. Fundamental Concepts

### Chunks

Every piece of data in an EIFF file is wrapped in a **chunk**. A chunk has:

- a 10-byte ASCII name (identifier), space-padded
- an 8-byte **big-endian** unsigned 64-bit size field (payload size in bytes)
- a payload of exactly `size` bytes

There are two kinds of chunks:

| Kind | Description |
|------|-------------|
| **Data chunk** | Leaf node. Carries actual data values. |
| **Group chunk** | Container node. Payload starts with a 10-byte content-type name, followed by child chunks. |

### No Padding

Unlike IFF, EIFF stores data as a **continuous byte stream with no alignment
padding**. The byte immediately after a chunk's payload is the start of the
next chunk.

---

## 2. Identifiers (Chunk Names)

All identifiers are **exactly 10 bytes of ASCII**, right-padded with spaces
(`0x20`), no null terminator:

```
[Name: 8 chars][Version: 2 chars]
```

- Characters: uppercase A-Z and digits 0-9 only; must begin with a letter.
- Version field: two spaces (`"  "`) means version 00. Subsequent versions are
  `"01"` through `"99"` (decimal ASCII).

### Reserved Identifiers

```
FORM          single image container
LIST          multi-image list container
CAT           simple list of uniform objects
PROP          shared-property container (only valid inside LIST)
BODY          pixel data container
PROPINFO      metadata sub-chunk inside PROP groups
```

---

## 3. Byte Order

**Chunk headers** (name + size) use **big-endian** byte order for the 64-bit
size field.

**Payload data** varies:
- EVKSTNRD header fields are **big-endian** u16 values
- Compressed data envelopes use **little-endian** fields internally
- Pixel data is **little-endian** uint16 (for standard HELIOS configuration)
- EVKWAVTB wavelength values are **little-endian** uint32
- EVKDVINF strings are UTF-16LE

---

## 4. Binary Layout

### 4.1 Data Chunk (leaf)

```
Offset  Size   Field       Description
     0    10   name        ASCII identifier, space-padded
    10     8   size        uint64 BE -- byte count of the payload that follows
    18  size   payload     raw data bytes
```

**Total chunk size on disk** = 18 + `size` bytes.

### 4.2 Group Chunk (container)

```
Offset  Size   Field         Description
     0    10   keyword       "FORM      ", "LIST      ", "CAT       " or "PROP      "
    10     8   total_size    uint64 BE -- bytes from content_name to end of last child
    18    10   content_name  ASCII identifier, e.g. "SPEHLIOS  "
    28   ...   children      sequence of child chunks
```

The `total_size` includes the 10-byte `content_name` and all child bytes.

---

## 5. File Structure

### 5.1 Multi-Image Stack (typical HELIOS output)

```
LIST                         <- group keyword (10 B)
total_size                   <- uint64 BE (8 B)
SPEHLIOS                     <- content_name (10 B)

  PROP / EVKSTNRD            <- standard header (no PROPINFO, raw 54-byte struct)
    EVKSTNRD  54  {header}

  PROP / EVKDVINF            <- device information
    PROPINFO  12  {info}
    EVKDVINF  N   {compressed payload}

  PROP / EVKWAVTB            <- wavelength calibration table
    PROPINFO  12  {info}
    EVKWAVTB  N   {compressed payload}

  PROP / EVKROIBI            <- ROI bitmap
    PROPINFO  12  {info}
    EVKROIBI  N   {compressed payload}

  PROP / EVKMODEL            <- model data
    PROPINFO  12  {info}
    EVKMODEL  N   {compressed payload}

  PROP / EVKMACOL            <- material colors
    PROPINFO  12  {info}
    EVKMACOL  N   {compressed payload}

  FORM / SPEHLIOS            <- frame 0
    BODY  size  {compressed pixel data}

  FORM / SPEHLIOS            <- frame 1
    BODY  size  {compressed pixel data}

  ...                        <- frames 2 through N-1
```

### 5.2 PROP Chunk Patterns

There are two patterns:

1. **EVKSTNRD PROP**: Contains only the raw EVKSTNRD data sub-chunk (no
   PROPINFO). This is the first PROP and must be parsed differently.

2. **All other PROPs** (EVKDVINF, EVKWAVTB, etc.): Start with a
   `PROPINFO` sub-chunk (12 bytes, content: `01 00 00 00 02 00 00 00 00 00 00 00`)
   followed by the actual data sub-chunk, which uses the compression envelope
   (see section 8).

---

## 6. EVK Standard Header (`EVKSTNRD`)

**Payload size:** exactly **54 bytes**

All multi-byte fields are **big-endian u16**, except date/time which use
individual bytes.

```
Offset  Size  Type     Field          Description
     0     2  u16 BE   Version        Encoded as major.minor (0x0100 = v1.0)
     2     4  bytes    Reserved       Legacy content ID (typically 0xFFFFFFF8)
     6     2  bytes    Padding        Always 0x0000
     8     2  u16 BE   Width          Image width in pixels
    10     2  bytes    Padding        Always 0x0000
    12     2  u16 BE   Height         Image height in pixels
    14     2  u16 BE   BytesPerValue  Bytes per pixel (e.g. 2 for 16-bit mono)
    16     2  u16 BE   Compression    Compression type (see section 8)
    18     2  u16 BE   BitsPerValue   Significant bits (e.g. 12 for 12-bit in 16-bit)
    20     2  u16 BE   Unknown1       Typically 1
    22     2  u16 BE   Unknown2       Typically 1
    24     4  bytes    Reserved       Always 0x00000000
    28     2  u16 BE   Count          Number of frames in the file
    30     2  u16 BE   Year           Capture year (e.g. 2025)
    32     1  u8       Month          Capture month (1-12)
    33     1  u8       Day            Capture day (1-31)
    34     1  u8       Hour           Capture hour (0-23)
    35     1  u8       Minute         Capture minute (0-59)
    36     1  u8       Second         Capture second (0-59)
    37     1  u8       Centisecond    Hundredths of a second (0-99)
    38    16  bytes    Reserve        All zeros
```

---

## 7. Additional PROP Chunks

### 7.1 EVKDVINF (Device Information)

After decompression, contains a 54-byte binary header followed by
fixed-length UTF-16LE null-terminated string fields:

| Offset | Field | Description |
|--------|-------|-------------|
| 54 | Serial | Camera serial number (e.g. "220332240003") |
| 120 | Camera Model | Camera model name (e.g. "EVK-AlphaVision") |
| 296 | Firmware | Firmware version (e.g. "0.13.1") |
| 362 | SDK Version | SDK version string (e.g. "0.0.0-evkalpha") |
| 460 | Manufacturer | Manufacturer name (e.g. "EVK DI Kerschhaggl GmbH") |
| 526 | Interface | Interface identifier (e.g. "hsi0") |

### 7.2 EVKWAVTB (Wavelength Table)

After decompression, contains wavelength calibration data:

```
Offset  Size  Type     Field     Description
     0     2  u16 LE   Version   Table version (typically 1)
     2     2  u16 LE   Type      Table type (typically 1)
     4     2  u16 LE   Reserved  Always 0
     6     2  u16 LE   Count     Number of wavelength entries
     8   N*4  u32 LE   Values    Wavelength values in units of 0.025 nm
```

To convert to nanometers: `wavelength_nm = value * 0.025`

Typical range for HELIOS SWIR cameras: 991-1677 nm (220 spectral bands).

### 7.3 Other PROP Chunks

| Chunk | Description |
|-------|-------------|
| EVKROIBI | Region-of-interest bitmap |
| EVKMODEL | Embedded model data |
| EVKMACOL | Material color mapping |

These are present in files but not required for basic spectral data loading.

---

## 8. Compression

### 8.1 Compression Envelope

All compressed payloads (BODY chunks and PROP data chunks except EVKSTNRD)
use the same envelope format:

```
Offset  Size  Type     Field            Description
     0     4  u32 LE   CompressionType  0=none, 2=zlib
     4     4  u32 LE   UncompressedSize Expected size after decompression
     8     1  u8       Flag             Always 0x01
     9   ...  bytes    CompressedData   zlib-compressed payload
```

Decompression (Python):
```python
import struct, zlib

comp_type = struct.unpack('<I', data[0:4])[0]
uncomp_size = struct.unpack('<I', data[4:8])[0]
if comp_type == 2:
    payload = zlib.decompress(data[9:])
    assert len(payload) == uncomp_size
elif comp_type == 0:
    payload = data[9:]
```

### 8.2 Compression Types

| Value | Description | Status |
|-------|-------------|--------|
| 0 | Uncompressed | Supported |
| 1 | 7-zip / LZMA | Documented, not observed in practice |
| 2 | zlib (deflate) | **Default in HELIOS files** |

> The original SDK documentation only listed types 0 and 1. Type 2 (zlib) is
> used by current HELIOS firmware and is the only compression observed in
> real-world files.

---

## 9. Pixel Data (BODY Chunk)

### Layout

After decompression, pixels are stored as a flat array in row-major order:

```
row 0, col 0 ... row 0, col (Width-1)
row 1, col 0 ... row 1, col (Width-1)
...
row (Height-1), col 0 ... row (Height-1), col (Width-1)
```

Each pixel occupies `BytesPerValue` bytes. Standard HELIOS configuration:
- `BytesPerValue=2`, `BitsPerValue=12`
- Each pixel = 1 x `uint16 LE`
- Only the 12 least-significant bits are valid (values 0-4095)
- Frame size = `Width * Height * BytesPerValue` bytes

> **Note:** `load_eiff()` only supports `BytesPerValue=2` (uint16). Files with
> other pixel depths (e.g. 8-bit or 32-bit) are rejected with `EIFFError` at
> load time rather than being silently misdecoded.

```python
import numpy as np
frame = np.frombuffer(decompressed_body, dtype='<u2').reshape(height, width)
```

---

## 10. Python Loader Output Convention

The `load_eiff()` function returns a `SpectralData` named tuple:

```python
from enginir.io import load_eiff

data = load_eiff("path/to/file.eiff")
data.X              # np.ndarray, dtype=uint16, shape (n_frames, height, n_bands)
data.wavelengths    # np.ndarray, dtype=float64, shape (n_bands,), nm
data.meta           # dict with keys: source_path, format, x_units, version, width,
                    #   height, count, bytes_per_value, bits_per_value, compression,
                    #   stnrd_u16_20, stnrd_u16_22 (unidentified header fields),
                    #   timestamp (camera local time), frames_total, frames_loaded,
                    #   source_shape, truncated,
                    #   device (sub-dict with serial, camera_model, ...)

# tuple unpacking also works:
cube, wavelengths, meta = load_eiff("path/to/file.eiff")
```

### Interpreting the `cube` array

HELIOS cameras are push-broom imagers: every frame is one spatial line of the
scene, recorded at all wavelengths at once. The EVKSTNRD `height` is the number
of spatial pixels along the line and `width` is the number of spectral bands
(220 for the HELIOS sample file, matching the EVKWAVTB table). The cube
returned by `load_eiff` therefore has shape `(n_frames, height, n_bands)`:

```
cube[frame, pixel, band]  →  raw counts of spatial position (frame, pixel) at wavelengths[band]
```

### Reshaping for preprocessing / models

Preprocessing transformers and models expect spectra as rows,
`(n_samples, n_wavelengths)`. Every spatial pixel is one spectrum:

```python
n_frames, height, n_bands = cube.shape
X = cube.reshape(n_frames * height, n_bands)       # (n_pixels, n_bands)
prediction_map = y_pred.reshape(n_frames, height)  # back to image layout
```

`enginir.utils.flatten_cube` / `unflatten_cube` perform the same round trip.

---

## 11. Identifier Byte Values (Reference)

All 10-byte identifiers, space-padded:

```
"FORM      "  46 4F 52 4D 20 20 20 20 20 20
"LIST      "  4C 49 53 54 20 20 20 20 20 20
"PROP      "  50 52 4F 50 20 20 20 20 20 20
"BODY      "  42 4F 44 59 20 20 20 20 20 20
"PROPINFO  "  50 52 4F 50 49 4E 46 4F 20 20
"EVKSTNRD  "  45 56 4B 53 54 4E 52 44 20 20
"EVKDVINF  "  45 56 4B 44 56 49 4E 46 20 20
"EVKWAVTB  "  45 56 4B 57 41 56 54 42 20 20
"EVKROIBI  "  45 56 4B 52 4F 49 42 49 20 20
"EVKMODEL  "  45 56 4B 4D 4F 44 45 4C 20 20
"EVKMACOL  "  45 56 4B 4D 41 43 4F 4C 20 20
"SPEHLIOS  "  53 50 45 48 4C 49 4F 53 20 20
```
