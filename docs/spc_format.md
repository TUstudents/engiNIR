# SPC Binary Format Reference

SPC (Spectral Data Format) is a binary file format originally developed by
Galactic Industries Corporation and later adopted by Thermo Scientific as the
standard interchange format for spectroscopy software (GRAMS/AI, Thermo
OMNIC, and many instrument control packages).

Files carry the `.spc` extension. This reference covers the **new format**
(version byte `0x4B`), which is produced by all modern instruments. The old
format (`0x4D`) is not supported.

---

## 1. File Structure

An SPC file consists of:

```
[512-byte main header]
[optional: fnpts × 4 bytes of float32 X values  (only if TXVALS flag set)]
[fnsub sub-spectra, each:]
  [32-byte sub-header]
  [fnpts × 4 bytes of Y values]
[optional: log block]
```

All multi-byte integers are **little-endian** unless stated otherwise.

---

## 2. Main Header (512 bytes)

| Offset | Size | Type    | Field    | Description |
|--------|------|---------|----------|-------------|
| 0      | 1    | u8      | ftflgs   | Flags (see §2.1) |
| 1      | 1    | u8      | fversn   | Version: `0x4B` = new, `0x4D` = old |
| 2      | 1    | u8      | fexper   | Experiment type code |
| 3      | 1    | i8      | fexp     | Y exponent (signed): `-128` (0x80) = IEEE float32; other = integer scale |
| 4      | 4    | u32 LE  | fnpts    | Number of data points per sub-spectrum |
| 8      | 8    | f64 LE  | ffirst   | First X value |
| 16     | 8    | f64 LE  | flast    | Last X value |
| 24     | 4    | u32 LE  | fnsub    | Number of sub-spectra (1 for single-spectrum files) |
| 28     | 1    | u8      | fxtype   | X axis type (see §2.2) |
| 29     | 1    | u8      | fytype   | Y axis type (see §2.3) |
| 30     | 1    | u8      | fztype   | Z axis type (same codes as fxtype) |
| 31     | 1    | u8      | fpost    | Posting disposition |
| 32     | 4    | u32 LE  | fdate    | Date/time packed (see §2.4) |
| 36     | 9    | char[9] | fres     | Resolution description string (null-padded) |
| 45     | 9    | char[9] | fsource  | Source instrument name (null-padded) |
| 54     | 2    | u16 LE  | fpeakpt  | Peak point number for interferograms |
| 56     | 32   | f32[8]  | fspares  | Reserved float values |
| 88     | 130  | char[]  | fcmnt    | Comment string (null-terminated, 130 bytes) |
| 218    | 30   | char[]  | fcatxt   | Axis category label text (null-terminated) |
| 248    | 4    | u32 LE  | flogoff  | Byte offset to log block; 0 = no log |
| 252    | 4    | u32 LE  | fmods    | File modification flags |
| 256    | 1    | u8      | fprocs   | Processing code |
| 257    | 1    | u8      | flevel   | Calibration level |
| 258    | 2    | u16 LE  | fsampin  | Sub-method sample injection number |
| 260    | 4    | f32 LE  | ffactor  | Floating-point scaling factor |
| 264    | 48   | char[]  | fmethod  | Method/program file name |
| 312    | 4    | f32 LE  | fzinc    | Z sub-file increment |
| 316    | 4    | u32 LE  | fwplanes | Number of W planes |
| 320    | 4    | f32 LE  | fwinc    | W plane increment |
| 324    | 1    | u8      | fwtype   | W axis type code |
| 325   | 187  | bytes   | freserv  | Reserved (zero) |

### 2.1 ftflgs Bit Flags

| Bit | Mask | Name    | Meaning |
|-----|------|---------|---------|
| 0   | 0x01 | TSPREC  | 16-bit Y precision (rare) |
| 1   | 0x02 | TCGRAM  | Enables fexper chromatogram codes |
| 2   | 0x04 | TMULTI  | Multiple sub-files (multi-spectrum) |
| 3   | 0x08 | TRANDM  | Z values random (unordered) |
| 4   | 0x10 | TORDRD  | Z values in reverse order |
| 5   | 0x20 | TALABS  | Axis label strings in fcatxt |
| 6   | 0x40 | TXYXYS  | Each sub-file has own X array |
| 7   | 0x80 | TXVALS  | Explicit X array precedes sub-files |

When `TXVALS` is set, `fnpts` float32 values appear immediately after the
main header (before the first sub-header). When `TMULTI` is not set the file
contains exactly one sub-spectrum.

### 2.2 X Axis Type Codes (fxtype)

| Value | Unit |
|-------|------|
| 0 | Arbitrary |
| 1 | Wavenumber (cm⁻¹) |
| 2 | Micrometers (µm) |
| 3 | Nanometers (nm) |
| 4 | Seconds |
| 5 | Minutes |
| 6 | Hertz |
| 7 | Kilohertz |
| 8 | Megahertz |
| 9 | Mass (m/z) |
| 10 | Parts per million |
| 11 | Days |
| 12 | Years |
| 13 | Raman shift (cm⁻¹) |
| 14 | eV |

### 2.3 Y Axis Type Codes (fytype)

| Value | Unit |
|-------|------|
| 0 | Arbitrary |
| 1 | Interferogram |
| 2 | Absorbance |
| 3 | Kubelka-Munk |
| 4 | Counts |
| 5 | Volts |
| 11 | Percent |
| 12 | Intensity |
| 128 | Transmittance |
| 129 | Reflectance |
| 131 | Emission |

### 2.4 Date Encoding (fdate)

Packed uint32 LE:

```
bits 31–20  Year    (12 bits, 0 = no date)
bits 19–16  Month   (4 bits, 1–12)
bits 15–11  Day     (5 bits, 1–31)
bits 10–6   Hours   (5 bits, 0–23)
bits  5–0   Minutes (6 bits, 0–59)
```

Python decode:
```python
minutes = fdate & 0x3F
hours   = (fdate >> 6) & 0x1F
day     = (fdate >> 11) & 0x1F
month   = (fdate >> 16) & 0x0F
year    = (fdate >> 20) & 0xFFF
```

---

## 3. Sub-Header (32 bytes)

Each sub-spectrum is preceded by a 32-byte sub-header.

| Offset | Size | Type   | Field     | Description |
|--------|------|--------|-----------|-------------|
| 0      | 1    | u8     | subflgs   | Sub-file flags (0x80 = float Y) |
| 1      | 1    | i8     | subexp    | Y exponent override (signed; used when main fexp = 0) |
| 2      | 2    | u16 LE | subindx   | Integer index of sub-file |
| 4      | 4    | f32 LE | subtime   | Z axis value for this sub-file (e.g. time in seconds) |
| 8      | 4    | f32 LE | subnext   | Z axis value of next sub-file |
| 12     | 4    | f32 LE | subnois   | Noise value (0 if not measured) |
| 16     | 4    | u32 LE | subnpts   | Number of points (0 = use main fnpts) |
| 20     | 4    | u32 LE | subscan   | Number of co-added scans |
| 24     | 4    | f32 LE | subwlevel | W axis value |
| 28     | 4    | bytes  | subresv   | Reserved |

---

## 4. Y Value Encoding

Y values are stored as 4 bytes per point, immediately following the
sub-header.

### Float32 encoding (most common)

When `fexp == -128` (stored as byte 0x80),
Y values are **IEEE 754 float32 LE**:

```python
y = np.frombuffer(data, dtype='<f4', count=fnpts).astype(np.float64)
```

### Integer encoding

When `fexp != 0x80`, Y values are **int32 LE** scaled by an exponent.
The effective exponent comes from the sub-header's `subexp` field if the main
`fexp == 0`, otherwise from the main header's `fexp`:

```python
effective_exp = sub_hdr["subexp"] if fexp == 0 else fexp
y = np.frombuffer(data, dtype='<i4', count=fnpts).astype(np.float64)
y *= 2.0 ** (effective_exp - 32)
```

---

## 5. X Axis Values

When `TXVALS` flag is **not** set (the common case), X values are evenly
spaced and computed from the header:

```python
x = np.linspace(ffirst, flast, fnpts)
```

When `TXVALS` is set, `fnpts` float32 values follow the main header directly,
before the first sub-header. These allow non-uniform X spacing:

```python
x = np.frombuffer(f.read(fnpts * 4), dtype='<f4').astype(np.float64)
```

---

## 6. Log Block

The optional log block starts at byte offset `flogoff` (from the main header
field). It has a 64-byte header:

| Offset | Size | Type   | Field    | Description |
|--------|------|--------|----------|-------------|
| 0      | 4    | u32 LE | logsizd  | Size of binary log data (bytes) |
| 4      | 4    | u32 LE | logsizm  | Total log block size (bytes) |
| 8      | 4    | u32 LE | logtxto  | Byte offset within log block where text starts |
| 12     | 4    | u32 LE | logbins  | Size of binary data (bytes) |
| 16     | 4    | u32 LE | logdsks  | Disk block size |
| 20     | 44   | bytes  | logreserv| Reserved |

Log text starts at `flogoff + logtxto` and is CR/LF-terminated
`key=value` pairs encoded in Latin-1.

Example (from EVK Sentronic instrument):
```
Channel=channel1
Timestamp=2020/07/09 18:18:59.880
AverageNumber=0
IntegrationTime=0
```

---

## 7. Python Reader

```python
import struct
import numpy as np

MAIN_HDR = 512
SUB_HDR  = 32
TXVALS   = 0x80
TMULTI   = 0x04
FLOAT32  = -128  # fexp sentinel (0x80 as signed byte)

def load_spc(path):
    with open(path, 'rb') as f:
        hdr = f.read(MAIN_HDR)
        fversn = hdr[1]
        assert fversn == 0x4B, f"Unsupported version 0x{fversn:02X}"

        ftflgs = hdr[0]
        fexp   = struct.unpack_from('b', hdr, 3)[0]  # signed byte
        fnpts  = struct.unpack_from('<I', hdr, 4)[0]
        ffirst = struct.unpack_from('<d', hdr, 8)[0]
        flast  = struct.unpack_from('<d', hdr, 16)[0]
        fnsub  = struct.unpack_from('<I', hdr, 24)[0]

        # X axis
        if ftflgs & TXVALS:
            x = np.frombuffer(f.read(fnpts * 4), dtype='<f4').astype(float)
        else:
            x = np.linspace(ffirst, flast, fnpts)

        # Sub-spectra
        spectra = np.empty((fnsub, fnpts))
        for i in range(fnsub):
            sub = f.read(SUB_HDR)
            subexp = struct.unpack_from('b', sub, 1)[0]  # signed byte
            y_raw = f.read(fnpts * 4)
            exp = subexp if fexp == 0 else fexp
            if exp == FLOAT32:
                spectra[i] = np.frombuffer(y_raw, dtype='<f4').astype(float)
            else:
                spectra[i] = (np.frombuffer(y_raw, dtype='<i4').astype(float)
                              * 2.0 ** (exp - 32))

    return spectra, x   # shape: (n_spectra, n_points)
```

---

## 8. Validated File Examples

| File | fnsub | fnpts | X range | fxtype | fytype |
|------|-------|-------|---------|--------|--------|
| `Avicel_PH101_20200709_061859_902.spc` | 1 | 1101 | 1100–2200 nm | 3 (nm) | 2 (abs) |
| `20160428_101549_Kruisz_Avicel112.spc` | 9 | 551 | 1100–2200 nm | 3 (nm) | 2 (abs) |

Both files use `fexp=0x80` (IEEE float32), no TXVALS, TMULTI set only in the
multi-spectrum file, fversn=0x4B.
