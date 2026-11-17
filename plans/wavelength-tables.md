# Wavelength tables (planned after 1.6.0; approved 2026-11-17)

## Problem

Scans streamed with the Kestreleye software (`stream_hsi.h5`) hold only
`data` (frames × 312 × 176, uint16): no wavelengths, no camera serial, no
binning, no attributes. engiNIR shows band numbers 0–175 and warns on every
open. The camera setting is always the same in these experiments.

The vendor software writes EVK `.eiff` files with metadata. Every `.eiff`
checked so far (EVK-AlphaVision, serial 220332240003) has 220 bands,
991.675–1677.125 nm, 3.125–3.15 nm apart. The 176 Kestreleye bands are most
likely 176 consecutive bands of those 220; which ones is not known yet.

## Input still missing

The 176 wavelengths. The user will provide an `.eiff` (and, if possible, the
Kestreleye spectral window) to extract them. Ways to get them, best first:
the Kestreleye configuration (first row and number of rows), the vendor's
calibration, or aligning spectra of the same material from both programs
(an estimate, only to check the others).

## Design

- **Table file:** CSV `band,wavelength_nm` with a name ("Kestreleye 176
  bands") and the camera. Built-in tables ship in the package; users add
  theirs in `~/.enginir/wavelength_tables/`.
- **Import:** the Add Dataset dialog gets a **Wavelengths** dropdown: "From
  the file", "Band numbers", and every table with the file's band count. A
  file without wavelengths and exactly one matching table preselects it.
- **Project:** the values are copied into the dataset record with the
  table's name and a content id; a project never depends on a table file
  on one machine. A changed table is a new id; nothing stored is
  overwritten.
- **Prediction:** reads the dataset's stored axis through the same function
  as calibration (`transform_spectra_to_signal_domain` path unchanged).
- **Existing projects:** Project > Set Wavelengths… applies a table to
  datasets added with band numbers. Wavelengths are metadata, not a
  prediction gate (G18): existing models stay valid; band numbers vs nm only
  warns.
- **Checks:** a Data-step check "wavelengths are band numbers" whose fix
  applies the matching table.
- **Library API:** `Project.add_dataset(path, wavelength_table=...)` (or
  `wavelengths=`), and the same for the set-wavelengths action.
- **Later:** store the table's camera as `sensor_name`, reserved in the v3
  contracts for calibration transfer.

## Contracts and docs

Attachment contract and `docs/hdf5_format.md`: the dataset record's
wavelength source (table name and id). `docs/contract_coverage.md`: a row,
and a gap if anything is deferred. User guide: import dialog, Set
Wavelengths. The false-colour bands (1650/1300/1050 nm) only work once the
axis is in nm.
