# Wavelength tables (approved 2026-11-17; built 2026-11-19 for 1.7.2)

## Problem

Scans streamed with the Kestreleye software (`stream_hsi.h5`) hold only
`data` (frames × 312 × 176, uint16): no wavelengths, no camera serial, no
binning, no attributes. engiNIR shows band numbers 0–175 and warns on every
open. The camera setting is always the same in these experiments.

The vendor software writes EVK `.eiff` files with metadata. Every `.eiff`
checked so far (EVK-AlphaVision, serial 220332240003) has 220 bands,
991.675–1677.125 nm, 3.125–3.15 nm apart. The 176 Kestreleye bands are most
likely 176 consecutive bands of those 220; which ones is not known yet.

## Input (found 2026-11-19)

The Kestreleye scans come from an **EVK-Helios-EC-32** (serial
181642214419), not the AlphaVision of the earlier `.eiff` files. Its
acquisition setting (Helios configuration, Acquisition): 176 spectral
pixels from wavelength start 36, 312 spatial pixels. The vendor software
writes `.eiff` files with this setting and their wavelengths
(`FrischerAbgleich.eiff` and `Sensor_roh.eiff`, 2026-11-19, identical):
176 bands, 1041.75–1589.5 nm, 3.125 or 3.15 nm apart.

Extracted to `src/enginir/_resources/wavelength_tables/kestreleye_176.csv`
(rounded to 0.001 nm; the file's float noise is below 1e-12 nm). The band
order was checked against a Kestreleye `stream_hsi.h5` of 10 % API tablets:
its derivative absorbance spectrum matches tablets measured on the
AlphaVision's known axis with r = 0.97 at a shift under one band; reversed,
r = −0.71.

`FrischerAbgleich.eiff` is a white balance (flat at about 3100 counts, the
camera's white-balance target), so the camera applies the white balance to
the data it sends, Kestreleye streams included.

## Decisions (user, 2026-11-19)

- **D1. Set Wavelengths changes the dataset and its ROIs.** Only band-number
  axes are replaced by the table; the change is logged and is an identity
  exception of the schema contract, like Locate. Models keep their stored
  band-number axis (prediction builds its steps from the model's own input
  contract) and predict as before; a refit gives nm.
- **D2. The viewer preselects.** A file without wavelengths whose band
  count matches exactly one table: the Add Dataset dialog preselects it and
  the Data check offers it for existing datasets. `Project.add_dataset`
  uses a table only when one is given.
- **D3. Release 1.7.2**, with the 1.7.2 backlog. No format change: the table
  name and id are new optional dataset attributes.

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
