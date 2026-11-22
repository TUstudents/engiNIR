# Plan: spatial pixel binning, an option of Add Dataset

Status: built on branch `pixel-binning` (2026-11-22), phases 1 to 5; not merged. Binning is an option of the
dataset, applied on every read (no derived file, no extra tool); a file to
predict is binned like the model; D1 to D4 below.

## Goal and acceptance test

`k × k` blocks of pixels (k rows across track, k frames along track) become one
pixel; k = 1 (none), 2, 3, 4. Bands and wavelengths do not change.

The feature is right when **a dataset added with `binning=k` gives the same
result as the same file pre-binned with NumPy and added without binning**: the
same ROI pixels, references, fit metrics and prediction map. That test is the
proof that the rest of the pipeline needs no change, and it is written first.

## The rule: bin the raw counts, keep the dtype

- The mean of each block, taken on the **raw counts before any reference
  correction**, like a sensor that bins its read-out. Reflectance is then
  (mean counts − mean dark) / (mean white − mean dark), what a binned sensor
  would give.
- Integer cubes: the mean is rounded (`np.rint`) and cast back to the cube's own
  dtype (uint16 stays uint16). The pipeline decides "raw counts" vs "already
  corrected" by dtype (`kind == "f"`, `_workflow.py:846`, `_state.py:1087`), so
  a float mean would be misread as corrected data, and int16 would wrap counts
  above 32767. The rounding error is at most 0.5 count (about 0.3 rms), far
  below the shot noise at 3100.
- Float (already corrected) cubes: the mean in float32, no rounding.
- The detector levels (128, 3100) stay valid: a mean of counts has the
  same scale as one count.
- Incomplete edge blocks are dropped: a 100-frame file binned by 3 has 33
  frames. `cube_shape` is the binned shape.

## One seam: the readers take `binning`

1. `enginir.io.bin_blocks(cube, k)`: a pure function on a `(frames, rows,
   bands)` array (reshape-mean, rounding by dtype, edge crop).
2. `read_header`, `read_frames`, `load_spectra` and the three
   `iter_*_frame_chunks` take `binning: int = 1`. `read_frames(first, last)`
   takes **binned** frame indices and reads raw frames `first*k .. (last+1)*k − 1`.
   The chunk iterators read in multiples of k frames so chunked equals one-shot.
3. Every call site passes the dataset's binning. There are about 12 (`_workflow`
   ROI and reference reads, `_header_mismatch`, Locate, the viewer's load and the
   Add Dataset preview, the prediction chunk reader through its existing
   `loader_kwargs`). A test fails when a call site reads a binned dataset
   without it: fixture of a dataset at k = 2, every public path compared with
   the pre-binned file.

## Dataset and references

- `DatasetRecord.binning: int = 1`, stored only when k > 1, so the ids and
  content of existing datasets do not change and the format stays
  `enginir-1.7`. Part of the dataset's content: never changed (D2).
- **One binning per file per project.** The project finds a dataset by its path
  (`Project._dataset_ref(path)`, Locate, `dataset_id_of`); a second dataset on
  the same path with another binning would break that. Adding a registered
  file with another binning is refused, naming the dataset.
- **Reference frames inside a scan are given in raw frames.** Dark frames 0-1 at
  k = 4 would otherwise take binned frames 0-1, raw frames 0-7, mixing dark with
  sample frames. The reference is the mean of its raw frames (frame binning
  does not matter for a mean), binned over the rows only. The Add Dataset dialog
  already shows raw frames (`_data_files.py:383`).
- **Reference files** (`dark=path`) are binned over the rows the same way
  (`_reference`, `_workflow.py:2163`), so the reference shape check
  (`_signal_domain.py:102`) holds unchanged: `(binned rows, bands)`.
- ROI reference snapshots are taken from the binned references, so they stay
  per binned pixel.

## Models and prediction

- `SignalDomainContract.binning: int = 1`, stored only when k > 1 (existing
  input contracts keep their ids). The fit takes it from the calibration set;
  a calibration set mixing binnings is refused, like mixed signals, because a
  model has one binning to predict with.
- `check_signal_domain_compatibility` adds one rule: the dataset's binning
  equals the model's.
- Predicting a file **not registered**: the in-memory dataset `resolve_dataset`
  builds gets the model's binning; its references (levels, files) are binned
  as above.
- Predicting a **registered** file with another binning is refused (D1).
- Prediction ROIs are stored with their dataset's binning, so they need nothing.

## Viewer

- Add Dataset: a "Binning" row (None, 2 × 2, 3 × 3, 4 × 4) under the
  Wavelengths row; the preview image shows the binned image.
- Explore shows the binned image; ROIs are drawn in binned pixels; the status
  line says "binned 2 × 2"; the pixel size in mm is multiplied by k.
- Properties of a dataset and of a model show the binning.

## What I left out of the earlier draft, and why

- Separate row and frame factors: square k × k is what was asked; one integer
  is simpler. Can be widened later without a format change.
- A "mixed binnings" Checks warning: the fit refuses instead (one rule, one place).
- Sum binning: needs levels that scale with k²; not asked.

## Decisions (user, 2026-11-22)

- **D1. A registered file predicted with a model of another binning is
  refused**, naming both binnings: its stored reference correction has the
  other row count.
- **D2. A dataset's binning never changes** after Add Dataset. No Set Binning
  action; another binning of the same file needs a new project.
- **D3. Saturation is not tracked through binning:** the "above white" share is
  computed on the binned values, like every other statistic.
- **D4. Edge blocks are accepted and documented:** blocks start at frame 0, so in
  a whole-scan prediction map the blocks at the reference frames mix reference
  and sample frames; the user guide says so.

## Phases (each with its tests, fault-injected)

1. `bin_blocks` and the `binning` argument of the readers; NumPy reshape-mean
   oracle, dtype, rounding, edge crop, chunked equals one-shot, EIFF parallel
   readers.
2. `DatasetRecord.binning`, Add Dataset (`Project.add_dataset(binning=)`),
   references in raw frames, the attachment check, Locate. The acceptance test
   for ROIs and references.
3. `SignalDomainContract.binning`, the fit refusal, prediction. The acceptance
   test for fit and prediction.
4. Viewer: the dialog row, Explore, Properties, pixel size.
5. Contracts (dataset attachment, dataset reference correction, signal domain,
   project schema §5), `hdf5_format.md`, `coverage.md`, the user guide and
   CHANGELOG.

## Risks (checked)

- engiNIR 1.7.4 opening a project with a binned dataset (checked with the
  released 1.7.4): it opens, refits the model from the stored ROIs with the
  same RMSECV, and refuses to predict the file ("it has shape (24, 8, 6), the
  dataset has (12, 4, 6)"). No silent unbinned read, so no format bump.
- A read site that misses the binning reads unbinned pixels: the acceptance
  tests go through every public path (ROIs, references, Locate, fit,
  prediction, the viewer's image), each fault-injected.
- Changed from the plan: the Add Dataset preview stays unbinned, because the
  reference frames are chosen there in frames as stored.
