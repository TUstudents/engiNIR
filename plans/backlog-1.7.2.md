# Backlog: engiNIR 1.7.2 feature list

Status: collected (user, 2026-11-19). Not planned yet; each item gets its
decisions before work starts. 1.7.1 is the startup speedup
(`plans/startup-speed-1.7.1.md`).

## Items

1. **Blosc compression.** engiNIR reads Blosc-compressed HDF5 data files
   (`hdf5plugin`, e.g. Kestreleye raw files), but writes its own HDF5 files
   with gzip only. The project store (`project_store`, `_hdf5_groups`) and
   `save_hdf5` (`io/hdf5_io.py`, `compression="gzip"` or `"lzf"`) write
   them. To decide:
   - which files use Blosc (project files, `save_hdf5`, or both);
   - which codec and level (Blosc with LZ4 or Zstandard, with shuffle);
   - whether it becomes the default.

   A Blosc file needs `hdf5plugin` to be read, which tools without it
   (HDFView, MATLAB) may not have; gzip is readable everywhere. A change of
   the project files' compression is not a format change (the reader
   decodes either), but the contract and `docs/hdf5_format.md` name the
   default.
2. **ROI name placeholders.** The ROI grid dialog's name pattern lists its
   placeholders only in a tooltip (`viewer/_roi_grid.py`: `{file}`,
   `{row}`, `{column}`, `{n}`; `Project.add_rois`/`roi_grid` in
   `_workflow.py`). Show them in the dialog, and add **`{folder}`**: the
   data file's folder name. Also check the single-ROI dialog
   (`NewROIDialog`) for the same list.
3. **About dialog:** add the citation (the text of the README, with the
   concept DOI 10.5281/zenodo.23235907) and the GitHub link
   (https://github.com/TUstudents/engiNIR). Today it shows the version
   (`viewer/_viewer.py`, "About engiNIR").
4. **ROI name and file name in the grid.** ROIs have a name and a file name
   field (`TrainingROI.name`, `source_file`). Use the name for the grid.
   *To clarify:* which grid? The Project ROIs mosaic labels its tiles with a
   shortened name (`_project_rois._short_names`), and the ROI table, the
   ROI grid dialog and the Predict per-ROI plot also show names.
5. **Update `docs/workspace_domain_model.md`.** It still describes the state
   before several changes (looked at 2026-11-19):
   - "HDF5 v3 project files", where the format is now `enginir-1.7`;
   - `TrainingROI` without its `job` (calibration / prediction) and
     `sample` label;
   - `ModelRun` without `fit_spec`, `software` and the core limits
     `dcrit_095`/`t2_crit_095`;
   - `PredictionRun` with a `spatial_map` artifact, which G31 removed, and
     without the staging under `/staging/`.

   It also uses the internal class names (`SampleSet`,
   `SignalDomainContract`) without the terms of the terminology contract
   (calibration set, input contract). Compare it with the contracts and
   `docs/hdf5_format.md`, which are current.
