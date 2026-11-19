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
6. **The PLS and PCR component curve up to 10 components.** Each fit
   already stores its component curve (`component_curve`, RMSECV and Q²
   per component count; `modeling/_components.py`, `_gram_fit.py`,
   `_service.py`), up to `A_max = max(n, min(15, max(1, G - 2), M))`
   (`default_max`; `contracts/sample-set-model-run-contract_v3.md`, `components/max`). The user (2026-11-19): only up to
   10 components; 5 is often the best Pareto option. So the cap 15 becomes
   10. *To decide:* whether a fit with a given count above 10 still extends
   the curve to that count (today `max(n, …)` does), and whether the
   suggested count (`suggested_*`, `guarded_choice`) changes with the
   shorter curve. A contract change, with the tests that pin 15.
7. **The log respects the logging level; the default is INFO.** No DEBUG
   message appears at INFO. The log file already follows the level of
   Tools > Preferences ("Log file level", default INFO;
   `viewer/_preferences.py`, `enable_viewer_logging` in `viewer/_app.py`).
   The **Log panel** does not: its sink is added with `level="DEBUG"`
   (`viewer/_viewer.py`, `add_log_sink(self._log_panel.sink, ...)`), so
   it shows every debug message. It should use the same level and follow a
   change of the setting (today the setting applies from the next start).
   Also check that no message logged at INFO should really be DEBUG (the
   "Startup timing" lines are DEBUG and INFO mixed).
8. **A better workflow for Add ROI and Add ROI Grid** (to discuss). The
   user (2026-11-19): adding ROIs does not switch to the Data file view, the
   dialog may not appear, and one has to use the menu where buttons could
   be. As built (`viewer/_eda.py`): the buttons Add ROI (checkable: draw on
   the Spatial Image) and Add ROI Grid (the dialog with a live preview on
   the canvas) sit in the ROI group of the Explore settings, with Delete
   ROI, ROI Table and Export; the ROI menu presses the same buttons
   (`_menu_button_action`). Both are enabled only while a dataset of the
   project is shown (`refresh_attachment_state`); a project with ROIs opens
   on the Project ROIs tab, where no image is shown, so both are disabled
   there and nothing leads to the Data file tab. Ideas to decide:
   - pressing either from the Project ROIs tab (or the ROI menu) switches
     to the Data file tab, showing the last shown dataset or asking for one,
     then starts drawing or opens the grid dialog;
   - buttons where the user is: on the Project ROIs tab (per dataset tile
     group, "Add ROIs to this dataset") and on the empty Data file page;
   - the workflow strip's ROIs step leads to the same place.
9. **The image opens as Reset View shows it.** The user (2026-11-19):
   Reset View changes the plot's size and uses the view better; that should
   be the default. As built (`viewer/_canvas.py`): showing an image calls
   `autoRange()` with pyqtgraph's default padding (`set_image`), possibly
   while the page does not have its final size yet; **Reset View**
   (`reset_view`, the button in the Spatial image header) calls
   `autoRange(padding=0.02)` on the view as it is. The fix: one function
   for both, also applied when the view first gets its size (and after a
   maximize or a restore of the plot), so a new image looks like Reset View
   without the button. Check the Sensor frame and the Spatial profile, which
   share the image's row axis (`link_rows`).
10. **Glossary update** (`docs/glossary.rst`, with
    `contracts/terminology_v3.md`, which fixes the terms). The user
    (2026-11-19): some items are not defined, e.g. Frame. *To clarify:*
    Frame is in the glossary, but only together with Row ("the two spatial
    axes of a push-broom cube"), so where was it missed: in the viewer (the
    *Sensor frame* plot is a different thing: one frame, bands by rows), a
    frame range of a dataset, or a term the glossary should define on its
    own? Terms used in the viewer and the guides that the glossary lacks
    (looked at 2026-11-19), to check against the contract:
    - data: project, data file, cube, push-broom, wavelength axis, frame
      range, dark and white reference, detector levels (black/white level);
    - the Explore plots: spatial image, false-color image, sensor frame,
      band image, spatial profile, ROI grid;
    - preprocessing by name: SNV, MSC, EMSC, Savitzky-Golay, detrend;
    - modeling: cross-validation, hold-out, Kennard-Stone, component curve;
    - results: R², Q², RMSEC, RMSECV, RMSEP, bias, VIP, scores, loadings,
      coefficients, checks and the verdict.
11. **The welcome window's tagline.** It says "Reference correction models
    for NIR spectra and hyperspectral images." (`viewer/_welcome.py`, the
    subtitle under "engiNIR"), which is wrong. The format 1.7 rename (commit
    `7ce1f0c`) turned "Calibration models ..." into it, although
    "calibration" meant model calibration there. A search of the rename's
    removed lines (2026-11-19) found no other text of that kind; this is the
    only one. Use the engiNIR tagline: the citation's title is "A universal
    engineering tool for near-infrared spectra and hyperspectral imaging"
    (`CITATION.cff`). *To decide:* that wording, or another (the PyPI
    description in `pyproject.toml` and the Windows shortcut's description
    in `viewer/_windows_shell.py` say it differently).
