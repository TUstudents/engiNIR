Glossary
========

The terms engiNIR uses, as fixed by its terminology contract. Each entry
gives the meaning, the words other tools and standards use (*related* marks
a similar but not identical quantity) and, where it differs, the name in the
Python API and the project file.

Data
----

.. glossary::

   Dataset
      One imported measurement: a hyperspectral cube or a file of point
      spectra, with its own :term:`reference correction`. A dataset may be a
      frame range of a file; "file" means the file on disk.
      Elsewhere: SIMCA *dataset*, The Unscrambler *data set*, PLS_Toolbox
      *DataSet object*, ENVI *image* or *file*.
      API: ``DatasetRef``, ``add_dataset``, ``dataset_id``.

   Frame
      One line the push-broom camera reads as the sample moves: the first axis
      of a cube ``(frames, rows, bands)``, the scan direction. A frame holds a
      spectrum for every :term:`row`. A *frame range* is a run of frames, e.g.
      the white tile at the end of a scan. Elsewhere: ENVI *line*.

   Row
      A position along the camera's slit: the second axis of a cube
      ``(frames, rows, bands)``. Elsewhere: ENVI *sample* (ENVI's *samples* are
      rows here, not :term:`samples <sample>`).

   Band
      One spectral channel, with its wavelength in nm when known.
      Elsewhere: SIMCA *X variable*, The Unscrambler *variable*.

   Pixel
   Spectrum
      One spectrum: *pixel* where the image matters, *spectrum* in general. A
      model is fitted on pixels or on ROI means ("Fitted on").
      Elsewhere: SIMCA *observation*; The Unscrambler and PLS_Toolbox
      *sample* (one row of the data matrix).

   ROI
      Region of interest: a marked area of a dataset. The reporting unit:
      target values, fold groups and the metrics per ROI use the mean of its
      pixels. Elsewhere: ENVI *ROI*. API: ``TrainingROI``.

   Sample
      The physical sample (tablet, blend, batch portion) whose ROIs share a
      sample label and can be held out together in cross-validation. Never a
      pixel, spectrum or ROI. In The Unscrambler and PLS_Toolbox *sample*
      means one spectrum.

Signal
------

.. glossary::

   Raw counts (DN)
      The uncorrected detector values, before reference correction.
      Elsewhere: ENVI *DN* (digital number). API: signal unit ``raw_dn``.

   Reference correction
      Converting raw counts to reflectance with dark and white references or
      with detector levels. Elsewhere: HSI vendors *reflectance
      calibration*; related: ENVI *Flat Field Correction*, *Empirical Line*.
      API and project file: ``ReferenceCorrection``, ``reference_correction_id``,
      ``/reference_corrections/`` (since 1.7; ``CalibrationSpec``,
      ``calibration_id`` and ``/calibrations/`` before).

   Reflectance
      The reference-corrected signal R, nominally between 0 and 1.

   Dark reference
   White reference
      The detector's response with the shutter closed (dark) and on a white
      tile (white), per row and band; averaged over frames when taken from a
      frame range. Reflectance is R = (x − dark) / (white − dark).

   Detector levels
      A black level and a white level (128 and 3500 counts by default) that
      convert raw counts to reflectance without references: R = (x − black) /
      (white − black). For cameras that apply their own white balance.

   Wavelength table
      The wavelengths of a camera setting, for files that store none (band
      numbers only, e.g. a Kestreleye ``stream_hsi.h5``); copied into the
      dataset when it is chosen. API: ``enginir.io.wavelength_tables``.

   Absorbance
      log10(1/R) computed from reflectance; for diffuse reflectance the
      apparent (pseudo-)absorbance.

Preprocessing
-------------

.. glossary::

   Preprocessing recipe
      The ordered chain of preprocessing steps of a model, stored with it.
      Elsewhere: SIMCA *spectral filters*, The Unscrambler *transformations*
      or *pretreatments*, PLS_Toolbox *preprocessing*. API:
      ``PreprocessingRecipe``.

   Scatter correction
      SNV, MSC, EMSC and Detrend (Rinnan et al., 2009). "Normalization"
      means scaling to a unit norm, or SIMCA's normalized DModX.

   Smoothing and derivatives
      Savitzky-Golay smoothing and 1st and 2nd derivatives.

   SNV
      Standard normal variate: each spectrum minus its mean, divided by its
      standard deviation.

   MSC
   EMSC
      Multiplicative scatter correction: each spectrum regressed on a
      reference spectrum (the mean of the calibration set) and corrected for
      offset and slope. EMSC adds polynomial baseline terms. Both are fitted
      inside every cross-validation fold.

   Savitzky-Golay
      A local polynomial fit over a moving window that smooths a spectrum or
      gives its 1st or 2nd derivative.

   Detrend
      Removes a polynomial baseline (order 1: a straight line) from each
      spectrum.

Modeling
--------

.. glossary::

   Target
      The quantity a model predicts, with an optional unit. Elsewhere: ASTM
      E1655 *property* or *constituent*; SIMCA and The Unscrambler *Y
      variable* or *response*; PLS_Toolbox *y block*.

   Target value
      The measured value of a target for an ROI, from the reference method.
      Elsewhere: ASTM E1655 *reference value*; SIMCA *observed* Y. API:
      ``TargetAssignment``.

   Calibration
      Building a model from the :term:`calibration set` (fitting and
      cross-validating it), as in ASTM E1655.

   Calibration ROI
   Prediction ROI
   Prediction
      The two roles of an ROI (the ROI table's "Set"). Calibration ROIs build
      models; prediction ROIs are predicted and never enter a model; those
      with target values give the RMSEP (an external test set). Elsewhere:
      SIMCA *prediction set*. Project file and API: ``job = calibration |
      prediction``.

   Calibration set
      The frozen calibration ROIs, with their target values and the
      preprocessing recipe, that one model was fitted on. Elsewhere: SIMCA
      *workset*; USP <1039> and The Unscrambler *calibration set*. API:
      ``SampleSet``.

   Calibration range
      The range (minimum to maximum) of the target values in the calibration
      set; the default color scale of a prediction map.

   Fold group
      The unit cross-validation holds out together: by ROI, by sample label
      or by target value.

   Pixel subsample
      The pixels a fit draws from every ROI (the same number from each, by a
      seed), so that every ROI weighs the same; "sample" alone is the
      physical sample. API: ``fit_spec={"pixels": {"per_roi": n, "seed": s}}``.

   Model
      One fitted model with its id, never overwritten. API: ``ModelRun``.

   Input contract
      What a dataset must provide for a model to predict it: input format,
      signal unit and the reference correction it requires. The wavelength
      axis is not part of it (a differing axis only warns). API:
      ``SignalDomainContract``.

   LV
   PC
   Components
      Model size: latent variables (LV) for PLS, principal components (PC)
      for PCR; *components* in general. Elsewhere: SIMCA *components*, The
      Unscrambler *factors*.

Diagnostics and results
-----------------------

.. glossary::

   Metrics per ROI
   Metrics per pixel
      Every metric (RMSEE, RMSECV, RMSEP, R², Q², bias, RPD, RPIQ) names its
      level: "RMSECV (per ROI)", "RMSECV (per pixel)". Per ROI is the
      reported level. API: ``rmsecv``, ``pixel_rmsecv``.

   DModX
   DCrit
      The normalized distance of a spectrum to the model (its residual,
      scaled by the calibration residual s0) and its critical limit: DModX
      for calibration spectra, DModXPS for predicted spectra. Elsewhere:
      SIMCA *DModX*, *DModXPS*, *DCrit*; related: PLS_Toolbox *Q residuals*,
      The Unscrambler *F-residuals* (different scaling and limits).

   Hotelling's T²
   T²
   T²Crit
      The distance of a spectrum within the model plane and its critical
      limit: T² for calibration spectra, T²PS for predicted spectra.
      Related: leverage (see prediction interval).

   Outside the model
   Above DCrit
   Above T²Crit
      Above DCrit or above T²Crit (at 0.95, SIMCA's default). A statement about the spectrum,
      not a judgement that it is faulty; engiNIR does not say "outlier".

   Excluded pixels
      Pixels left out by a distance limit (DModX in the fit, DModXPS in a
      prediction): from the fit (the fit specification)
      or from the statistics of a prediction ROI (the Predict page).

   Prediction interval
      ŷ ± q · s · √(1 + h), the leverage formula of ASTM E1655 with the R
      calibration ROIs as the independent samples: h = 1/R + T²PS/(R − 1)
      for a pixel, h̄ = 1/R′ + d²/(R′ − 1) for the mean of a prediction ROI
      (:ref:`prediction-intervals`). The standard's h = 1/N + T²/(N − 1)
      over N calibration spectra would vanish for image pixels. Elsewhere: SIMCA *YPredPS confidence
      interval* (jack-knifed, a different number), The Unscrambler
      *Y-deviation* (a different estimate). API:
      ``prediction.interval_map(level)``.

   Prediction map
      The image of the predicted values of a dataset. Elsewhere: NIR
      chemical imaging *prediction image*, *chemical image*. API:
      ``prediction.map``.

   R²
      Coefficient of determination, 1 − Σe² / Σ(y − ȳ)², of the fitted values
      (per ROI or per pixel). Elsewhere: SIMCA *R2Y*, The Unscrambler
      *R-square (calibration)*.

   Q²
      R² of the cross-validated predictions: how well the model predicts
      ROIs it was not fitted on. Elsewhere: SIMCA *Q2*; The Unscrambler
      *R-square (validation)*.

   RMSEC
   RMSECV
   RMSEP
      Root mean squared error, √(mean e²), of the fitted values (C), the
      cross-validated predictions (CV) and the predictions of ROIs with a
      value that the model was not fitted on (P). No degrees-of-freedom
      correction. Elsewhere: ASTM E1655 *SEC*, *SECV*, *SEP* (related, with
      degrees of freedom); SIMCA *RMSEE*, *RMSECV*, *RMSEP*.

   Bias
      The mean of the errors e = predicted − reference: positive when the model
      over-predicts.

   Component curve
      For 1, 2, … up to 10 components (or the count a fit asks for), the
      cross-validated RMSECV, Q², R²Y and R²X of a PLS or PCR model; it
      suggests the number of components. API: ``model.component_curve``.

   Scores
      The coordinates of a spectrum in the model's components (t). Elsewhere:
      SIMCA *t*, The Unscrambler *scores*.

   Loadings
      How much each band contributes to a component (p for PLS and PCR, w for
      PLS weights). Elsewhere: SIMCA *p*, *w*, *w\**.

   Coefficients
      The regression coefficients b per band: the prediction is
      ŷ = b₀ + Σ bⱼ xⱼ after preprocessing. Elsewhere: SIMCA *CoeffCS*
      (scaled and centred; engiNIR shows both).

   VIP
      Variable importance in projection of a PLS model, per band: bands with
      VIP above 1 matter more than the average band. Elsewhere: SIMCA *VIP*.

Viewer
------

.. glossary::

   Spatial image
      The image of a scan, frames by rows, in false color: the Explore plot
      where ROIs are drawn.

   False-color image
      Three bands shown as red, green and blue (1650, 1300 and 1050 nm when
      the wavelengths are known).

   Band image
      The image of one band, frames by rows.

   Sensor frame
      One :term:`frame` as the camera's sensor sees it: bands by rows, the
      spectra of every row at one moment. Not a frame range.

   Spatial profile
      The value of the shown band along the rows of the shown frame.

   Check
      One item the library evaluates about a project (a step, a severity, a
      message), often with a fix that the viewer offers as a button. API:
      ``Project.checks()``, ``Model.checks()``.

   Verdict
      The library's judgement of a model: good, usable with care or not
      usable, with its reasons. API: ``Model.verdict()``.
