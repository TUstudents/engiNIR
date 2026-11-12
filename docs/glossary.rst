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
   Row
      The two spatial axes of a push-broom cube ``(frames, rows, bands)``. A
      frame is one line read by the camera as the sample moves; a row is a
      position along the slit. Elsewhere: ENVI *lines* and *samples* (ENVI's
      *samples* are rows here, not :term:`samples <sample>`).

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
      API and project file: ``CalibrationSpec``, ``calibration_id``,
      ``/calibrations/`` (to be renamed in 2.0).

   Reflectance
      The reference-corrected signal R, nominally between 0 and 1.

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
      Above DCrit or above T²Crit (at 0.99). A statement about the spectrum,
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
