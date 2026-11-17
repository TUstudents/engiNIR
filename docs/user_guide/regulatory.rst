Regulatory context
==================

engiNIR is built to support NIR methods that are developed and maintained
under pharmaceutical guidelines. It provides the evidence these guidelines
ask for: a documented model, its performance, the detection of samples
outside the calibration, and traceable records. It does not make a method
compliant on its own; the method, its validation and the system it runs on
remain the user's responsibility.

What engiNIR guarantees
-----------------------

- **One pipeline.** Calibration and prediction convert and preprocess spectra
  with the same code; fitted MSC/EMSC states are stored with the model.
- **Validated by sample, not by pixel.** Pixels are the data points of the
  fit; regions of interest (ROIs) are the unit of cross-validation, stateful
  preprocessing and every reported metric, so no sample is split across
  folds (:doc:`validation`).
- **Wrong input is refused.** A scan that does not match the model's
  instrument format, number of bands, reference correction or signal unit is
  not predicted.
- **Every prediction states its applicability.** Each pixel is tested against
  the model's DModXPS and T²PS limits (0.99) and gets a 95 % prediction
  interval; outside the model it gets no interval (:doc:`model_plots`).
- **Records are traceable and never overwritten.** One project file links
  data, references, ROIs, recipes and models by stable ids, with the engiNIR
  version and time that wrote them. A refit is a new model; the original is
  never changed. Raw ROI spectra are stored as recorded
  (:doc:`../hdf5_format`).

Guidelines and evidence
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Expectation
     - Guideline
     - engiNIR
   * - Documented model development: preprocessing, variable selection,
       number of components
     - ICH Q14 (multivariate procedures), USP <1039>
     - Recipe, bands, fit settings and algorithm stored with each model;
       components chosen by a stated rule (one-SE, Q² or F-test)
       (:doc:`fit_specification`, :doc:`band_selection`)
   * - Performance characteristics
     - ICH Q2(R2), USP <1119>, FDA guidance on NIR analytical procedures
     - RMSEE, RMSECV, bias, R², Q² and RPD per ROI; RMSEP and bias on
       independent prediction ROIs with reference values
   * - Detection of samples outside the calibration
     - USP <1039>, FDA PAT guidance
     - Hotelling's T² and DModX with limits for the calibration; DModXPS and
       T²PS for every predicted pixel
   * - Model lifecycle and change management
     - ICH Q12, ICH Q14
     - Immutable models with stable ids; a refit is a new model linked to its
       calibration set and input contract
   * - Data integrity
     - FDA data integrity guidance (ALCOA+)
     - Raw data as recorded, stable ids, writer version and timestamps, and a
       log file of the work done (:doc:`logging`)

Not covered
-----------

- 21 CFR Part 11 controls: user accounts, electronic signatures and a
  per-user audit trail.
- Computerized system validation of your installation (for example under
  GAMP 5).
- A validation report. Predictions are logged but not stored in the project
  file.
