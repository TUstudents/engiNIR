Validation & Figures of Merit
==============================

Metrics
-------

enginir provides the standard NIR chemometrics figures of merit.
All metric functions accept ``(y_true, y_pred)`` numpy arrays.
Inputs must have matching non-empty shapes and finite numeric values.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Function
     - Formula
     - Notes
   * - ``rmsec``
     - :math:`\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}`
     - Applied to calibration-set predictions
   * - ``rmsecv``
     - :math:`\sqrt{\frac{1}{n}\sum(y - \hat{y}_{cv})^2}`
     - Applied to cross-validated predictions
   * - ``rmsep``
     - :math:`\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}`
     - Applied to an independent test set
   * - ``bias``
     - :math:`\frac{1}{n}\sum(\hat{y} - y)`
     - Mean prediction error; positive when the model over-predicts
   * - ``sep``
     - :math:`\sqrt{\frac{1}{n-1}\sum(\hat{y} - y - \text{bias})^2}`
     - Standard Error of Prediction: random error only
   * - ``r2_score``
     - :math:`1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}`
     - Coefficient of determination
   * - ``rpd``
     - :math:`\frac{\text{SD}(y)}{\text{RMSEP}}`
     - Ratio of Performance to Deviation
   * - ``rpiq``
     - :math:`\frac{\text{IQR}(y)}{\text{RMSEP}}`
     - Ratio of Performance to Inter-Quartile range; more robust than RPD

The RMSEP-SEP-bias identity:

.. math::

   \text{RMSEP}^2 = \text{bias}^2 + \frac{n-1}{n}\,\text{SEP}^2

SEP measures random scatter in the predictions; bias measures systematic
offset. A large bias relative to RMSEP indicates a model that consistently
over- or under-predicts.

RPD and RPIQ use RMSEP in the denominator. Some authors use SEP instead, which
gives the same value for unbiased predictions and large :math:`n`. Published
RPD thresholds differ between fields and between these two definitions, so
compare RPD values only with references that use the same definition.

.. code-block:: python

   from enginir.validation.metrics import bias, rmsec, rmsecv, rmsep, sep, r2_score, rpd, rpiq

   print(f"RMSEC  = {rmsec(y_cal, y_pred_cal):.4f}")
   print(f"RMSECV = {rmsecv(y_cal, y_pred_cv):.4f}")
   print(f"RMSEP  = {rmsep(y_test, y_pred_test):.4f}")
   print(f"SEP    = {sep(y_test, y_pred_test):.4f}")
   print(f"bias   = {bias(y_test, y_pred_test):.4f}")
   print(f"R2     = {r2_score(y_test, y_pred_test):.4f}")
   print(f"RPD    = {rpd(y_test, y_pred_test):.2f}")

metrics_report
~~~~~~~~~~~~~~

:func:`~enginir.validation.metrics_report` computes all figures of merit at once.
Pass calibration predictions for RMSEC/R2_cal, cross-validated predictions for
RMSECV/R2_cv, and independent test set predictions for RMSEP/SEP/bias/RPD/RPIQ:

.. code-block:: python

   from enginir.validation import metrics_report

   # Calibration + CV only
   report = metrics_report(y_cal, y_pred_cal, y_pred_cv=y_pred_cv)
   # dict keys: RMSEC, R2_cal, RMSECV, R2_cv

   # Full report with independent test set
   report = metrics_report(y_cal, y_pred_cal, y_pred_cv=y_pred_cv,
                           y_true_test=y_test, y_pred_test=y_pred_test)
   # dict keys: RMSEC, R2_cal, RMSECV, R2_cv, RMSEP, bias, SEP, R2, RPD, RPIQ

   # as a pandas DataFrame (one row per call)
   df = metrics_report(y_cal, y_pred_cal, as_dataframe=True)

Cross-validation
----------------

:func:`~enginir.validation.cross_validate` wraps scikit-learn's ``KFold`` /
``StratifiedKFold`` (classifiers) / ``GroupKFold`` / ``LeaveOneOut`` /
``LeaveOneGroupOut`` and computes chemometric metrics on the out-of-fold
predictions. The default ``cv="auto"`` uses shuffled 7-fold splitting for
ungrouped data (seeded by ``random_state=0``) and grouped splitting when
``groups`` is given. ``results["cv_scheme"]`` records the splitter that was
used, e.g. ``{"type": "KFold", "n_splits": 7}``.

.. code-block:: python

   from enginir.validation import cross_validate

   results = cross_validate(model, X, y)
   # results["y_pred"]   - concatenated out-of-fold predictions
   # results["rmsecv"]   - RMSECV over all folds
   # results["r2"]       - R2
   # results["rpd"]      - RPD

   # explicit observation-level leave-one-out
   results_loo = cross_validate(model, X, y, cv="loo")

   # classification with string labels (SIMCA, LDA)
   from enginir.models import SIMCA
   results_cls = cross_validate(SIMCA(n_components=5), X, y_labels, cv=5)
   # results_cls["y_pred"] - predicted labels
   # regression metrics (rmsecv, r2, ...) are omitted for classifiers

Grouped cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~

When X contains multiple spectra (pixels) from the same physical sample,
as in ROI-based data, standard k-fold allows pixels from the same
sample to appear in both train and test. This leaks ROI identity into the
validation fold and usually makes RMSECV look too optimistic.

Pass a ``groups`` array (one group label per spectrum, e.g. the ROI id) to use grouped
validation by default. All pixels belonging to the same sample are always kept
in the same fold:

.. code-block:: python

   import numpy as np

   # groups[i] is the sample/ROI identity for spectrum i
   groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, ...])

   # Auto grouped CV: 7-fold GroupKFold when >=7 groups are available,
   # otherwise LeaveOneGroupOut.
   results = cross_validate(model, X, y, groups=groups)

   # Explicit 7-fold grouped CV.
   results_7 = cross_validate(model, X, y, cv=7, groups=groups)

   # leave-one-group-out (most conservative; one ROI withheld per fold)
   results_logo = cross_validate(model, X, y, cv="logo", groups=groups)

Explicit integer ``cv`` uses ``GroupKFold(n_splits=cv)`` and therefore requires
at least that many unique groups. ``GroupKFold`` assigns groups to folds by size
and, among groups of equal size, by the order of their labels, so use labels
whose order is reproducible (for example sample numbers, not random ids); model
runs label each ROI with its position in the sample set. ``cv="loo"`` remains available for explicit
observation-level leave-one-out, but it ignores groups, logs that choice when
groups are present, and is not the default.

The metrics in ``results`` are computed over pixels. For ROI-based data the
primary figures of merit are ROI-level: average the out-of-fold predictions per
group first, then compute the metrics on one value per ROI. This is how model
runs report RMSECV, R² and bias.

.. code-block:: python

   from enginir.validation import aggregate_regression_predictions, rmsecv

   roi_ids, y_roi, y_pred_roi = aggregate_regression_predictions(
       results["y_true"], results["y_pred"], groups
   )
   print(f"ROI-level RMSECV = {rmsecv(y_roi, y_pred_roi):.4f}")

Kennard-Stone Split
-------------------

Kennard-Stone selects a representative calibration subset that spans the
full range of spectral variation, maximising coverage of the spectral space.

.. code-block:: python

   from enginir.validation import kennard_stone

   cal_idx, test_idx = kennard_stone(X, n_select=40)
   X_cal, y_cal   = X[cal_idx], y[cal_idx]
   X_test, y_test = X[test_idx], y[test_idx]

   # Mahalanobis distance (accounts for covariance structure); pass PCA scores,
   # because raw spectra with more wavelengths than samples are rejected
   cal_idx, test_idx = kennard_stone(scores, n_select=40, metric="mahalanobis")

Distances are computed one row at a time, never as the full
:math:`n \times n` matrix, so memory grows linearly with the number of
samples; run time still grows quadratically, because the first step searches
for the two most distant samples.

Outlier Detection
-----------------

Leverage outliers (hat matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identifies samples with high influence on the regression model. The leverage
:math:`h_i = 1/n + t_i^\top (T^\top T)^{-1} t_i` is computed from the
mean-centred input; the default threshold is :math:`3(r+1)/n`, three times the
mean leverage, where :math:`r` is the rank of the centred input.

Pass model scores, not raw spectra: with more wavelengths than samples every
spectrum has leverage 1 and the function warns.

.. code-block:: python

   from enginir.validation.outliers import leverage_outliers

   lev, mask = leverage_outliers(model.transform(X))
   print(f"{mask.sum()} leverage outliers detected")

Hotelling T2 (score space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects outliers in the latent variable space using the Hotelling T2
statistic. For the calibration samples themselves the critical value comes
from the Beta distribution; for new samples, pass the calibration scores and
the F-distribution limit is used (Tracy et al., 1992).

.. code-block:: python

   from enginir.validation.outliers import hotelling_t2

   T = model.transform(X)   # PLS or PCA scores
   t2, t2_crit, mask = hotelling_t2(T, alpha=0.05)

   # new samples, judged against the calibration scores
   t2_new, t2_crit_new, mask_new = hotelling_t2(model.transform(X_new),
                                                calibration_scores=T)

Q-residuals (spectral outliers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects samples that are poorly reconstructed by the model: spectral
features not captured by the latent variables. The residual
:math:`Q_i = \sum_k (x_{ik} - \hat{x}_{ik})^2` is compared with a critical
value from a scaled :math:`\chi^2` distribution whose mean and variance match
those of the observed Q values (Nomikos & MacGregor, 1995). float32 spectra are
accepted directly; the residuals are accumulated in float64.

.. code-block:: python

   from enginir.validation.outliers import spectral_outliers

   # PLSRegression scales X, so undo the scaling and centring
   X_recon = (model.transform(X) @ model.x_loadings_.T) * model.x_std_ + model.x_mean_
   q, q_crit, mask = spectral_outliers(X, X_recon, alpha=0.05)
