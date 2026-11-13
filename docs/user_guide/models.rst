Chemometric models
==================

All models follow the scikit-learn estimator API and are usable inside
``sklearn.pipeline.Pipeline``.

Regression
----------

PLSRegression (PLS1 / PLS2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Partial Least Squares regression.  PLS1 handles a single response variable;
PLS2 handles multiple responses simultaneously.

.. code-block:: python

   from enginir.models import PLSRegression

   model = PLSRegression(n_components=6)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

   # project into score space
   T = model.transform(X_test)

   # VIP scores: values > 1 indicate above-average importance
   vip = model.vip_scores()

`PLSRegression` also exposes public fitted normalization statistics on the
enginir wrapper:

- ``x_mean_`` and ``x_std_`` for the calibration X centering/scaling
- ``y_mean_`` and ``y_std_`` for the calibration Y centering/scaling
- ``coef_`` and ``intercept_`` in raw data units, so that
  ``y = X @ coef_ + intercept_`` (the same contract as :class:`PCR`)

``coef_`` and ``intercept_`` are the values stored in project models
(``model_runs/{id}/artifact_refs/prediction_model``) and replayed for
prediction.

``predict`` and ``transform`` work in the floating dtype of ``X``: float32
spectra give float32 predictions and scores without a float64 copy.

Fitting uses improved kernel PLS (``algorithm="kernel"``, the default): one
chunked pass over the rows builds XᵀX in float64, and every component then
works on that small band × band matrix, with no full copy of X.
``algorithm="nipals"`` runs scikit-learn's NIPALS instead; both give the same
model to floating-point precision. NIPALS is kept until engiNIR 2.0.

PCR (principal component regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PCA followed by ordinary least squares on the retained scores.

.. code-block:: python

   from enginir.models import PCR

   model = PCR(n_components=10)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

The principal components come from the eigendecomposition of XᵀX
(``algorithm="gram"``) or a full SVD of X (``"svd"``, kept until 2.0). The
default ``"auto"`` uses ``gram`` unless the last component is nearly
degenerate (σ_A / σ₁ < 10⁻⁶, e.g. close to the rank limit after SNV) and
tells which it used in ``algorithm_``.

Machine learning models
-----------------------

Thin wrappers around scikit-learn estimators with the same input validation
and logging as the chemometric models. They do not scale the spectra: put a
scaler or scatter correction in front of them in a ``Pipeline`` where the
learner needs it (SVR with an RBF kernel and MLPs usually do). The random
forest and the MLP also classify with ``task="classification"``.

Support Vector Regression::

   from enginir.models import SVRModel
   model = SVRModel(kernel="rbf", C=10, epsilon=0.01)

Random Forest::

   from enginir.models import RandomForestModel
   model = RandomForestModel(n_estimators=200)

Multi-Layer Perceptron::

   from enginir.models import MLPModel
   model = MLPModel(hidden_layer_sizes=(64, 32), max_iter=500)

   # classification: is_classifier(model) is True, predict_proba is available
   model = RandomForestModel(task="classification")

Classification
--------------

SIMCA (soft independent modeling of class analogy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SIMCA builds one PCA model per class. A sample is accepted by a class when
its Q residual (squared distance to the class PCA subspace) is below the
Jackson-Mudholkar limit at significance level ``alpha``, computed from the
eigenvalues of the discarded components. Among the accepting classes the one
with the smallest Q residual wins; a sample accepted by no class is reported
as not assigned. ``model.q_residuals(X)`` returns the Q residuals of every
sample for every class.

.. code-block:: python

   from enginir.models import SIMCA

   model = SIMCA(n_components=5, alpha=0.05)
   model.fit(X_train, y_train)          # y_train: string or integer labels
   y_pred = model.predict(X_test)       # "unknown" or -1 = not assigned to any class

Samples accepted by no class get a rejection label that fits the label dtype:
``-1`` for integer labels (unsigned labels are returned as ``int64``),
``"unknown"`` for string labels (including object arrays such as a pandas
Series), ``nan`` for float labels and ``None`` otherwise. Requested
``n_components`` are reduced, with a warning, for classes whose data have too
few independent samples to leave a residual subspace.

LDA classifier
~~~~~~~~~~~~~~

Linear Discriminant Analysis with optional PCA pre-reduction to handle
high-dimensional spectral data.

.. code-block:: python

   from enginir.models import LDAClassifier

   model = LDAClassifier(n_pca_components=20)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

Choosing a model
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Model
     - Type
     - When to use
   * - PLSRegression
     - Regression
     - Primary choice for NIR quantification; interpretable; handles collinear spectra
   * - PCR
     - Regression
     - Useful when Y is noisy; interpretable via PCA loadings
   * - SVRModel
     - Regression
     - Non-linear relationships; small-medium datasets
   * - RandomForestModel
     - Regression
     - Non-linear; robust to outliers; feature importance available
   * - MLPModel
     - Regression
     - Large datasets with complex non-linearities
   * - SIMCA
     - Classification
     - Class modeling; detection of samples outside all classes
   * - LDAClassifier
     - Classification
     - Multi-class discrimination; linear decision boundaries

Fitting speed
-------------

A fit in a project (:meth:`enginir.Project.fit`, or **Fit** in the viewer)
with a recipe without MSC or EMSC works on sums: one pass over the pixels
builds XᵀX per fold group, and every fold, every component count of the
component curve and the final model are fitted from those sums. Held-out
ROIs are predicted from their mean spectra. One more pass computes what is
per pixel (T², DModX, every pixel's held-out prediction). The band proposal
works on the same sums. The results equal the classical fit within
floating-point precision.

On a calibration set of 20 ROIs and 396 000 pixels × 176 bands (SNV, 7
folds, component curve to 15), a fit took 43 s and 1.7 GB of memory with
NIPALS and takes 4.5 s and 0.45 GB with the kernel engine. With MSC or EMSC
the recipe is fitted inside every fold, so the folds keep their own pass
over the pixels; they still use the kernel engine.

The model stores its algorithm (``hyperparameters/algorithm``); a refit and
**Fit Again** use it. Models from engiNIR 1.5 have none and were fitted by
NIPALS (PLS) or the full SVD (PCR), which is what a refit uses for them.
On machines that share their cores with other work, the engine's passes
use half the logical CPUs; set ``ENGINIR_BLAS_THREADS`` to change that.

Models in a project
-------------------

In a project (:class:`enginir.Project`), every pixel spectrum of an ROI is a data point of the fit with the ROI's
target value; cross-validation keeps the pixels
of one ROI together, and the reported metrics are per ROI, from the mean of
its held-out pixel predictions.

Project models are ``PLS`` or ``PCR``. :meth:`enginir.Project.fit` uses
every ROI with a value for the target:

.. code-block:: python

   from enginir import PreprocessingRecipe, Project

   project = Project.open("project.h5")
   model = project.fit("moisture", recipe=PreprocessingRecipe(snv=True),
                       model="PCR", n_components=5)
   print(model.metrics["rmsecv"])

The project stores the calibration set, the cross-validation scheme, the metrics
and the fitted model. Calibration converts each ROI's raw spectra with its own
reference correction before the recipe is applied (raw counts stay ``uint16`` until
then). A model therefore fixes three pieces of input provenance:

- the sample-set contract
- the input contract
- the preprocessing recipe

Prediction must satisfy that frozen contract before the estimator is used.
When the recipe enables MSC or EMSC, the fitted calibration-side transformer state
is stored with the prediction-model artifact and restored after project reopen.
