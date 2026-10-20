Chemometric Models
==================

All models follow the scikit-learn estimator API and are usable inside
``sklearn.pipeline.Pipeline``.

Regression
----------

PLSRegression (PLS1 / PLS2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

- ``x_mean_`` and ``x_std_`` for the training X centering/scaling
- ``y_mean_`` and ``y_std_`` for the training Y centering/scaling
- ``coef_`` and ``intercept_`` in raw data units, so that
  ``y = X @ coef_ + intercept_`` (the same contract as :class:`PCR`)

``coef_`` and ``intercept_`` are the values stored in project model runs
(``model_runs/{id}/artifact_refs/prediction_model``) and replayed for
prediction.

``predict`` and ``transform`` work in the floating dtype of ``X``: float32
spectra give float32 predictions and scores without a float64 copy. Fitting
uses scikit-learn's NIPALS implementation, which works in float64 internally.

PCR (Principal Component Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PCA followed by ordinary least squares on the retained scores.

.. code-block:: python

   from enginir.models import PCR

   model = PCR(n_components=10)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

Machine Learning Models
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

SIMCA (Soft Independent Modelling of Class Analogy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

LDA Classifier
~~~~~~~~~~~~~~~

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
     - Class modelling; detection of samples outside all classes
   * - LDAClassifier
     - Classification
     - Multi-class discrimination; linear decision boundaries

Models in a project
-------------------

In a project (:class:`enginir.Project`), every pixel spectrum of an ROI is an
observation with the ROI's reference value; cross-validation keeps the pixels
of one ROI together, and the reported metrics are per ROI, from the mean of
its held-out pixel predictions.

Project model runs are ``PLS`` or ``PCR``. :meth:`enginir.Project.fit` uses
every ROI with a value for the target:

.. code-block:: python

   from enginir import PreprocessingRecipe, Project

   project = Project.open("project.h5")
   model = project.fit("moisture", recipe=PreprocessingRecipe(snv=True),
                       model="PCR", n_components=5)
   print(model.metrics["rmsecv"])

The project stores the sample set, the cross-validation scheme, the metrics
and the fitted model. Training converts each ROI's raw spectra with its own
calibration before the recipe is applied (raw counts stay ``uint16`` until
then). A model therefore fixes three pieces of input provenance:

- the sample-set contract
- the signal-domain contract
- the preprocessing recipe

Prediction must satisfy that frozen contract before the estimator is used.
When the recipe enables MSC or EMSC, the fitted training-side transformer state
is stored with the prediction-model artifact and restored after project reopen.
