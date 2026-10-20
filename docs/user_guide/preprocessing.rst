Preprocessing
=============

All preprocessing classes follow the scikit-learn transformer API
(``fit`` / ``transform`` / ``fit_transform``) and can be composed in a
``sklearn.pipeline.Pipeline``.

For sample-wise transforms (SNV, SavitzkyGolay, Detrend, Smoother) the
``fit`` step is a no-op: the same transform is applied to any new data
without re-fitting.  MSC and EMSC store a reference spectrum during ``fit``
and use it during ``transform``.

Transformers expect finite two-dimensional spectral matrices with shape
``(n_samples, n_wavelengths)``. Invalid shapes, non-numeric values, and
``NaN`` or infinite values are rejected before numerical routines run.

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from enginir.preprocessing import SNV, SavitzkyGolay, MSC, EMSC, Detrend, Smoother

Scatter Correction
------------------

Standard Normal Variate (SNV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SNV centres and scales each spectrum independently:

.. math::

   x_\text{SNV} = \frac{x - \bar{x}}{\sigma_x}

No fitting required.

.. code-block:: python

   X_snv = SNV().fit_transform(X)

Multiplicative Scatter Correction (MSC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MSC corrects each spectrum relative to a reference (mean) spectrum
by fitting :math:`x = a + b \cdot r` and returning
:math:`(x - a) / b`:

.. code-block:: python

   msc = MSC()
   X_msc = msc.fit_transform(X_train)   # stores mean spectrum as reference_
   X_test_msc = msc.transform(X_test)   # uses same reference

Pass a custom reference spectrum::

   msc = MSC(reference=my_reference)

Extended Multiplicative Scatter Correction (EMSC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EMSC extends MSC by including polynomial baseline terms, optional known
interference spectra, and optional wavelength weights in the correction model:

.. math::

   x = b \cdot r + a_0 + a_1 u + a_2 u^2 + \sum_k c_k \cdot z_k + \epsilon

where :math:`u` is the band index rescaled to :math:`[-1, 1]` (EMSC does not use
the wavelength values, so the baseline is polynomial in band position). The
corrected spectrum removes the baseline and interference contribution, then
normalizes by the multiplicative reference coefficient :math:`b`. The design
matrix must have full rank: a reference that is itself a polynomial of degree
``<= order``, or interferences collinear with the other terms, are rejected.

The implementation follows the scikit-learn transformer contract: ``fit``
learns the reference spectrum and design matrix from training spectra, and
``transform`` reuses that fitted state for later spectra. Inputs are validated
before fitting so malformed correction settings fail early:

- ``method`` must be ``"mean"`` or ``"median"``. It controls reference
  estimation only when no custom ``reference`` is supplied.
- ``order`` must be an integer greater than or equal to zero.
- ``reference`` and each ``interferences`` row must match the fitted wavelength
  count.
- ``weights`` must be finite, non-negative, and contain at least one positive
  value.

.. code-block:: python

   emsc = EMSC(method="mean", order=2)
   X_emsc = emsc.fit_transform(X)

   # with interference spectra (e.g. water, protein background)
   emsc = EMSC(order=2, interferences=water_spectra)
   X_corrected = emsc.fit_transform(X)

   # with a custom reference and wavelength weights
   emsc = EMSC(reference=reference_spectrum, weights=wavelength_weights)
   X_weighted = emsc.fit_transform(X)

For model validation and prediction, fit EMSC only on the training fold or
calibration set. The easiest safe pattern is to put it inside a
``sklearn.pipeline.Pipeline`` so every fold receives its own fitted EMSC state.
Saved model runs persist the fitted MSC/EMSC training state required for
reopened prediction. Prediction restores that state and never estimates a new
reference from prediction spectra.

Derivatives
-----------

Savitzky-Golay
~~~~~~~~~~~~~~

Applies a Savitzky-Golay polynomial filter row-wise, computing the
0th (smoothing), 1st, or 2nd derivative:

.. code-block:: python

   # 1st derivative, window of 11 points, 2nd-order polynomial
   sg = SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
   X_d1 = sg.fit_transform(X)

   # smoothing only (deriv=0)
   X_smooth = SavitzkyGolay(window_length=15, polyorder=3, deriv=0).fit_transform(X)

Savitzky-Golay assumes evenly spaced bands. Derivatives are per band by
default; pass the band spacing as ``delta`` (e.g. in nm) to express them per
unit wavelength.

Baseline Correction
-------------------

Detrend
~~~~~~~

Fits and subtracts a polynomial baseline from each spectrum:

.. code-block:: python

   det = Detrend(order=1)           # linear baseline
   X_det = det.fit_transform(X)

   det2 = Detrend(order=2, wavelengths=wl)   # quadratic, using real wl axis

Smoother
~~~~~~~~

Two smoothing methods are available:

**Moving average** (default)::

   sm = Smoother(method="moving_average", window_length=5)
   X_sm = sm.fit_transform(X)

**Whittaker smoother**: penalised least squares, produces smooth spectra
without edge artefacts.  Larger ``lam`` gives stronger smoothing::

   sm = Smoother(method="whittaker", lam=1e4, order=2)
   X_sm = sm.fit_transform(X)

PreprocessingRecipe
-------------------

:class:`~enginir.preprocessing.PreprocessingRecipe` captures the full
preprocessing configuration as a serializable value object. In a project it
is persisted under ``/recipes/{recipe_id}`` and referenced by sample sets and
prediction runs, so model fitting and prediction replay exactly the same
steps; ROI provenance only keeps a human-readable summary
(:meth:`~enginir.preprocessing.PreprocessingRecipe.describe`). It can also be
used standalone to save and restore pipeline settings:

.. code-block:: python

   from enginir.preprocessing import PreprocessingRecipe

   # Create a recipe
   recipe = PreprocessingRecipe(
       filter="sg_d1", sg_window=11, sg_polyorder=2,
       snv=True,
   )

   # Serialise to a plain dict (suitable for JSON / HDF5 storage)
   d = recipe.to_dict()
   # {"filter": "sg_d1", "sg_window": 11, "sg_polyorder": 2,
   #  "snv": True, "msc": False, "emsc": False,
   #  "detrend": False, "detrend_order": 1}

   # Round-trip from dict (unknown keys are logged and discarded)
   recipe2 = PreprocessingRecipe.from_dict(d)
   assert recipe == recipe2

The viewer exposes ``ViewerState.recipe`` (read current settings) and
``ViewerState.apply_recipe(recipe)`` (restore settings atomically).

Saved recipes validate their numerical settings: ``sg_window`` must be an odd
integer greater than or equal to 3, ``sg_polyorder`` must be at least 1 and
less than ``sg_window``, and ``detrend_order`` must be non-negative.
``build_steps`` rejects a ``sg_window`` longer than the spectrum instead of
shrinking it, so the filter that runs is always the one the recipe states.
Persisted project recipes contain all eight fields shown above. Missing fields
are rejected rather than reconstructed from Python defaults; unknown fields
are logged and discarded. The recipe id is supplied by its
``/recipes/{recipe_id}`` project path.

Pipeline example
----------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from enginir.preprocessing import EMSC, SavitzkyGolay
   from enginir.models import PLSRegression

   pipe = Pipeline([
       ("emsc", EMSC(method="mean", order=2)),
       ("sg",  SavitzkyGolay(window_length=11, polyorder=2, deriv=1)),
       ("pls", PLSRegression(n_components=6)),
   ])

   pipe.fit(X_train, y_train)
   y_pred = pipe.predict(X_test)
