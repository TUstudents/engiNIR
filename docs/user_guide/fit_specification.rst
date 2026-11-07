Fit specification
=================

A model fitted in a project records exactly how it was fitted: how the
validation folds were formed, which pixels were used, how X and Y were
scaled and which pixels were left out. ``model.refit()`` repeats it, and the
numbers can be compared with Sartorius SIMCA 18.

Everything is set with ``fit_spec`` (a mapping, or the typed
:class:`~enginir.project.FitSpec`); keys left out keep their defaults. The
scaling is a model setting and is passed like ``n_components``:

.. code-block:: python

   model = project.fit(
       "API content", recipe=PreprocessingRecipe(snv=True), n_components=2,
       x_scaling="center", y_scaling="uv",
       fit_spec={
           "folds": {"rule": "round_robin", "n_folds": 7, "group_by": "sample"},
           "pixels": {"per_roi": 40, "seed": 1},
           "exclusion": {"method": "dmodx", "level": 0.99},
       })
   model.ref.fit_specification      # the complete FitSpec, as stored

The viewer's fit dialog sets the same options (**Hold out together** and
the **Fitting** group), and the Project browser's inspector shows them for a
stored model.

Validation folds
----------------

Pixels are observations, but targets and metrics belong to ROIs: the pixels
of one ROI are never split between training and validation, and the
reported metrics are per ROI (held-out pixel predictions averaged per ROI).

By default the ROIs are dealt **round-robin** to 7 folds: the ROI at position
*i* of the sample set goes to fold *i* mod 7, and with fewer than 7 ROIs
every fold holds out one ROI. ``cv=5`` gives five folds, ``cv="logo"``
leaves one ROI out at a time (``"rule": "leave_one_group_out"`` in
``fit_spec``).

**Several ROIs from one sample.** When ROIs come from the same physical
sample, for example three regions of one tablet, holding out one while the
others stay in training leaks that sample into its own validation and
RMSECV comes out too low. Give their ROIs the same sample label (the
**Sample** column of the ROI table, or ``project.set_sample``) and hold them
out together:

.. code-block:: python

   for roi in ("T1 a", "T1 b", "T1 c"):
       project.set_sample(roi, "tablet 1")
   model = project.fit("API content", n_components=2,
                       fit_spec={"folds": {"rule": "leave_one_group_out",
                                           "group_by": "sample"}})

``"group_by": "target"`` holds out the ROIs with the same reference value
together, without labels; with round-robin the levels, sorted by value, are
dealt like venetian blinds, so every fold spans the range. The groups a fit
used are stored with it: a refit repeats them even after labels are edited,
and warns when the current labels would group differently.

Pixel sample
------------

Larger ROIs contribute more pixels and so weigh more. ``"pixels":
{"per_roi": n, "seed": s}`` draws *n* pixels from every ROI (at most the
smallest ROI's pixel count), so every ROI weighs the same. The draw depends
only on the seed and each ROI's stored spectra, not on ids, ROI order or the
software version; adding an ROI does not change the others' pixels. Every
fold, the final model and the reported metrics use the drawn pixels; a refit
draws again and reports a different sample.

Scaling
-------

New models centre the spectra and scale the reference values to unit
variance (``x_scaling="center"``, ``y_scaling="uv"``), as usual for spectra
and in SIMCA. ``"uv"`` divides by the standard deviation (N − 1),
``"pareto"`` by its square root. Centring and scaling are learnt inside
every fold. For one target, the Y scaling does not change the predictions.
Models stored before engiNIR 1.2 keep their ``scale`` setting when refitted.

Excluding outlying pixels
-------------------------

DModX is the distance of a spectrum to the model: the size of the part the
model cannot describe, relative to the typical residual of the training
spectra (1 is typical). It finds spectra the calibration has never seen,
such as glare, shadows, background at a tablet's edge or a foreign material;
a pixel that is only more concentrated lies along the model and is not an
outlier in DModX. DCrit is the limit at a chosen level: at 0.99, about 1 %
of normal spectra lie above it.

``"exclusion": {"method": "dmodx"}`` fits the model, leaves out the pixels
above DCrit (largest distance first) and refits, in one or two
``rounds``. No ROI loses more than ``max_fraction`` (20 %) of its pixels,
and ROIs are never left out: an ROI with more outlying pixels than its share
is flagged. The exclusion runs inside every validation fold on that fold's
training pixels only, so the held-out ROI is predicted from all its pixels,
as a new scan is, and RMSECV is not flattered. ``model.excluded_pixels``
lists the excluded pixels per ROI.

``"formula": "dmodx-1"`` (default) treats all bands alike; ``"dmodx-2"``
weights each band's residual by its modelling power, SIMCA's default.
Weighting turns down the bands the model does not describe, which is where a
contaminant or glare often shows, so the unweighted form is the default for
excluding pixels. :func:`~enginir.validation.dmodx`,
:func:`~enginir.validation.dmodx_limit` and
:func:`~enginir.validation.modelling_power` compute both forms directly.

Choose the number of components first: real chemical variation the model
leaves out ends up in the residuals and would make valid pixels look
outlying.

Comparison with SIMCA 18
------------------------

A reference project (``tests/data/simca_reference``) was fitted in engiNIR
and in SIMCA 18.1. engiNIR converts raw counts to absorbance (SIMCA has no
such conversion); from the absorbance tables on, these agree to float32
precision:

- the PLS model: scores, weights, loadings, coefficients, fitted values,
  VIP, R²X, R²Y, RMSEE (N − 1 − A), and predictions of a test set (RMSEP)
- SNV and the Savitzky-Golay first derivative (SIMCA drops the 7 edge
  bands of a 15-point window, engiNIR extrapolates them)
- Hotelling's T²Range and T²Crit, DModX (unweighted and weighted), DModXPS,
  DCrit and the modelling power

Differences:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Topic
     - SIMCA 18
     - engiNIR
   * - Cross-validation
     - one component at a time on the full model's residuals; the held-out
       rows were part of the model that deflated them, so Q² is more
       optimistic
     - every fold refits the complete model on its training ROIs
   * - CV groups by a variable (e.g. concentration)
     - sorts the observations and cuts them into equal blocks, which can
       split a sample across folds
     - not offered; ``group_by="target"`` or ``"sample"`` never splits a
       sample
   * - CV groups by an ID
     - one group per value
     - ``leave_one_group_out`` with ``group_by="roi"`` or ``"sample"``
   * - Default CV groups
     - every Nth observation
     - round-robin over ROIs: the same partition on ROI means
   * - Scaling ``none``
     - offered
     - every scaling centres
   * - AsLS baseline correction
     - offered (Eilers' algorithm; the reference exports match it with
       λ 10000, p 0.001, 10 iterations)
     - not offered yet
