Band Importance and Band Selection
==================================

Which bands does a model rely on, and should some bands be left out? engiNIR
answers the first question with two measures stored with every PLS and PCR
model, and the second with band ranges in the recipe, chosen by hand or
proposed by a search whose own effect is validated.

Reading VIP and the selectivity ratio
-------------------------------------

Every model fitted with engiNIR 1.4 or later stores, on the model's bands:

- **VIP** (variable importance in projection; PLS only). VIP of a band is
  the weight it has in the components, weighted by the Y variance each
  component explains. The mean of VIP² over the bands is 1, so VIP > 1 marks
  bands of above-average importance; bands below about 0.8 contribute
  little. PLS runs from before 1.4 get VIP in the viewer from their stored
  weights.
- **Selectivity ratio** (Rajalahti et al.; PLS and PCR). The model is
  turned into one predictive component along its regression vector (target
  projection); for each band the ratio of the variance this component
  explains to the variance it leaves. A large ratio means the band varies
  with what the model predicts; it does not depend on how one algorithm
  spreads weights over components, so it compares PLS and PCR.
- The **mean spectrum** of the fitted pixels, which the viewer shades behind
  both.

.. code-block:: python

   model = project.fit("API", recipe=PreprocessingRecipe(snv=True), n_components=3)
   vip = model.diagnostics["vip"]
   ratio = model.diagnostics["selectivity_ratio"]
   wavelengths = model.diagnostics["wavelengths"]

The viewer shows them on the Model workspace's **Spectral** page: VIP over
wavelength with lines at 1.0 and 0.8, VIP sorted from the largest value (the
top bands labelled), and the selectivity ratio.

What changes them:

- **The number of components.** More components spread importance over more
  bands; read VIP for the model you use.
- **Preprocessing.** A derivative moves importance from the centre of an
  absorption band to its flanks, where the derivative is largest; SNV and
  MSC change which variation is left to explain.
- **X scaling.** Unit-variance scaling gives noisy, low-variance bands the
  same weight as strong ones and changes VIP; the selectivity ratio changes
  only through the fitted regression vector.

Neither is a significance test: a band with VIP 0.7 can still help, and
importance does not prove a chemical cause. Use them to look for spectral
regions that make sense, and to decide which bands to try leaving out.

Leaving bands out
-----------------

A recipe can keep or exclude band ranges; the selection is the **last
step**, after AsLS, Savitzky-Golay, SNV, MSC, EMSC and Detrend. Normalization
and MSC/EMSC references therefore see the whole spectrum, and only the model
sees the selected bands:

.. code-block:: python

   from enginir import PreprocessingRecipe

   # Leave out the two main NIR water bands.
   recipe = PreprocessingRecipe(filter="sg_d1", sg_window=11, snv=True,
                                bands_exclude=[(1400, 1500), (1900, 2000)])
   recipe.output_wavelengths(wavelengths)   # the bands the model sees
   model = project.fit("API", recipe=recipe, n_components="auto")

   # Keep two regions, minus a narrow part of the first.
   recipe = PreprocessingRecipe(snv=True, bands_keep=[(1100, 1350), (1500, 2300)],
                                bands_exclude=[(1200, 1210)])

- Ranges are inclusive, in nm on the axis after Savitzky-Golay (band numbers
  of the data file when it has no wavelengths). A band is kept when it lies
  in a keep range (or none is given) and in no exclude range.
- Ranges are normalized: each pair ordered, sorted, overlapping or touching
  ranges merged, so the same selection always gives the same recipe id.
- A selection that keeps no band is refused; a range that selects nothing is
  logged. A fit refuses more components than selected bands.
- The scans to predict keep all their bands; prediction applies the same
  selection. Plots over wavelength leave a gap where bands are left out.

Common reasons to leave bands out: noisy detector ends, saturated or very
strong water bands (around 1450 and 1940 nm in the NIR; offered in the
viewer as suggestions, never applied automatically), or regions that carry
an interference which differs between calibration and use.

In the viewer, the fit dialog's **Bands** row has **Keep** and **Exclude**
fields (``1100-1350, 1500-2300``), a **Water bands** menu, **Select on
plot...** (drag green keep and red exclude regions on the mean spectrum of
the target's ROIs, with the selected model's VIP or coefficients below as a
guide) and **Propose...** (below). **Fit again** keeps a model's ranges, and
the Explore preview shows unselected bands as missing.

Proposing bands, and why the proposal needs nested validation
-------------------------------------------------------------

Choosing bands by their cross-validated error is itself fitting: try enough
subsets and one will look good by chance. The RMSECV of the chosen bands is
then optimistic. engiNIR therefore reports the error of the whole procedure
next to the error of its result.

.. code-block:: python

   proposal = project.propose_bands("API", recipe=PreprocessingRecipe(snv=True),
                                    n_intervals=20)
   proposal.ranges            # e.g. ((1100.0, 1350.0), (2050.0, 2300.0))
   proposal.steps[-1].rmsecv  # RMSECV of the proposed ranges (optimistic)
   proposal.nested_rmsecv     # RMSECV of the procedure, selection included
   proposal.all_bands_rmsecv  # RMSECV with every band

   model = project.fit("API", recipe=PreprocessingRecipe(snv=True,
                                                         bands_keep=proposal.ranges),
                       n_components=proposal.n_components)

How the search works (forward interval PLS):

- The axis after Savitzky-Golay is split into ``n_intervals`` contiguous
  intervals of (nearly) equal band counts.
- Step 0 predicts the mean of the reference values. Each step adds the
  interval whose addition gives the lowest RMSECV, with the number of
  components chosen by the component rule (``one_se`` by default). The
  search stops when the best addition does not beat the current RMSECV by
  more than its standard error, the same idea as the one-SE rule for
  components.
- Every RMSECV uses the folds and fold groups of the fit specification, on
  ROI means, with MSC/EMSC fitted inside the folds; preprocessing always
  runs on the whole spectrum.
- **Nested validation.** For every outer fold the whole search runs again on
  the training ROIs only; the bands it chooses are fitted on them and
  predict the held-out ROIs. ``nested_rmsecv`` is the RMSECV of those
  predictions.

Reading the result: if ``nested_rmsecv`` is clearly below
``all_bands_rmsecv``, leaving bands out helps for data like these. If it is
about the same or above, the gain shown by the selected ranges is mostly
chance; keep all bands, or cut only bands you can justify (detector ends,
water). The proposal is not stored; a fit with its ranges is an ordinary
fit, and refitting it never searches again. The search costs about
``n_intervals`` × steps × (outer folds + 1) × inner folds fits; the viewer
shows its progress and can cancel it.

Not provided: genetic algorithms, CARS and other stochastic searches, and
automatic elimination by VIP (use the VIP plot to decide by hand).
