Reading the Model plots
=======================

Every plot of the viewer's Model workspace has a **?** button in its header that
shows how to read it. The same texts are collected here, page by page.

Two levels appear throughout. **Per ROI** numbers are computed over ROI means: the
target, the cross-validation folds and the reported model quality are per ROI.
**Per pixel** numbers are computed over single spectra, the pixels the model
is fitted on: R²X, R²Y over pixels, DModX and T² with their limits. Single pixels
are noisy, so pixel numbers are worse than ROI numbers; a plot names its level in
its title or legend.

Fit page
--------

**Predicted vs target value, per ROI**

Each point is one ROI: its target value against the mean of its pixels'
cross-validated predictions, so the statistics are per ROI. Q² (the
cross-validated R²), RMSECV (per ROI) and bias come from these points; RMSEE
(per ROI) and R²adj from the fit. Points should lie along the dashed line. A
tilt (low values over-, high values under-predicted) means the model is pulled
to the mean; points in groups by dataset mean the model may separate scans
rather than the target.

**Residuals, per ROI**

Per ROI: the cross-validated prediction minus the target value, against the
target value. Residuals should scatter around zero with no pattern. A slope
means the model is pulled to the mean; a curve means a non-linear relation; one
level far off points to a problem with those samples or their target values.

**Target value distribution, per ROI**

Per ROI: how many ROIs have each target value (one bar per value when there are
at most 12 distinct values, else bins). A model predicts best inside the range
it was calibrated on and where values are dense; few distinct levels limit how
many components the data can support.

**Calibration scores (LV1 vs LV2), per ROI**

Each point is one ROI: the mean of its pixels' scores on the first two latent
variables, colored by target value. The gray ellipses are Hotelling's T² limits
for single calibration pixels; ROI means sit well inside them. The colored
ellipses are computed from the ROI means: an ROI outside them is unusual among
the ROIs. A smooth color gradient means LV1 and LV2 follow the target.

Spectral page
-------------

**Regression coefficients**

The regression coefficients over the model's bands: how much each band adds to
the prediction, in the units of the preprocessed spectra. Broad features at
known absorption bands are a good sign; many sharp swings from band to band, or
large values at the detector ends, suggest the model fits noise.

**Scaled coefficients (CoeffCS)**

The coefficients in the model's scaled units (SIMCA's CoeffCS): each
coefficient times its band's scaling, divided by the target's. Comparable
between bands, so their size shows importance when bands are scaled
differently.

**Latent loadings (LV1-LV3)**

The loadings of the first three latent variables over the bands: which spectral
shape each component describes. A loading that looks like an absorption band is
chemistry; a smooth slope or offset is usually scatter or baseline; a rough,
noisy loading is a component that models noise.

**VIP**

VIP (PLS): the weight of each band in the components, weighted by how much of
the target each component explains; the mean of VIP² is 1. Bands above 1 matter
more than average, below 0.8 little. The model's mean spectrum is shaded
behind. High VIP at the detector ends or in a saturated band is a warning, not
chemistry.

**VIP, sorted**

VIP of every band, sorted from the largest, with the five largest listed. Bars
above 1 (colored) are the bands the model relies on most; the share of bands
above 1 is in the VIP plot.

**Selectivity ratio**

Selectivity ratio (PLS and PCR): per band, the variance explained by the
model's prediction direction divided by the variance it leaves. Independent of
how one algorithm spreads weights over components. Values well below 1
everywhere mean most spectral variation is unrelated to the target.

Components page
---------------

**RMSECV (per ROI) and RMSEE (per ROI) vs components**

Per ROI, against the number of components: RMSECV (per ROI) with its standard
error (cross-validated) and RMSEE (per ROI) (fit, with R − A − 1 degrees of
freedom). The vertical line is the model's number; the list gives each rule's
choice and the number that decided it, and what the noise guard changed. RMSECV
(per ROI) that keeps creeping down after a clear knee usually means optimistic
folds, not more chemistry.

**R²Y and Q² vs components, per ROI**

Per ROI: R²Y(cum), the share of the target the fit explains, and Q²(cum), the
share it predicts in cross-validation, both over ROI means. The shaded gap is
what the fit gains over prediction; a gap that widens with more components is
over-fitting.

**Each component: Q²ₐ, ΔR²X, roughness**

Each component on its own: Q²ₐ (per ROI, how much it improves the prediction;
the dashed line is the limit L of SIMCA's rule R1), ΔR²X (per pixel, the share
of the spectra it explains) and the Durbin–Watson value of its weights (orange,
right axis: 0 smooth, 2 noise). Gray components explain under 1 % of the
spectra and do not improve the prediction: noise, which the noise guard drops.

**R²X and R²Y vs components, per pixel**

Per pixel, against the number of components: R²X(cum), the share of the spectra
explained, and R²Y(cum), the share of the target explained over single pixels.
Single pixels are noisy, so R²Y per pixel stays below the ROI value; R²X
levelling off shows where the spectra have no more structure.

Distance page
-------------

**Distance to the model (DModX), per pixel**

Per pixel: the distance of each calibration pixel to the model (normalized
DModX), with its 0.99 limit. About 1 % of the pixels should lie above it; many
more mean the pixels are not one homogeneous population. Bins go up to the
99.5th percentile; the gray bar holds the rest.

**Hotelling's T², per pixel**

Per pixel: Hotelling's T² of each calibration pixel in the model's components,
with its 0.99 limit. About 1 % should lie above it; a long tail means extreme
pixels such as glints or edges. Bins go up to the 99.5th percentile; the gray
bar holds the rest.

**Pixels above the limits, per ROI**

Per ROI: the share of its calibration pixels above the DModX limit (bars,
colored by target value) and above the T² limit (markers), sorted from the
largest. The dashed line is the 1 % a 0.99 limit expects. ROIs far above it
have spectra the model does not describe: another substrate, saturation, drift
along a scan. Click a bar to select the ROI.
