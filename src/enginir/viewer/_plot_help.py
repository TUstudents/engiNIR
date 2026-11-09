"""Help texts of the Model diagnostics plots (model-workspace-diagnostics contract, Plot help).

Each text says what is drawn, at which level (per ROI or per pixel), how to
read it and what a bad sign looks like. The header's **?** button shows it;
``docs/user_guide/model_plots.rst`` carries the same texts (a test keeps them
equal).
"""

from __future__ import annotations

#: Plot id → help text, in the order of the Model pages.
MODEL_PLOT_HELP: dict[str, str] = {
    "model.pred_vs_ref": (
        "Each point is one ROI: its reference value against the mean of its pixels' "
        "cross-validated predictions, so the statistics are per ROI. Q² (the cross-validated "
        "R²), RMSECV and bias come from these points; RMSEE and R²adj from the fit. Points "
        "should lie along the dashed line. A tilt (low values over-, high values "
        "under-predicted) means the model is pulled to the mean; points in groups by data "
        "file mean the model may separate scans rather than the target."),
    "model.residuals": (
        "Per ROI: the cross-validated prediction minus the reference value, against the "
        "reference. Residuals should scatter around zero with no pattern. A slope means "
        "the model is pulled to the mean; a curve means a non-linear relation; one level far "
        "off points to a problem with those samples or their reference values."),
    "model.target_distribution": (
        "Per ROI: how many ROIs have each reference value (one bar per value when there are "
        "at most 12 distinct values, else bins). A model predicts best inside the range it "
        "was trained on and where values are dense; few distinct levels limit how many "
        "components the data can support."),
    "model.scores": (
        "Each point is one ROI: the mean of its pixels' scores on the first two latent "
        "variables, coloured by reference value. The grey ellipses are Hotelling T² limits "
        "for single training pixels; ROI means sit well inside them. The coloured ellipses "
        "treat the ROI means as the observations: an ROI outside them is unusual among the "
        "ROIs. A smooth colour gradient means LV1 and LV2 follow the target."),
    "model.coefficients": (
        "The regression coefficients over the model's bands: how much each band adds to the "
        "prediction, in the units of the preprocessed spectra. Broad features at known "
        "absorption bands are a good sign; many sharp swings from band to band, or large "
        "values at the detector ends, suggest the model fits noise."),
    "model.coef_cs": (
        "The coefficients in the model's scaled units (SIMCA's CoeffCS): each coefficient "
        "times its band's scaling, divided by the target's. Comparable between bands, so "
        "their size shows importance when bands are scaled differently."),
    "model.loadings": (
        "The loadings of the first three latent variables over the bands: which spectral "
        "shape each component describes. A loading that looks like an absorption band is "
        "chemistry; a smooth slope or offset is usually scatter or baseline; a rough, noisy "
        "loading is a component that models noise."),
    "model.vip": (
        "VIP (PLS): the weight of each band in the components, weighted by how much of the "
        "target each component explains; the mean of VIP² is 1. Bands above 1 matter more "
        "than average, below 0.8 little. The model's mean spectrum is shaded behind. High "
        "VIP at the detector ends or in a saturated band is a warning, not chemistry."),
    "model.vip_sorted": (
        "VIP of every band, sorted from the largest, with the five largest listed. Bars "
        "above 1 (coloured) are the bands the model relies on most; the share of bands "
        "above 1 is in the VIP plot."),
    "model.selectivity_ratio": (
        "Selectivity ratio (PLS and PCR): per band, the variance explained by the model's "
        "prediction direction divided by the variance it leaves. Independent of how one "
        "algorithm spreads weights over components. Values well below 1 everywhere mean "
        "most spectral variation is unrelated to the target."),
    "model.components": (
        "Per ROI, against the number of components: RMSECV with its standard error "
        "(cross-validated) and RMSEE (fit, with R − A − 1 degrees of freedom). The vertical "
        "line is the model's number; the list gives each rule's choice and the number that "
        "decided it, and what the noise guard changed. RMSECV that keeps creeping down "
        "after a clear knee usually means optimistic folds, not more chemistry."),
    "model.r2_q2": (
        "Per ROI: R²Y(cum), the share of the target the fit explains, and Q²(cum), the "
        "share it predicts in cross-validation, both over ROI means. The shaded gap is what "
        "the fit gains over prediction; a gap that widens with more components is "
        "over-fitting."),
    "model.per_component": (
        "Each component on its own: Q²ₐ (per ROI, how much it improves the prediction; "
        "the dashed line is the limit L of SIMCA's rule R1), ΔR²X (per pixel, the share of "
        "the spectra it explains) and the Durbin–Watson value of its weights (orange, right "
        "axis: 0 smooth, 2 noise). Grey components explain under 1 % of the spectra and do "
        "not improve the prediction: noise, which the noise guard drops."),
    "model.r2_pixels": (
        "Per pixel, against the number of components: R²X(cum), the share of the spectra "
        "explained, and R²Y(cum), the share of the target explained over single pixels. "
        "Single pixels are noisy, so R²Y per pixel stays below the ROI value; R²X levelling "
        "off shows where the spectra have no more structure."),
    "model.dmodx": (
        "Per pixel: the distance of each training pixel to the model (normalised DModX), "
        "with its 0.99 limit. About 1 % of the pixels should lie above it; many more mean "
        "the pixels are not one homogeneous population. Bins go up to the 99.5th "
        "percentile; the grey bar holds the rest."),
    "model.t2": (
        "Per pixel: Hotelling T² of each training pixel in the model's components, with "
        "its 0.99 limit. About 1 % should lie above it; a long tail means extreme pixels "
        "such as glints or edges. Bins go up to the 99.5th percentile; the grey bar holds "
        "the rest."),
    "model.roi_distance": (
        "Per ROI: the share of its training pixels above the DModX limit (bars, coloured by "
        "reference value) and above the T² limit (markers), sorted from the largest. The "
        "dashed line is the 1 % a 0.99 limit expects. ROIs far above it have spectra the "
        "model does not describe: another substrate, saturation, drift along a scan. Click "
        "a bar to select the ROI."),
}

__all__ = ["MODEL_PLOT_HELP"]
