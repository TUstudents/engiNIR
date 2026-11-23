"""Detector read-out settings: the default levels and the binning factors (a light
module: the launcher's Preferences and the typed project objects import it)."""

#: Black and white detector levels used when a dataset is converted to reflectance or
#: absorbance without reference spectra: the camera has corrected its counts with its own
#: dark and white reference, scaling white to this level. 3100 is the white-balance target
#: of the EVK Helios (Kestreleye) setting; it was 3500 before engiNIR 1.7.3. The viewer's
#: Tools > Preferences can change both.
DEFAULT_BLACK_LEVEL = 128
DEFAULT_WHITE_LEVEL = 3100

#: Spatial binning factors a dataset may use: ``k`` averages ``k x k`` blocks of frames
#: and rows (``enginir.io.bin_blocks``); 1 is no binning.
BINNING_FACTORS = (1, 2, 3, 4)

#: Confidence level of the DModX and T² limits (DCrit, T²Crit) and of the default DModX
#: exclusion: 0.95, SIMCA's default. Re-exported by :mod:`enginir.validation.outliers`.
LIMIT_LEVEL = 0.95
