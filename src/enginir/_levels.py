"""The default detector levels (a light module: the launcher's Preferences import it)."""

#: Black and white detector levels used when a dataset is converted to reflectance or
#: absorbance without reference spectra: the camera has corrected its counts with its own
#: dark and white reference, scaling white to this level. 3100 is the white-balance target
#: of the EVK Helios (Kestreleye) setting; it was 3500 before engiNIR 1.7.3. The viewer's
#: Tools > Preferences can change both.
DEFAULT_BLACK_LEVEL = 128
DEFAULT_WHITE_LEVEL = 3100
