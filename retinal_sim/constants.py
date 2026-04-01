"""Shared constants used across retinal_sim pipeline stages.

Centralising these here prevents silent divergence when one module's
copy of a duplicated value drifts from the others.
"""
import numpy as np

# Canonical wavelength grid: 380–720 nm inclusive at 5 nm steps (69 bands).
# All pipeline stages that produce or consume spectral data must use this grid
# (or explicitly interpolate to/from it).  Defined once here and imported by:
#   spectral/upsampler.py  — SpectralUpsampler default wavelength range
#   retina/mosaic.py       — MosaicGenerator default sensitivity grid
#   retina/stage.py        — RetinalStage._WAVELENGTHS
WAVELENGTHS: np.ndarray = np.arange(380, 721, 5, dtype=np.float64)
