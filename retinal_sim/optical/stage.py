"""Optical stage: integrates pupil, PSF, and media into a single convolution pass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class OpticalParams:
    """Species-specific optical parameters."""
    pupil_shape: str                      # 'circular' | 'slit'
    pupil_diameter_mm: float
    axial_length_mm: float
    focal_length_mm: float
    corneal_radius_mm: float
    lca_diopters: float                   # Longitudinal chromatic aberration range
    media_transmission: Optional[Callable[[np.ndarray], np.ndarray]] = None
    zernike_coeffs: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetinalIrradiance:
    """(H, W, N_λ) spectral irradiance at the retinal surface."""
    data: np.ndarray          # float32
    wavelengths: np.ndarray   # nm
    metadata: dict = field(default_factory=dict)


class OpticalStage:
    """Applies the full anterior-eye optical model to a spectral image.

    Args:
        params: Species-specific optical parameters.
    """

    def __init__(self, params: OpticalParams) -> None:
        raise NotImplementedError

    def compute_psf(self, wavelengths: np.ndarray) -> np.ndarray:
        """Return (N_λ, K, K) PSF array for the given wavelengths."""
        raise NotImplementedError

    def apply(self, spectral_image: object, scene: object = None) -> RetinalIrradiance:
        """Convolve spectral image with wavelength-dependent PSF."""
        raise NotImplementedError
