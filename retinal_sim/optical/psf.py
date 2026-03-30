"""PSF computation: diffraction-limited + Zernike aberrations."""
from __future__ import annotations

import numpy as np


class PSFGenerator:
    """Computes wavelength-dependent PSF kernels.

    Supports:
    - Gaussian placeholder PSF (fast, for PoC)
    - Diffraction-limited PSF (Fraunhofer, FFT of pupil function)
    - Zernike aberration overlay (defocus, spherical, coma, astigmatism)
    """

    def __init__(self, optical_params: object) -> None:
        raise NotImplementedError

    def gaussian_psf(
        self, wavelengths: np.ndarray, kernel_size: int = 31
    ) -> np.ndarray:
        """Return (N_λ, K, K) Gaussian PSF kernels as a placeholder."""
        raise NotImplementedError

    def diffraction_psf(
        self, wavelengths: np.ndarray, kernel_size: int = 64
    ) -> np.ndarray:
        """Return (N_λ, K, K) diffraction-limited PSF via FFT of pupil function."""
        raise NotImplementedError
