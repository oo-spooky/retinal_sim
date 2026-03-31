"""PSF computation: diffraction-limited + Zernike aberrations."""
from __future__ import annotations

import numpy as np


# Default retinal pixel scale used when no scene metadata is available (1 µm/px).
_DEFAULT_PIXEL_SCALE_MM = 0.001


class PSFGenerator:
    """Computes wavelength-dependent PSF kernels.

    Supports:
    - Gaussian placeholder PSF (fast, for PoC)
    - Diffraction-limited PSF (Fraunhofer, FFT of pupil function)
    - Zernike aberration overlay (defocus, spherical, coma, astigmatism)

    Args:
        optical_params: OpticalParams instance (reads focal_length_mm, pupil_diameter_mm).
        pixel_scale_mm_per_px: Retinal pixel scale in mm per pixel. Defaults to 1 µm/px.
    """

    def __init__(
        self,
        optical_params: object,
        pixel_scale_mm_per_px: float = _DEFAULT_PIXEL_SCALE_MM,
    ) -> None:
        self._params = optical_params
        self._pixel_scale_mm = float(pixel_scale_mm_per_px)

    def gaussian_psf(
        self,
        wavelengths: np.ndarray,
        kernel_size: int = 31,
        defocus_diopters: float = 0.0,
    ) -> np.ndarray:
        """Return (N_λ, K, K) Gaussian PSF kernels as a placeholder.

        Sigma is the quadrature sum of two physical components:

        1. Diffraction-limited component — Gaussian approximation of the Airy disk:
               FWHM_diff = 1.22 * λ * f#
               sigma_diff = FWHM_diff / (2 * sqrt(2 * ln 2))
           where f# = focal_length_mm / pupil_diameter_mm.

        2. Defocus blur — geometric circle of confusion (thin-lens approximation):
               r_coc = focal_length_mm * defocus_D * pupil_diameter_mm / 2000
               sigma_defocus = r_coc / sqrt(2)

        Each kernel is normalised to sum exactly 1.0 (float64) so that
        |sum(PSF) - 1.0| < 1e-6 per wavelength band (§11b energy conservation).

        Args:
            wavelengths: Wavelength values in nm, shape (N_λ,).
            kernel_size: Side length K of the square kernel (should be odd).
            defocus_diopters: Residual defocus in diopters (0 = perfectly in focus).

        Returns:
            kernels: float64 array of shape (N_λ, K, K), each plane sums to 1.0.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        n_lam = len(wavelengths)
        k = kernel_size

        f_mm = self._params.focal_length_mm
        d_p_mm = self._params.pupil_diameter_mm
        f_number = f_mm / d_p_mm

        # Coordinate grid centred on kernel centre (in pixels).
        center = k // 2
        y, x = np.mgrid[-center : k - center, -center : k - center].astype(float)
        r2 = x**2 + y**2

        kernels = np.empty((n_lam, k, k), dtype=np.float64)

        for i, lam_nm in enumerate(wavelengths):
            lam_mm = lam_nm * 1e-6  # nm → mm

            # Diffraction-limited sigma (Gaussian approximation of Airy disk).
            fwhm_diff_mm = 1.22 * lam_mm * f_number
            sigma_diff_mm = fwhm_diff_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # Defocus blur circle radius: r = f * δD * D_p / 2000 (thin-lens, diopters).
            r_coc_mm = f_mm * float(defocus_diopters) * d_p_mm / 2000.0
            sigma_defocus_mm = abs(r_coc_mm) / np.sqrt(2.0)

            # Combined sigma (quadrature sum); minimum 0.5 px to avoid delta singularities.
            sigma_mm = np.sqrt(sigma_diff_mm**2 + sigma_defocus_mm**2)
            sigma_px = max(sigma_mm / self._pixel_scale_mm, 0.5)

            kernel = np.exp(-r2 / (2.0 * sigma_px**2))
            kernel /= kernel.sum()  # exact normalisation in float64
            kernels[i] = kernel

        return kernels

    def diffraction_psf(
        self, wavelengths: np.ndarray, kernel_size: int = 64
    ) -> np.ndarray:
        """Return (N_λ, K, K) diffraction-limited PSF via FFT of pupil function."""
        raise NotImplementedError
