"""PSF computation: diffraction-limited Gaussian approximations."""
from __future__ import annotations

import numpy as np


_DEFAULT_PIXEL_SCALE_MM = 0.001


class PSFGenerator:
    """Computes wavelength-dependent PSF kernels."""

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
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """Return Gaussian PSF kernels plus optional axis-aware diagnostics."""
        wavelengths = np.asarray(wavelengths, dtype=float)
        n_lam = len(wavelengths)
        center = kernel_size // 2
        y, x = np.mgrid[-center : kernel_size - center, -center : kernel_size - center]
        x = x.astype(float)
        y = y.astype(float)

        kernels = np.empty((n_lam, kernel_size, kernel_size), dtype=np.float64)
        sigma_mm_x = np.empty(n_lam, dtype=float)
        sigma_mm_y = np.empty(n_lam, dtype=float)
        sigma_px_x = np.empty(n_lam, dtype=float)
        sigma_px_y = np.empty(n_lam, dtype=float)

        for i, lam_nm in enumerate(wavelengths):
            axis_sigmas_mm = self._axis_sigmas_mm(lam_nm, defocus_diopters)
            sigma_mm_x[i] = axis_sigmas_mm["x"]
            sigma_mm_y[i] = axis_sigmas_mm["y"]
            sigma_px_x[i] = max(sigma_mm_x[i] / self._pixel_scale_mm, 0.5)
            sigma_px_y[i] = max(sigma_mm_y[i] / self._pixel_scale_mm, 0.5)

            exponent = (x**2) / (2.0 * sigma_px_x[i] ** 2) + (y**2) / (2.0 * sigma_px_y[i] ** 2)
            kernel = np.exp(-exponent)
            kernel /= kernel.sum()
            kernels[i] = kernel

        if not return_metadata:
            return kernels

        metadata = {
            "sigma_mm_x": sigma_mm_x,
            "sigma_mm_y": sigma_mm_y,
            "sigma_px_x": sigma_px_x,
            "sigma_px_y": sigma_px_y,
            "effective_f_number": self._params.effective_f_number(),
            "effective_f_number_x": self._params.effective_f_number("x"),
            "effective_f_number_y": self._params.effective_f_number("y"),
            "anisotropy_active": self._params.anisotropy_active(),
        }
        return kernels, metadata

    def _axis_sigmas_mm(self, wavelength_nm: float, defocus_diopters: float) -> dict[str, float]:
        """Return quadrature-combined Gaussian sigmas in mm for x/y axes."""
        f_mm = float(self._params.focal_length_mm)
        lam_mm = float(wavelength_nm) * 1e-6

        axis_sigmas = {}
        for axis in ("x", "y"):
            pupil_extent_mm = float(self._params.pupil_extent_mm(axis))
            f_number = float(self._params.effective_f_number(axis))
            sigma_diff_mm = 0.42 * lam_mm * f_number
            r_coc_mm = f_mm * float(defocus_diopters) * pupil_extent_mm / 2000.0
            sigma_defocus_mm = abs(r_coc_mm) / np.sqrt(2.0)
            axis_sigmas[axis] = float(np.sqrt(sigma_diff_mm**2 + sigma_defocus_mm**2))
        return axis_sigmas

    def diffraction_psf(
        self, wavelengths: np.ndarray, kernel_size: int = 64
    ) -> np.ndarray:
        """Return (N_lambda, K, K) diffraction-limited PSF via FFT of pupil function."""
        raise NotImplementedError
