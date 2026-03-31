"""Optical stage: integrates pupil, PSF, and media into a single convolution pass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.ndimage import convolve

from retinal_sim.optical.psf import PSFGenerator


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

    Convolves each wavelength band with a Gaussian PSF sized from diffraction
    physics and (optionally) defocus, then scales by media transmission.

    Args:
        params: Species-specific optical parameters.
    """

    def __init__(self, params: OpticalParams) -> None:
        self._params = params
        # PSFGenerator with default pixel scale; apply() rebuilds with actual scale.
        self._psf_gen = PSFGenerator(params)

    def compute_psf(self, wavelengths: np.ndarray) -> np.ndarray:
        """Return (N_λ, K, K) PSF array for the given wavelengths (default pixel scale)."""
        return self._psf_gen.gaussian_psf(np.asarray(wavelengths, dtype=float))

    def apply(self, spectral_image: object, scene: object = None) -> RetinalIrradiance:
        """Convolve spectral image with wavelength-dependent Gaussian PSF.

        Args:
            spectral_image: SpectralImage with .data (H, W, N_λ) float32 and
                            .wavelengths (N_λ,) in nm.
            scene: Optional SceneDescription; supplies pixel scale and defocus
                   residual when available.

        Returns:
            RetinalIrradiance with PSF-blurred spectral data (float32, same shape).
        """
        data = np.asarray(spectral_image.data, dtype=np.float32)   # (H, W, N_λ)
        wavelengths = np.asarray(spectral_image.wavelengths, dtype=float)

        # Pixel scale and defocus from scene when provided.
        if (
            scene is not None
            and hasattr(scene, "mm_per_pixel")
            and scene.mm_per_pixel[0] > 0
        ):
            pixel_scale_mm = float(scene.mm_per_pixel[0])
        else:
            pixel_scale_mm = 0.001  # 1 µm/px fallback

        defocus_diopters = (
            float(getattr(scene, "defocus_residual_diopters", 0.0))
            if scene is not None
            else 0.0
        )

        psf_gen = PSFGenerator(self._params, pixel_scale_mm_per_px=pixel_scale_mm)
        kernels = psf_gen.gaussian_psf(wavelengths, defocus_diopters=defocus_diopters)

        # Media transmission: wavelength-dependent attenuation applied before convolution.
        transmission = np.ones(len(wavelengths), dtype=np.float32)
        if self._params.media_transmission is not None:
            transmission = self._params.media_transmission(wavelengths).astype(np.float32)

        # Convolve each wavelength band with its PSF kernel.
        result = np.empty_like(data)
        for i in range(len(wavelengths)):
            band = data[:, :, i] * transmission[i]
            result[:, :, i] = convolve(band, kernels[i], mode="reflect").astype(np.float32)

        return RetinalIrradiance(
            data=result,
            wavelengths=wavelengths,
            metadata={
                "pixel_scale_mm": pixel_scale_mm,
                "defocus_diopters": defocus_diopters,
            },
        )
