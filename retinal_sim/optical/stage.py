"""Optical stage: integrates pupil, PSF, and media into a single convolution pass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.ndimage import convolve

from retinal_sim.optical.media import sample_media_transmission
from retinal_sim.optical.psf import PSFGenerator

_REFERENCE_PUPIL_DIAMETER_MM = 3.0


@dataclass
class OpticalParams:
    """Species-specific optical parameters."""

    pupil_shape: str                      # 'circular' | 'slit'
    pupil_diameter_mm: float             # diameter for circular, slit width for slit
    pupil_height_mm: Optional[float] = None
    axial_length_mm: float = 0.0
    focal_length_mm: float = 0.0
    corneal_radius_mm: float = 0.0
    lca_diopters: float = 0.0            # Total LCA span from 400-700 nm, focused at 555 nm
    media_transmission: Optional[Callable[[np.ndarray], np.ndarray]] = None
    zernike_coeffs: Dict[str, float] = field(default_factory=dict)

    def pupil_area_mm2(self) -> float:
        """Return pupil area in mm² using the R1 geometric approximation."""
        width_mm = float(self.pupil_diameter_mm)
        if self.pupil_shape == "slit":
            height_mm = float(self.pupil_height_mm or width_mm)
            return float(np.pi * (width_mm / 2.0) * (height_mm / 2.0))
        return float(np.pi * (width_mm / 2.0) ** 2)

    def pupil_extent_mm(self, axis: str) -> float:
        """Return the pupil extent controlling blur along one image axis."""
        width_mm = float(self.pupil_diameter_mm)
        if self.pupil_shape != "slit":
            return width_mm

        height_mm = float(self.pupil_height_mm or width_mm)
        if axis == "x":
            return width_mm
        if axis == "y":
            return height_mm
        raise ValueError(f"Unknown pupil axis {axis!r}; expected 'x' or 'y'")

    def area_equivalent_diameter_mm(self) -> float:
        """Return the diameter of a circular pupil with the same area."""
        return float(2.0 * np.sqrt(self.pupil_area_mm2() / np.pi))

    def effective_f_number(self, axis: Optional[str] = None) -> float:
        """Return the effective f-number overall or for one axis."""
        if axis is None:
            pupil_extent_mm = self.area_equivalent_diameter_mm()
        else:
            pupil_extent_mm = self.pupil_extent_mm(axis)
        return float(self.focal_length_mm / pupil_extent_mm)

    def anisotropy_active(self) -> bool:
        """Whether this pupil configuration should produce anisotropic blur."""
        return (
            self.pupil_shape == "slit"
            and self.pupil_height_mm is not None
            and self.pupil_height_mm > self.pupil_diameter_mm
        )


@dataclass
class RetinalIrradiance:
    """(H, W, N_lambda) spectral irradiance at the retinal surface."""

    data: np.ndarray          # float32
    wavelengths: np.ndarray   # nm
    metadata: dict = field(default_factory=dict)


class OpticalStage:
    """Applies the full anterior-eye optical model to a spectral image."""

    def __init__(self, params: OpticalParams) -> None:
        self._params = params
        self._psf_gen = PSFGenerator(params)

    def compute_psf(
        self,
        wavelengths: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """Return PSF kernels and optional axis-aware diagnostics."""
        return self._psf_gen.gaussian_psf(
            np.asarray(wavelengths, dtype=float),
            return_metadata=return_metadata,
        )

    def apply(self, spectral_image: object, scene: object = None) -> RetinalIrradiance:
        """Convolve spectral image with wavelength-dependent Gaussian PSF."""
        data = np.asarray(spectral_image.data, dtype=np.float32)
        wavelengths = np.asarray(spectral_image.wavelengths, dtype=float)

        if (
            scene is not None
            and hasattr(scene, "mm_per_pixel")
            and scene.mm_per_pixel[0] > 0
        ):
            pixel_scale_mm = float(scene.mm_per_pixel[0])
        else:
            pixel_scale_mm = 0.001

        defocus_diopters = (
            float(getattr(scene, "defocus_residual_diopters", 0.0))
            if scene is not None
            else 0.0
        )

        psf_gen = PSFGenerator(self._params, pixel_scale_mm_per_px=pixel_scale_mm)
        kernels, psf_metadata = psf_gen.gaussian_psf(
            wavelengths,
            defocus_diopters=defocus_diopters,
            return_metadata=True,
        )

        transmission, transmission_summary = sample_media_transmission(
            self._params.media_transmission,
            wavelengths,
        )
        transmission = transmission.astype(np.float32)

        reference_area_mm2 = float(np.pi * (_REFERENCE_PUPIL_DIAMETER_MM / 2.0) ** 2)
        pupil_area_mm2 = self._params.pupil_area_mm2()
        pupil_throughput_scale = float(pupil_area_mm2 / reference_area_mm2)

        result = np.empty_like(data)
        for i in range(len(wavelengths)):
            band = data[:, :, i] * transmission[i] * pupil_throughput_scale
            result[:, :, i] = convolve(band, kernels[i], mode="reflect").astype(np.float32)

        return RetinalIrradiance(
            data=result,
            wavelengths=wavelengths,
            metadata={
                "pixel_scale_mm": pixel_scale_mm,
                "defocus_diopters": defocus_diopters,
                "defocus_residual_diopters": defocus_diopters,
                "pupil_shape": self._params.pupil_shape,
                "pupil_area_mm2": pupil_area_mm2,
                "reference_pupil_area_mm2": reference_area_mm2,
                "pupil_throughput_scale": pupil_throughput_scale,
                "effective_f_number": self._params.effective_f_number(),
                "effective_f_number_x": self._params.effective_f_number("x"),
                "effective_f_number_y": self._params.effective_f_number("y"),
                "anisotropy_active": self._params.anisotropy_active(),
                "lca_reference_wavelength_nm": float(psf_metadata["lca_reference_wavelength_nm"]),
                "lca_anchor_wavelengths_nm": psf_metadata["lca_anchor_wavelengths_nm"].tolist(),
                "lca_offset_diopters": psf_metadata["lca_offset_diopters"].tolist(),
                "total_defocus_diopters_by_wavelength": psf_metadata["total_defocus_diopters_by_wavelength"].tolist(),
                "media_transmission_applied": True,
                "media_transmission_values": transmission.astype(float).tolist(),
                "media_transmission_source": transmission_summary.get("source", "unknown"),
                "media_transmission_summary": transmission_summary,
                "psf_sigma_mm_x": psf_metadata["sigma_mm_x"].tolist(),
                "psf_sigma_mm_y": psf_metadata["sigma_mm_y"].tolist(),
                "psf_sigma_px_x": psf_metadata["sigma_px_x"].tolist(),
                "psf_sigma_px_y": psf_metadata["sigma_px_y"].tolist(),
            },
        )
