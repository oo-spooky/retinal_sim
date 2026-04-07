"""Retinal stage: mosaic generation + photoreceptor response computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from retinal_sim.constants import WAVELENGTHS as _CANONICAL_WAVELENGTHS
from retinal_sim.optical.media import sample_media_transmission
from retinal_sim.retina.mosaic import MosaicGenerator, PhotoreceptorMosaic
from retinal_sim.retina.transduction import NAKA_RUSHTON_DEFAULTS, naka_rushton


@dataclass
class RetinalParams:
    """Species-specific retinal parameters."""
    cone_types: List[str]
    cone_peak_wavelengths: Dict[str, float]       # nm
    rod_peak_wavelength: float                     # nm
    cone_density_fn: Callable[[float, float], Dict[str, float]]
    rod_density_fn: Callable[[float, float], float]
    cone_ratio_fn: Callable[[float], Dict[str, float]]
    naka_rushton_params: Dict[str, Dict[str, float]]
    patch_center_mm: Tuple[float, float] = (0.0, 0.0)
    patch_extent_deg: float = 2.0


@dataclass
class MosaicActivation:
    """Per-receptor responses from a single simulation run."""
    mosaic: PhotoreceptorMosaic
    responses: np.ndarray   # (N_receptors,) float32, [0, 1]
    metadata: dict = field(default_factory=dict)


class RetinalStage:
    """Generates a photoreceptor mosaic and computes spectral integration + transduction.

    Args:
        params:         Species-specific retinal parameters.
        optical_params: Optical parameters (for coordinate scaling).
    """

    # Canonical wavelength grid — imported from constants.py to stay in sync
    # with SpectralUpsampler and MosaicGenerator.
    _WAVELENGTHS = _CANONICAL_WAVELENGTHS.astype(np.float32)

    def __init__(self, params: RetinalParams, optical_params: object) -> None:
        self._rp = params
        self._op = optical_params
        self._mosaic_gen = MosaicGenerator(params, optical_params, self._WAVELENGTHS)

    def generate_mosaic(self, seed: int = 0) -> PhotoreceptorMosaic:
        """Generate a photoreceptor mosaic via the jittered-grid algorithm.

        Args:
            seed: Integer random seed for reproducibility.

        Returns:
            PhotoreceptorMosaic populated with positions, types, apertures,
            and spectral sensitivities on the canonical 380–720 nm 5 nm grid.
        """
        return self._mosaic_gen.generate(seed=seed)

    def compute_response(
        self,
        mosaic: PhotoreceptorMosaic,
        retinal_irradiance: object,
        scene: object = None,
    ) -> MosaicActivation:
        """Compute per-receptor responses from retinal irradiance.

        Pipeline per receptor i:
          1. Bilinearly interpolate irradiance E(x_i, y_i, λ) from the
             regular (H, W, N_λ) grid to the receptor's mm position.
          2. Spectral integration:  excitation_i = Σ_λ S_i(λ)·E_i(λ)·Δλ
          3. Naka-Rushton:          response_i = NR(excitation_i; R_max, n, σ)

        NOTE: Aperture-function weighting (Gaussian inner-segment acceptance
        cone, σ ≈ aperture_diameter/2) is deferred to a later phase.  At PoC
        pixel scales the PSF already dominates aperture-induced blur, so
        bilinear interpolation is a good approximation.

        Args:
            mosaic:              PhotoreceptorMosaic from generate_mosaic().
            retinal_irradiance:  RetinalIrradiance with .data (H, W, N_λ)
                                 float32 and .wavelengths (N_λ,) in nm.
            scene:               Optional SceneDescription supplying
                                 mm_per_pixel; falls back to irradiance
                                 metadata then to patch-extent estimate.

        Returns:
            MosaicActivation with responses (N,) float32 in [0, R_max].
        """
        irr_data = np.asarray(retinal_irradiance.data, dtype=np.float32)  # (H, W, N_λ)
        irr_wl = np.asarray(retinal_irradiance.wavelengths, dtype=float)
        H, W, N_lam = irr_data.shape

        if not bool(retinal_irradiance.metadata.get("media_transmission_applied", False)):
            transmission, _ = sample_media_transmission(
                getattr(self._op, "media_transmission", None),
                irr_wl,
            )
            irr_data = irr_data * transmission[np.newaxis, np.newaxis, :].astype(np.float32)

        # ------------------------------------------------------------------
        # 1. Pixel scale (mm/px)
        # ------------------------------------------------------------------
        mm_per_px: float
        if (
            scene is not None
            and hasattr(scene, "mm_per_pixel")
            and np.asarray(scene.mm_per_pixel).flat[0] > 0
        ):
            mm_per_px = float(np.asarray(scene.mm_per_pixel).flat[0])
        elif "pixel_scale_mm" in retinal_irradiance.metadata:
            mm_per_px = float(retinal_irradiance.metadata["pixel_scale_mm"])
        else:
            # Fallback: fit 2×patch_half_mm across the larger image dimension.
            patch_half_mm = float(
                self._op.focal_length_mm
                * np.tan(np.radians(float(self._rp.patch_extent_deg) / 2.0))
            )
            mm_per_px = (2.0 * patch_half_mm) / max(H, W)

        # ------------------------------------------------------------------
        # 2. Map receptor mm positions → floating-point pixel coords
        #    Image centre (col W/2, row H/2) → patch_center_mm.
        # ------------------------------------------------------------------
        cx_mm = float(self._rp.patch_center_mm[0])
        cy_mm = float(self._rp.patch_center_mm[1])
        pos = mosaic.positions  # (N, 2) float32

        col_f = (pos[:, 0] - cx_mm) / mm_per_px + (W - 1) / 2.0
        row_f = (pos[:, 1] - cy_mm) / mm_per_px + (H - 1) / 2.0

        # ------------------------------------------------------------------
        # 3. Bilinear interpolation (fully vectorised)
        # ------------------------------------------------------------------
        # Receptors whose mm positions fall outside the image boundary receive
        # zero irradiance (architecture §0: "surrounding receptors receive no
        # stimulation").  Clip only for the in-bounds interpolation look-up.
        oob = (col_f < 0.0) | (col_f > W - 1) | (row_f < 0.0) | (row_f > H - 1)
        col_f_c = np.clip(col_f, 0.0, W - 1)
        row_f_c = np.clip(row_f, 0.0, H - 1)

        r0 = np.floor(row_f_c).astype(np.intp)
        c0 = np.floor(col_f_c).astype(np.intp)
        r1 = np.minimum(r0 + 1, H - 1)
        c1 = np.minimum(c0 + 1, W - 1)

        dr = (row_f_c - r0.astype(float))[:, np.newaxis]  # (N, 1)
        dc = (col_f_c - c0.astype(float))[:, np.newaxis]  # (N, 1)

        # Sampled irradiance at each receptor position: (N, N_λ)
        sampled = (
            irr_data[r0, c0] * (1.0 - dr) * (1.0 - dc)
            + irr_data[r0, c1] * dc * (1.0 - dr)
            + irr_data[r1, c0] * (1.0 - dc) * dr
            + irr_data[r1, c1] * dc * dr
        )
        # Zero out receptors that fall outside the image footprint.
        sampled[oob] = 0.0

        # ------------------------------------------------------------------
        # 4. Spectral integration:  excitation_i = Σ_λ S_i(λ)·E_i(λ)·Δλ
        # ------------------------------------------------------------------
        dlam = float(np.mean(np.diff(irr_wl))) if len(irr_wl) > 1 else 5.0

        # Align sensitivities to irradiance wavelength grid when grids differ.
        sens = self._align_sensitivities(mosaic.sensitivities, irr_wl)

        excitation = np.einsum("nl,nl->n", sens, sampled) * dlam  # (N,)

        # ------------------------------------------------------------------
        # 5. Naka-Rushton transduction per receptor type
        # ------------------------------------------------------------------
        responses = np.zeros(len(pos), dtype=np.float32)
        for rtype in np.unique(mosaic.types):
            mask = mosaic.types == rtype
            nr = self._rp.naka_rushton_params.get(
                rtype,
                NAKA_RUSHTON_DEFAULTS.get(
                    rtype, {"R_max": 1.0, "n": 0.7, "sigma": 0.5}
                ),
            )
            responses[mask] = naka_rushton(
                excitation[mask],
                R_max=nr["R_max"],
                n=nr["n"],
                sigma=nr["sigma"],
            ).astype(np.float32)

        return MosaicActivation(
            mosaic=mosaic,
            responses=responses,
            metadata={
                "mm_per_pixel": mm_per_px,
                "mean_excitation": float(np.mean(excitation)),
                "mean_response": float(np.mean(responses)),
                "n_receptors": len(pos),
                "media_transmission_applied": bool(
                    retinal_irradiance.metadata.get("media_transmission_applied", False)
                    or getattr(self._op, "media_transmission", None) is not None
                ),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _align_sensitivities(
        self, sensitivities: np.ndarray, target_wl: np.ndarray
    ) -> np.ndarray:
        """Return sensitivities interpolated onto *target_wl* if grids differ.

        When the mosaic's sensitivity wavelength grid matches the irradiance
        grid (the common case) the array is returned as-is.  Interpolation
        uses linear interp with out-of-range values set to zero.
        """
        stage_wl = self._WAVELENGTHS.astype(float)
        if sensitivities.shape[1] == len(target_wl) and np.allclose(
            stage_wl, target_wl, atol=0.5
        ):
            return sensitivities

        # Interpolate each receptor's curve onto the target grid.
        # Groups by unique type for efficiency (all receptors of the same
        # type share an identical curve, but mosaic stores per-receptor).
        result = np.zeros(
            (sensitivities.shape[0], len(target_wl)), dtype=np.float32
        )
        for i in range(sensitivities.shape[0]):
            result[i] = np.interp(
                target_wl, stage_wl, sensitivities[i], left=0.0, right=0.0
            )
        return result
