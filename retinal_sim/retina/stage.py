"""Retinal stage: mosaic generation + photoreceptor response computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from retinal_sim.constants import WAVELENGTHS as _CANONICAL_WAVELENGTHS
from retinal_sim.optical.media import sample_media_transmission
from retinal_sim.retina.mosaic import MosaicGenerator, PhotoreceptorMosaic
from retinal_sim.retina.transduction import NAKA_RUSHTON_DEFAULTS, naka_rushton


@dataclass(frozen=True)
class ProvenanceNote:
    """Short provenance note for a retinal model component."""

    source: str = "unspecified"
    confidence: str = "unspecified"
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RetinalPhysiologyProvenance:
    """Provenance for the retinal-front-end assumptions exposed to users."""

    lambda_max: ProvenanceNote = field(default_factory=ProvenanceNote)
    density_functions: ProvenanceNote = field(default_factory=ProvenanceNote)
    naka_rushton: ProvenanceNote = field(default_factory=ProvenanceNote)

    def as_dict(self) -> dict:
        return {
            "lambda_max": self.lambda_max.as_dict(),
            "density_functions": self.density_functions.as_dict(),
            "naka_rushton": self.naka_rushton.as_dict(),
        }


@dataclass(frozen=True)
class ApertureSamplingParams:
    """Configuration for optional receptor aperture weighting."""

    enabled: bool = False
    gaussian_sigma_scale: float = 0.5
    sigma_bin_px: float = 0.1
    truncate_sigma: float = 3.0
    method: str = "gaussian_prefilter"
    notes: str = (
        "Optional Gaussian aperture weighting approximates inner-segment acceptance "
        "by prefiltering the irradiance grid before bilinear sampling. At current "
        "proof-of-concept pixel scales many receptor apertures remain sub-pixel, "
        "so this is a controlled retinal-front-end approximation rather than a full "
        "continuous-space integration."
    )

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "gaussian_sigma_scale": self.gaussian_sigma_scale,
            "sigma_bin_px": self.sigma_bin_px,
            "truncate_sigma": self.truncate_sigma,
            "method": self.method,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class VisualStreakConfig:
    """Optional future hook for dog/cat mosaic anisotropy."""

    supported: bool = False
    enabled: bool = False
    axis: str = "horizontal"
    axis_ratio: float = 1.0
    status: str = "not_applicable"
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "supported": self.supported,
            "enabled": self.enabled,
            "axis": self.axis,
            "axis_ratio": self.axis_ratio,
            "status": self.status,
            "notes": self.notes,
        }


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
    species_name: str = ""
    provenance: RetinalPhysiologyProvenance = field(
        default_factory=RetinalPhysiologyProvenance
    )
    aperture_weighting: ApertureSamplingParams = field(
        default_factory=ApertureSamplingParams
    )
    visual_streak: VisualStreakConfig = field(default_factory=VisualStreakConfig)
    patch_center_mm: Tuple[float, float] = (0.0, 0.0)
    patch_extent_deg: float = 2.0

    def physiology_metadata(self) -> dict:
        """Return report-safe metadata describing the retinal front-end model."""
        lambda_max_values = dict(self.cone_peak_wavelengths)
        lambda_max_values["rod"] = float(self.rod_peak_wavelength)
        density_model = (
            "anisotropic_gaussian_hook"
            if self.visual_streak.enabled and self.visual_streak.supported
            else "radially_symmetric_gaussian"
        )
        return {
            "species": self.species_name,
            "model_scope": "retinal_front_end_only",
            "scope_note": (
                "This simulation stops at the photoreceptor/retinal front end. "
                "It does not include post-receptoral, retinal-circuit, or cortical "
                "perception modeling."
            ),
            "lambda_max_values_nm": lambda_max_values,
            "lambda_max_provenance": self.provenance.lambda_max.as_dict(),
            "density_function_model": density_model,
            "density_function_provenance": self.provenance.density_functions.as_dict(),
            "naka_rushton_configuration": {
                rtype: {key: float(value) for key, value in params.items()}
                for rtype, params in self.naka_rushton_params.items()
            },
            "naka_rushton_provenance": self.provenance.naka_rushton.as_dict(),
            "aperture_weighting": self.aperture_weighting.as_dict(),
            "visual_streak": self.visual_streak.as_dict(),
        }


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

        When ``RetinalParams.aperture_weighting.enabled`` is True, the spatial
        sampling step is upgraded from pure bilinear interpolation to a Gaussian
        prefilter approximation with per-receptor σ derived from the receptor
        aperture. This keeps the default path backward-compatible while making
        the refinement opt-in and explicit.

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

        runtime_aperture_meta = {
            "enabled": False,
            "method": "bilinear_only",
            "sigma_px_min": 0.0,
            "sigma_px_mean": 0.0,
            "sigma_px_max": 0.0,
            "approximation_note": self._rp.aperture_weighting.notes,
        }
        if self._rp.aperture_weighting.enabled:
            sampled, runtime_aperture_meta = self._sample_with_aperture_weighting(
                irr_data=irr_data,
                row_f=row_f,
                col_f=col_f,
                oob=oob,
                apertures_um=mosaic.apertures,
                mm_per_px=mm_per_px,
            )
        else:
            sampled = (
                irr_data[r0, c0] * (1.0 - dr) * (1.0 - dc)
                + irr_data[r0, c1] * dc * (1.0 - dr)
                + irr_data[r1, c0] * (1.0 - dc) * dr
                + irr_data[r1, c1] * dc * dr
            )
            sampled[oob] = 0.0

        physiology_meta = self._rp.physiology_metadata()
        physiology_meta["aperture_weighting"] = {
            **physiology_meta["aperture_weighting"],
            **runtime_aperture_meta,
            "pixel_scale_mm": mm_per_px,
        }

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
                "aperture_weighting_enabled": bool(
                    physiology_meta["aperture_weighting"]["enabled"]
                ),
                "aperture_sampling_method": physiology_meta["aperture_weighting"]["method"],
                "visual_streak_enabled": bool(physiology_meta["visual_streak"]["enabled"]),
                "visual_streak_status": physiology_meta["visual_streak"]["status"],
                "retinal_physiology": physiology_meta,
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

    def _sample_with_aperture_weighting(
        self,
        *,
        irr_data: np.ndarray,
        row_f: np.ndarray,
        col_f: np.ndarray,
        oob: np.ndarray,
        apertures_um: np.ndarray,
        mm_per_px: float,
    ) -> tuple[np.ndarray, dict]:
        """Approximate per-receptor aperture integration with Gaussian prefilters.

        The full continuous-space integral would convolve the retinal irradiance
        with each receptor's acceptance profile at its own aperture. For the
        current proof-of-concept grid sizes, a cached Gaussian-prefilter
        approximation is a stable compromise: receptors are grouped by binned
        σ values in pixels, each grouped irradiance stack is spatially blurred,
        and then standard bilinear sampling is applied at the receptor centers.
        """
        config = self._rp.aperture_weighting
        sigma_px = (
            np.asarray(apertures_um, dtype=np.float32)
            * 1e-3
            * float(config.gaussian_sigma_scale)
            / max(mm_per_px, 1e-12)
        )
        sigma_px = np.maximum(sigma_px, 0.0)
        sigma_bin_px = max(float(config.sigma_bin_px), 1e-6)
        sigma_bins = np.round(sigma_px / sigma_bin_px) * sigma_bin_px

        sampled = np.zeros((len(row_f), irr_data.shape[2]), dtype=np.float32)
        valid = ~oob
        if np.any(valid):
            for sigma_value in np.unique(sigma_bins[valid]):
                mask = valid & np.isclose(sigma_bins, sigma_value)
                if not np.any(mask):
                    continue

                if sigma_value <= 1e-8:
                    working = irr_data
                else:
                    working = gaussian_filter(
                        irr_data,
                        sigma=(float(sigma_value), float(sigma_value), 0.0),
                        mode="constant",
                        cval=0.0,
                        truncate=float(config.truncate_sigma),
                    ).astype(np.float32)
                sampled[mask] = self._bilinear_sample(working, row_f[mask], col_f[mask])

        sampled[oob] = 0.0
        valid_sigma = sigma_px[valid]
        sigma_stats = {
            "enabled": True,
            "method": config.method,
            "sigma_px_min": float(valid_sigma.min()) if valid_sigma.size else 0.0,
            "sigma_px_mean": float(valid_sigma.mean()) if valid_sigma.size else 0.0,
            "sigma_px_max": float(valid_sigma.max()) if valid_sigma.size else 0.0,
            "sigma_bin_px": float(config.sigma_bin_px),
            "gaussian_sigma_scale": float(config.gaussian_sigma_scale),
            "truncate_sigma": float(config.truncate_sigma),
            "approximation_note": config.notes,
        }
        return sampled, sigma_stats

    def _bilinear_sample(
        self,
        irr_data: np.ndarray,
        row_f: np.ndarray,
        col_f: np.ndarray,
    ) -> np.ndarray:
        """Sample one irradiance stack at floating-point coordinates."""
        h, w, _ = irr_data.shape
        col_f_c = np.clip(col_f, 0.0, w - 1)
        row_f_c = np.clip(row_f, 0.0, h - 1)

        r0 = np.floor(row_f_c).astype(np.intp)
        c0 = np.floor(col_f_c).astype(np.intp)
        r1 = np.minimum(r0 + 1, h - 1)
        c1 = np.minimum(c0 + 1, w - 1)

        dr = (row_f_c - r0.astype(float))[:, np.newaxis]
        dc = (col_f_c - c0.astype(float))[:, np.newaxis]

        return (
            irr_data[r0, c0] * (1.0 - dr) * (1.0 - dc)
            + irr_data[r0, c1] * dc * (1.0 - dr)
            + irr_data[r1, c0] * (1.0 - dc) * dr
            + irr_data[r1, c1] * dc * dr
        ).astype(np.float32)
