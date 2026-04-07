"""Species configuration loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from retinal_sim.optical.media import load_media_transmission_table
from retinal_sim.retina.opsin import LAMBDA_MAX_PROVENANCE

_DATA_DIR = Path(__file__).parent.parent / "data" / "species"
_VALID_SPECIES = {"human", "dog", "cat"}


@dataclass
class SpeciesConfig:
    """Consolidated optical + retinal parameters for one species."""
    name: str
    optical: "OpticalParams"
    retinal: "RetinalParams"

    @classmethod
    def load(cls, species_name: str) -> "SpeciesConfig":
        """Load species config from data/species/{species_name}.yaml.

        Args:
            species_name: 'human' | 'dog' | 'cat'

        Returns:
            SpeciesConfig with fully populated optical and retinal params.

        Raises:
            ValueError: If species_name is not recognised.
        """
        if species_name not in _VALID_SPECIES:
            raise ValueError(
                f"Unknown species '{species_name}'. "
                f"Choose from: {sorted(_VALID_SPECIES)}"
            )
        with open(_DATA_DIR / f"{species_name}.yaml") as fh:
            raw = yaml.safe_load(fh)
        return cls(
            name=species_name,
            optical=_build_optical(raw["optical"], species_name),
            retinal=_build_retinal(raw["retinal"], species_name),
        )


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_optical(d: dict, species_name: str) -> object:
    from retinal_sim.optical.stage import OpticalParams
    media_transmission = None
    media_table = d.get("media_transmission_table")
    if media_table is not None:
        media_transmission = load_media_transmission_table(str(media_table))
    params = OpticalParams(
        pupil_shape=str(d["pupil_shape"]),
        pupil_diameter_mm=float(d["pupil_diameter_mm"]),
        pupil_height_mm=(
            float(d["pupil_height_mm"])
            if d.get("pupil_height_mm") is not None
            else None
        ),
        axial_length_mm=float(d["axial_length_mm"]),
        focal_length_mm=float(d["focal_length_mm"]),
        corneal_radius_mm=float(d["corneal_radius_mm"]),
        lca_diopters=float(d["lca_diopters"]),
        media_transmission=media_transmission,
        zernike_coeffs=dict(d.get("zernike_coeffs") or {}),
    )
    params._species_name = species_name  # used by SceneGeometry for per-species accommodation limits
    return params


def _build_retinal(d: dict, species_name: str) -> object:
    from retinal_sim.retina.stage import (
        ApertureSamplingParams,
        ProvenanceNote,
        RetinalParams,
        RetinalPhysiologyProvenance,
        VisualStreakConfig,
    )

    cone_types = list(d["cone_types"])
    cone_peak_wl = {k: float(v) for k, v in d["cone_peak_wavelengths"].items()}
    cone_ratio = {k: float(v) for k, v in d["cone_ratio"].items()}
    naka_rushton = {
        rtype: {p: float(v) for p, v in params.items()}
        for rtype, params in d["naka_rushton_params"].items()
    }

    peak_cone = float(d["peak_cone_density_mm2"])
    peak_rod = float(d["peak_rod_density_mm2"])
    cone_scale = float(d["cone_density_scale_mm"])
    rod_scale = float(d["rod_density_scale_mm"])
    rod_free_r = float(d["rod_free_zone_radius_mm"])
    provenance_raw = d.get("provenance") or {}
    visual_streak = _build_visual_streak(d.get("visual_streak") or {})
    provenance = RetinalPhysiologyProvenance(
        lambda_max=_build_provenance_note(
            provenance_raw.get("lambda_max"),
            default=LAMBDA_MAX_PROVENANCE.get(species_name, {}),
        ),
        density_functions=_build_provenance_note(
            provenance_raw.get("density_functions"),
            default={
                "source": "Species YAML density curves are literature-backed proof-of-concept Gaussian fits.",
                "confidence": "moderate",
                "notes": (
                    "The current implementation uses radially symmetric Gaussian falloffs plus "
                    "an optional visual-streak hook, not a full digitized histology map."
                ),
            },
        ),
        naka_rushton=_build_provenance_note(
            provenance_raw.get("naka_rushton"),
            default={
                "source": (
                    "Naka-Rushton parameters are configurable retinal-front-end model settings "
                    "stored in species YAML."
                ),
                "confidence": "low",
                "notes": (
                    "These parameters preserve the proof-of-concept defaults unless explicitly "
                    "changed, and they are not claimed as externally validated species-specific "
                    "transduction fits."
                ),
            },
        ),
    )
    aperture_weighting = _build_aperture_weighting(d.get("aperture_weighting") or {})

    return RetinalParams(
        species_name=species_name,
        cone_types=cone_types,
        cone_peak_wavelengths=cone_peak_wl,
        rod_peak_wavelength=float(d["rod_peak_wavelength"]),
        cone_density_fn=_make_cone_density_fn(
            peak_cone, cone_scale, cone_ratio, visual_streak,
        ),
        rod_density_fn=_make_rod_density_fn(
            peak_rod, rod_scale, rod_free_r, visual_streak,
        ),
        cone_ratio_fn=_make_cone_ratio_fn(cone_ratio),
        naka_rushton_params=naka_rushton,
        provenance=provenance,
        aperture_weighting=aperture_weighting,
        visual_streak=visual_streak,
    )


def _build_provenance_note(raw: dict | None, default: dict | None = None):
    from retinal_sim.retina.stage import ProvenanceNote

    merged = dict(default or {})
    merged.update(raw or {})
    return ProvenanceNote(
        source=str(merged.get("source", "unspecified")),
        confidence=str(merged.get("confidence", "unspecified")),
        notes=str(merged.get("notes", "")),
    )


def _build_aperture_weighting(raw: dict):
    from retinal_sim.retina.stage import ApertureSamplingParams

    return ApertureSamplingParams(
        enabled=bool(raw.get("enabled", False)),
        gaussian_sigma_scale=float(raw.get("gaussian_sigma_scale", 0.5)),
        sigma_bin_px=float(raw.get("sigma_bin_px", 0.1)),
        truncate_sigma=float(raw.get("truncate_sigma", 3.0)),
        method=str(raw.get("method", "gaussian_prefilter")),
        notes=str(
            raw.get(
                "notes",
                ApertureSamplingParams().notes,
            )
        ),
    )


def _build_visual_streak(raw: dict):
    from retinal_sim.retina.stage import VisualStreakConfig

    return VisualStreakConfig(
        supported=bool(raw.get("supported", False)),
        enabled=bool(raw.get("enabled", False)),
        axis=str(raw.get("axis", "horizontal")),
        axis_ratio=float(raw.get("axis_ratio", 1.0)),
        status=str(raw.get("status", "not_applicable")),
        notes=str(raw.get("notes", "")),
    )


def _effective_eccentricity(
    ecc_mm: float,
    angle: float,
    visual_streak: object,
) -> float:
    """Return angle-aware eccentricity for optional dog/cat visual streak hooks."""
    if not getattr(visual_streak, "supported", False):
        return float(ecc_mm)
    if not getattr(visual_streak, "enabled", False):
        return float(ecc_mm)
    if getattr(visual_streak, "axis", "horizontal") != "horizontal":
        return float(ecc_mm)

    axis_ratio = max(float(getattr(visual_streak, "axis_ratio", 1.0)), 1.0)
    x_mm = float(ecc_mm) * float(np.cos(angle))
    y_mm = float(ecc_mm) * float(np.sin(angle))
    return float(np.sqrt((x_mm / axis_ratio) ** 2 + y_mm ** 2))


def _make_cone_density_fn(
    peak_density: float,
    sigma_mm: float,
    cone_ratio: dict,
    visual_streak: object,
):
    """Gaussian falloff from area centralis; returns per-type densities.

    The optional visual-streak hook is intentionally disabled in the shipped
    species YAML files. This preserves the current radially symmetric proof of
    concept while giving dog/cat a clear extension seam for future anisotropy.
    """
    def cone_density_fn(ecc_mm: float, angle: float = 0.0) -> dict:
        effective_ecc_mm = _effective_eccentricity(ecc_mm, angle, visual_streak)
        total = peak_density * np.exp(-((effective_ecc_mm / sigma_mm) ** 2))
        return {t: total * r for t, r in cone_ratio.items()}
    return cone_density_fn


def _make_rod_density_fn(
    peak_density: float,
    sigma_mm: float,
    rod_free_zone_radius_mm: float,
    visual_streak: object,
):
    """Gaussian falloff; zero inside rod-free zone (human foveal pit)."""
    def rod_density_fn(ecc_mm: float, angle: float = 0.0) -> float:
        effective_ecc_mm = _effective_eccentricity(ecc_mm, angle, visual_streak)
        if effective_ecc_mm < rod_free_zone_radius_mm:
            return 0.0
        return peak_density * np.exp(
            -(((effective_ecc_mm - rod_free_zone_radius_mm) / sigma_mm) ** 2)
        )
    return rod_density_fn


def _make_cone_ratio_fn(cone_ratio: dict):
    """Eccentricity-independent cone type ratios (PoC simplification)."""
    def cone_ratio_fn(ecc_mm: float) -> dict:
        return dict(cone_ratio)
    return cone_ratio_fn
