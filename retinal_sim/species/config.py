"""Species configuration loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

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
            retinal=_build_retinal(raw["retinal"]),
        )


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_optical(d: dict, species_name: str) -> object:
    from retinal_sim.optical.stage import OpticalParams
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
        zernike_coeffs=dict(d.get("zernike_coeffs") or {}),
    )
    params._species_name = species_name  # used by SceneGeometry for per-species accommodation limits
    return params


def _build_retinal(d: dict) -> object:
    from retinal_sim.retina.stage import RetinalParams

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

    return RetinalParams(
        cone_types=cone_types,
        cone_peak_wavelengths=cone_peak_wl,
        rod_peak_wavelength=float(d["rod_peak_wavelength"]),
        cone_density_fn=_make_cone_density_fn(peak_cone, cone_scale, cone_ratio),
        rod_density_fn=_make_rod_density_fn(peak_rod, rod_scale, rod_free_r),
        cone_ratio_fn=_make_cone_ratio_fn(cone_ratio),
        naka_rushton_params=naka_rushton,
    )


def _make_cone_density_fn(peak_density: float, sigma_mm: float, cone_ratio: dict):
    """Gaussian falloff from area centralis; returns per-type densities.

    NOTE: The `angle` parameter is accepted for API compatibility but ignored.
    Visual streak modeling (anisotropic density along horizontal meridian)
    is a PoC simplification deferred to a later phase.  Cat has a strong
    horizontal visual streak; dog has a weak one.
    TODO (Phase 5+): add angle-dependent Gaussian elongation from species YAML.
    """
    def cone_density_fn(ecc_mm: float, angle: float = 0.0) -> dict:
        total = peak_density * np.exp(-((ecc_mm / sigma_mm) ** 2))
        return {t: total * r for t, r in cone_ratio.items()}
    return cone_density_fn


def _make_rod_density_fn(
    peak_density: float, sigma_mm: float, rod_free_zone_radius_mm: float
):
    """Gaussian falloff; zero inside rod-free zone (human foveal pit)."""
    def rod_density_fn(ecc_mm: float, angle: float = 0.0) -> float:
        if ecc_mm < rod_free_zone_radius_mm:
            return 0.0
        return peak_density * np.exp(
            -(((ecc_mm - rod_free_zone_radius_mm) / sigma_mm) ** 2)
        )
    return rod_density_fn


def _make_cone_ratio_fn(cone_ratio: dict):
    """Eccentricity-independent cone type ratios (PoC simplification)."""
    def cone_ratio_fn(ecc_mm: float) -> dict:
        return dict(cone_ratio)
    return cone_ratio_fn
