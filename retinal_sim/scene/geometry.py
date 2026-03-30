"""Scene geometry: maps physical scene dimensions to retinal extent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# PoC patch size (degrees).  SceneDescription.clipped is True when scene exceeds this.
_POC_PATCH_DEG = 2.0

# Hard-cutoff maximum accommodation per species (diopters).
# Source: §0 of architecture doc.  Age-dependent for humans; PoC uses young-adult value.
_MAX_ACCOMMODATION = {
    "human": 10.0,
    "dog": 2.5,
    "cat": 3.0,
}
_DEFAULT_MAX_ACCOMMODATION = 10.0  # fallback for unknown species


@dataclass
class SceneDescription:
    scene_width_m: float
    scene_height_m: float
    viewing_distance_m: float
    angular_width_deg: float
    angular_height_deg: float
    retinal_width_mm: float
    retinal_height_mm: float
    mm_per_pixel: Tuple[float, float]
    accommodation_demand_diopters: float
    defocus_residual_diopters: float
    clipped: bool
    scene_covers_patch_fraction: float


class SceneGeometry:
    """Computes the mapping from physical scene dimensions to retinal coordinates.

    Args:
        scene_width_m: Physical width of the scene (metres).
        viewing_distance_m: Eye-to-scene distance (metres). Use inf for collimated.
        scene_height_m: Physical height (metres). Inferred from aspect ratio if None.
    """

    def __init__(
        self,
        scene_width_m: float,
        viewing_distance_m: float = float("inf"),
        scene_height_m: Optional[float] = None,
    ) -> None:
        if scene_width_m <= 0:
            raise ValueError(f"scene_width_m must be > 0, got {scene_width_m}")
        if viewing_distance_m <= 0:
            raise ValueError(f"viewing_distance_m must be > 0, got {viewing_distance_m}")
        self.scene_width_m = float(scene_width_m)
        self.viewing_distance_m = float(viewing_distance_m)
        self.scene_height_m = float(scene_height_m) if scene_height_m is not None else None

    def compute(
        self,
        image_shape: Tuple[int, int],
        optical_params: object,
    ) -> SceneDescription:
        """Compute retinal mapping for this scene/species combination.

        Args:
            image_shape: (height_px, width_px) of the input image.
            optical_params: OpticalParams instance (only focal_length_mm is read here).

        Returns:
            SceneDescription with all geometry fields populated.
        """
        height_px, width_px = image_shape

        # Infer scene height from aspect ratio when not explicitly given.
        scene_height_m = (
            self.scene_height_m
            if self.scene_height_m is not None
            else self.scene_width_m * height_px / width_px
        )

        d = self.viewing_distance_m
        focal_mm = optical_params.focal_length_mm

        # --- Angular subtense (exact arctan formula) ---
        if np.isinf(d):
            angular_width_deg = 0.0
            angular_height_deg = 0.0
        else:
            angular_width_deg = float(
                2.0 * np.degrees(np.arctan(self.scene_width_m / (2.0 * d)))
            )
            angular_height_deg = float(
                2.0 * np.degrees(np.arctan(scene_height_m / (2.0 * d)))
            )

        # --- Retinal extent ---
        retinal_width_mm = float(
            2.0 * focal_mm * np.tan(np.radians(angular_width_deg / 2.0))
        )
        retinal_height_mm = float(
            2.0 * focal_mm * np.tan(np.radians(angular_height_deg / 2.0))
        )

        # --- Pixels → retinal mm ---
        mm_per_pixel_x = retinal_width_mm / width_px if width_px > 0 else 0.0
        mm_per_pixel_y = retinal_height_mm / height_px if height_px > 0 else 0.0

        # --- Accommodation + defocus ---
        if np.isinf(d):
            accommodation_demand = 0.0
        else:
            accommodation_demand = 1.0 / d

        species_name = getattr(optical_params, "_species_name", None)
        max_accomm = _MAX_ACCOMMODATION.get(species_name, _DEFAULT_MAX_ACCOMMODATION)
        defocus_residual = float(max(0.0, accommodation_demand - max_accomm))

        # --- Patch clipping (2° PoC patch) ---
        clipped = angular_width_deg > _POC_PATCH_DEG or angular_height_deg > _POC_PATCH_DEG
        patch_area_deg2 = _POC_PATCH_DEG ** 2
        scene_area_deg2 = angular_width_deg * angular_height_deg
        scene_covers_patch_fraction = float(
            min(1.0, scene_area_deg2 / patch_area_deg2) if patch_area_deg2 > 0 else 0.0
        )

        return SceneDescription(
            scene_width_m=self.scene_width_m,
            scene_height_m=scene_height_m,
            viewing_distance_m=self.viewing_distance_m,
            angular_width_deg=angular_width_deg,
            angular_height_deg=angular_height_deg,
            retinal_width_mm=retinal_width_mm,
            retinal_height_mm=retinal_height_mm,
            mm_per_pixel=(mm_per_pixel_x, mm_per_pixel_y),
            accommodation_demand_diopters=float(accommodation_demand),
            defocus_residual_diopters=defocus_residual,
            clipped=clipped,
            scene_covers_patch_fraction=scene_covers_patch_fraction,
        )
