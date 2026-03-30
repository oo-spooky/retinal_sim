"""Scene geometry: maps physical scene dimensions to retinal extent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


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
        raise NotImplementedError

    def compute(
        self,
        image_shape: Tuple[int, int],
        optical_params: object,
    ) -> SceneDescription:
        raise NotImplementedError
