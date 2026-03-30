"""RetinalSimulator pipeline orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class SimulationResult:
    """Container for all pipeline stage outputs."""
    scene: object
    spectral_image: object
    retinal_irradiance: object
    mosaic: object
    activation: object


class RetinalSimulator:
    """Top-level pipeline orchestrator.

    Args:
        species: Species name ('human', 'dog', 'cat') or a SpeciesConfig object.
        patch_extent_deg: Angular size of the simulated retinal patch (degrees).
        light_level: 'photopic' | 'mesopic' | 'scotopic'
    """

    def __init__(
        self,
        species: Union[str, object],
        patch_extent_deg: float = 2.0,
        light_level: str = "photopic",
    ) -> None:
        raise NotImplementedError

    def simulate(
        self,
        input_image: np.ndarray,
        scene_width_m: Optional[float] = None,
        viewing_distance_m: float = float("inf"),
    ) -> SimulationResult:
        raise NotImplementedError

    def compare_species(
        self,
        input_image: np.ndarray,
        species_list: List[str],
        scene_width_m: Optional[float] = None,
        viewing_distance_m: float = float("inf"),
    ) -> Dict[str, SimulationResult]:
        raise NotImplementedError
