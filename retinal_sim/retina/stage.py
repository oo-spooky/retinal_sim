"""Retinal stage: mosaic generation + photoreceptor response computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


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
    mosaic: object
    responses: np.ndarray   # (N_receptors,) float32, [0, 1]
    metadata: dict = field(default_factory=dict)


class RetinalStage:
    """Generates a photoreceptor mosaic and computes spectral integration + transduction.

    Args:
        params:         Species-specific retinal parameters.
        optical_params: Optical parameters (for coordinate scaling).
    """

    def __init__(self, params: RetinalParams, optical_params: object) -> None:
        raise NotImplementedError

    def generate_mosaic(self, seed: int = 0) -> object:
        raise NotImplementedError

    def compute_response(
        self,
        mosaic: object,
        retinal_irradiance: object,
        scene: object = None,
    ) -> MosaicActivation:
        raise NotImplementedError
