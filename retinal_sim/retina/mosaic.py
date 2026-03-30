"""Photoreceptor mosaic generator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Photoreceptor:
    position: Tuple[float, float]   # (x_mm, y_mm)
    type: str                        # 'S_cone' | 'M_cone' | 'L_cone' | 'rod'
    aperture_um: float
    sensitivity: np.ndarray          # S(λ), shape (N_λ,)


@dataclass
class PhotoreceptorMosaic:
    positions: np.ndarray    # (N, 2) float32
    types: np.ndarray        # (N,) dtype str or int
    apertures: np.ndarray    # (N,) float32
    sensitivities: np.ndarray  # (N, N_λ) float32
    voronoi: Optional[object] = field(default=None, repr=False)

    @property
    def n_receptors(self) -> int:
        return len(self.positions)


class MosaicGenerator:
    """Generates species-appropriate photoreceptor mosaics.

    PoC implementation: jittered grid (10x faster than Poisson disk).
    Upgrade path: full Poisson disk sampling with spatially varying radius.
    """

    def __init__(self, retinal_params: object, optical_params: object) -> None:
        raise NotImplementedError

    def generate(self, seed: int = 0) -> PhotoreceptorMosaic:
        raise NotImplementedError
