"""Pupil aperture models: circular and slit."""
from __future__ import annotations

import numpy as np


class PupilModel:
    """Generates 2D pupil aperture masks and computes light throughput.

    Args:
        shape: 'circular' or 'slit'
        diameter_mm: Pupil diameter (or slit width) in mm.
    """

    def __init__(self, shape: str, diameter_mm: float) -> None:
        raise NotImplementedError

    def aperture_mask(self, size_px: int) -> np.ndarray:
        """Return 2D boolean mask of the pupil aperture."""
        raise NotImplementedError

    @property
    def area_mm2(self) -> float:
        raise NotImplementedError
