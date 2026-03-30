"""Ocular media transmission: lens absorption, vitreous scatter."""
from __future__ import annotations

import numpy as np


class MediaTransmission:
    """Wavelength-dependent transmission through ocular media.

    Models lens absorption (especially UV/blue) and forward scatter.

    Args:
        species: 'human' | 'dog' | 'cat'
        age_years: Affects human lens transmission (older → more UV absorption).
    """

    def __init__(self, species: str, age_years: float = 30.0) -> None:
        raise NotImplementedError

    def transmission(self, wavelengths: np.ndarray) -> np.ndarray:
        """Return T(λ) in [0, 1] for each wavelength."""
        raise NotImplementedError
