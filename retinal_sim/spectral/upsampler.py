"""Spectral upsampling: RGB → spectral radiance estimate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralImage:
    """(H, W, N_λ) spectral radiance array with wavelength axis."""
    data: np.ndarray       # float32, shape (H, W, N_λ)
    wavelengths: np.ndarray  # nm, shape (N_λ,)


class SpectralUpsampler:
    """Converts RGB images to spectral radiance estimates.

    Args:
        method: 'smits' (Smits 1999) or 'mallett_yuksel' (Mallett & Yuksel 2019).
        wavelength_range: (min_nm, max_nm) inclusive.
        wavelength_step: Sampling interval in nm.
    """

    def __init__(
        self,
        method: str = "smits",
        wavelength_range: tuple = (380, 720),
        wavelength_step: int = 5,
    ) -> None:
        raise NotImplementedError

    def upsample(self, rgb_image: np.ndarray) -> SpectralImage:
        """Convert (H, W, 3) uint8 or float32 RGB to SpectralImage."""
        raise NotImplementedError
