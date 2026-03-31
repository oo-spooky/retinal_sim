"""Photoreceptor transduction: Naka-Rushton nonlinearity and adaptation models."""
from __future__ import annotations

import numpy as np

# Default Naka-Rushton parameters per receptor type
NAKA_RUSHTON_DEFAULTS = {
    "S_cone": {"R_max": 1.0, "n": 0.7, "sigma": 0.5},
    "M_cone": {"R_max": 1.0, "n": 0.7, "sigma": 0.5},
    "L_cone": {"R_max": 1.0, "n": 0.7, "sigma": 0.5},
    "rod":    {"R_max": 1.0, "n": 1.0, "sigma": 0.1},  # rods saturate earlier; matches YAML species files
}


def naka_rushton(
    excitation: np.ndarray,
    R_max: float = 1.0,
    n: float = 0.7,
    sigma: float = 0.5,
) -> np.ndarray:
    """Naka-Rushton compressive nonlinearity.

    response = R_max × (excitation^n) / (excitation^n + sigma^n)

    Args:
        excitation: Photon catch / spectral excitation values (≥ 0).
        R_max:      Maximum response (normalised to 1.0).
        n:          Hill exponent (~0.7 cones, ~1.0 rods).
        sigma:      Half-saturation constant.

    Returns:
        Response array in [0, R_max], same shape as excitation.
    """
    exc = np.asarray(excitation, dtype=float)
    exc_n = np.power(np.maximum(exc, 0.0), n)
    sigma_n = sigma ** n
    return R_max * exc_n / (exc_n + sigma_n)
