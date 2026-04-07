"""Govardovskii et al. (2000) visual pigment nomogram.

Generates normalized absorption spectra from peak wavelength (λ_max) alone.
Supports both A1 (retinal-based) and A2 (3,4-dehydroretinal-based) chromophores.

Reference:
    Govardovskii VI, Fyhrquist N, Reuter T, Kuzmin DG, Donner K (2000).
    "In search of the visual pigment template." Visual Neuroscience 17(4): 509–528.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# A1 chromophore constants (Table 1, Govardovskii et al. 2000)
# ---------------------------------------------------------------------------
_A1_ALPHA = dict(A=69.7, a=0.880, B=28.0, b=0.922, C=-14.9, c=1.104, D=0.674)
_A1_BETA_AMPLITUDE = 0.26
_A1_BETA_LAMBDA_COEFF = 0.29   # λ_β = 0.29 × λ_max + 189  (nm)
_A1_BETA_LAMBDA_OFFSET = 189.0
_A1_BETA_WIDTH_COEFF = 0.20    # Gaussian σ = 0.20 × λ_β

# A2 chromophore α-band constants (Table 2, Govardovskii et al. 2000).
# Note: A, B, C differ from A1; a, b, c, D are close but not identical.
_A2_ALPHA = dict(A=62.7, a=0.880, B=22.7, b=0.924, C=-14.0, c=1.104, D=0.674)
_A2_BETA_AMPLITUDE = 0.29
_A2_BETA_LAMBDA_COEFF = 0.25
_A2_BETA_LAMBDA_OFFSET = 222.0
_A2_BETA_WIDTH_COEFF = 0.20


def _alpha_band(lam_max: float, wavelengths: np.ndarray, consts: dict) -> np.ndarray:
    """Compute the α-band (main absorption peak) of the nomogram.

    Uses the template function from Govardovskii et al. (2000) Eq. 1:

        S_α(x) = 1 / [exp(A(a-x)) + exp(B(b-x)) + exp(C(c-x)) + D]

    where  x = λ_max / λ.

    Args:
        lam_max:     Peak wavelength of the pigment (nm).
        wavelengths: Array of wavelengths at which to evaluate (nm).
        consts:      Dict of {A, a, B, b, C, c, D} template constants.

    Returns:
        Array of shape (N_λ,) with α-band values (not yet normalized).
    """
    x = lam_max / wavelengths
    return 1.0 / (
        np.exp(consts["A"] * (consts["a"] - x))
        + np.exp(consts["B"] * (consts["b"] - x))
        + np.exp(consts["C"] * (consts["c"] - x))
        + consts["D"]
    )


def _beta_band(
    lam_max: float,
    wavelengths: np.ndarray,
    amplitude: float,
    lam_coeff: float,
    lam_offset: float,
    width_coeff: float,
) -> np.ndarray:
    """Compute the β-band (secondary short-wavelength shoulder).

    Gaussian centred at:
        λ_β = lam_coeff × λ_max + lam_offset  (nm)

    with width:
        σ_β = width_coeff × λ_β

    Args:
        lam_max:     Pigment peak wavelength (nm).
        wavelengths: Evaluation wavelengths (nm).
        amplitude:   Peak height of the β-band relative to α-band peak.
        lam_coeff:   Coefficient in λ_β formula.
        lam_offset:  Offset (nm) in λ_β formula.
        width_coeff: Controls Gaussian σ as a fraction of λ_β.

    Returns:
        Array of shape (N_λ,) with β-band contribution.
    """
    lam_beta = lam_coeff * lam_max + lam_offset
    sigma = width_coeff * lam_beta
    return amplitude * np.exp(-((wavelengths - lam_beta) / sigma) ** 2)


def govardovskii_a1(
    lam_max: float,
    wavelengths: np.ndarray,
    include_beta: bool = True,
) -> np.ndarray:
    """Govardovskii A1 (retinal) pigment nomogram.

    Returns the normalized absorption spectrum for a visual pigment with A1
    (11-cis-retinal) chromophore and specified peak wavelength.

    Args:
        lam_max:      Peak wavelength of the pigment (nm).
        wavelengths:  1-D array of wavelengths at which to evaluate (nm).
        include_beta: If True (default), add the β-band shoulder.

    Returns:
        1-D float64 array, shape (N_λ,), peak-normalized to 1.0.

    Example:
        >>> import numpy as np
        >>> wl = np.arange(380, 721, 5, dtype=float)
        >>> s = govardovskii_a1(498.0, wl)   # rod rhodopsin
        >>> abs(wl[s.argmax()] - 498.0) < 5  # peak near 498 nm
        True
        >>> abs(s.max() - 1.0) < 1e-10
        True
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    alpha = _alpha_band(lam_max, wavelengths, _A1_ALPHA)
    if include_beta:
        beta = _beta_band(
            lam_max,
            wavelengths,
            _A1_BETA_AMPLITUDE,
            _A1_BETA_LAMBDA_COEFF,
            _A1_BETA_LAMBDA_OFFSET,
            _A1_BETA_WIDTH_COEFF,
        )
        template = alpha + beta
    else:
        template = alpha.copy()
    return template / template.max()


def govardovskii_a2(
    lam_max: float,
    wavelengths: np.ndarray,
    include_beta: bool = True,
) -> np.ndarray:
    """Govardovskii A2 (3,4-dehydroretinal) pigment nomogram.

    A2 pigments have broader absorption and a red-shifted peak relative to
    A1. They are found in many fish and some amphibians.

    Args:
        lam_max:      Peak wavelength (nm) of the A2 pigment.
        wavelengths:  Evaluation wavelengths (nm).
        include_beta: If True (default), add the β-band shoulder.

    Returns:
        1-D float64 array, shape (N_λ,), peak-normalized to 1.0.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    alpha = _alpha_band(lam_max, wavelengths, _A2_ALPHA)
    if include_beta:
        beta = _beta_band(
            lam_max,
            wavelengths,
            _A2_BETA_AMPLITUDE,
            _A2_BETA_LAMBDA_COEFF,
            _A2_BETA_LAMBDA_OFFSET,
            _A2_BETA_WIDTH_COEFF,
        )
        template = alpha + beta
    else:
        template = alpha.copy()
    return template / template.max()


# ---------------------------------------------------------------------------
# Known λ_max values (nm) from architecture doc + literature
# ---------------------------------------------------------------------------
LAMBDA_MAX: Dict[str, Dict[str, float]] = {
    "human": {"S_cone": 420.0, "M_cone": 530.0, "L_cone": 560.0, "rod": 498.0},
    "dog":   {"S_cone": 429.0,                   "L_cone": 555.0, "rod": 506.0},
    "cat":   {"S_cone": 450.0,                   "L_cone": 553.0, "rod": 501.0},
}

LAMBDA_MAX_PROVENANCE: Dict[str, Dict[str, str]] = {
    "human": {
        "source": (
            "Species λ_max anchors follow the retinal-front-end values documented in "
            "the architecture/spec set and are applied through the unchanged "
            "Govardovskii A1 nomogram."
        ),
        "confidence": "moderate",
        "notes": (
            "These are fixed species-reference peak wavelengths rather than "
            "individualized or adaptation-dependent measurements."
        ),
    },
    "dog": {
        "source": (
            "Dog λ_max anchors follow the retinal-front-end values documented in "
            "the architecture/spec set and are applied through the unchanged "
            "Govardovskii A1 nomogram."
        ),
        "confidence": "moderate",
        "notes": (
            "These anchors represent a dichromat reference model and should not be "
            "read as a full in-vivo spectral-sensitivity validation."
        ),
    },
    "cat": {
        "source": (
            "Cat λ_max anchors follow the retinal-front-end values documented in "
            "the architecture/spec set and are applied through the unchanged "
            "Govardovskii A1 nomogram."
        ),
        "confidence": "moderate",
        "notes": (
            "These anchors represent a species-reference dichromat template, not an "
            "individualized or state-dependent sensitivity estimate."
        ),
    },
}


def build_sensitivity_curves(
    species: str,
    wavelengths: np.ndarray,
    chromophore: str = "a1",
    include_beta: bool = True,
) -> Dict[str, np.ndarray]:
    """Build all spectral sensitivity curves for a given species.

    Applies the Govardovskii nomogram to each receptor type using the
    peak wavelengths from the architecture specification.

    Args:
        species:      'human' | 'dog' | 'cat'
        wavelengths:  Evaluation wavelengths (nm).
        chromophore:  'a1' (default) or 'a2'.
        include_beta: Whether to include the β-band shoulder.

    Returns:
        Dict mapping receptor type → normalized sensitivity array.

    Raises:
        ValueError: If species is not recognized.
    """
    if species not in LAMBDA_MAX:
        raise ValueError(
            f"Unknown species '{species}'. Choose from: {list(LAMBDA_MAX)}"
        )

    nomogram = govardovskii_a1 if chromophore == "a1" else govardovskii_a2
    wavelengths = np.asarray(wavelengths, dtype=float)

    return {
        receptor: nomogram(lam_max, wavelengths, include_beta=include_beta)
        for receptor, lam_max in LAMBDA_MAX[species].items()
    }
