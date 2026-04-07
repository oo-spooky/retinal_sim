"""Perceptual rendering: turn a MosaicActivation into a viewable sRGB image.

The rest of `retinal_sim` is a *physical* pipeline: it stops at per-receptor
photoreceptor activations. Turning those activations back into pixels you can
look at on a monitor requires assumptions about post-receptoral processing that
this project does not model. Everything in this file is therefore an
*appearance approximation* — a deliberately simple, documented mapping rather
than a perceptual model.

What this module does
---------------------
1. Reconstruct per-cone-type activation maps from the scattered receptor
   positions (one (H, W) array per cone type) via nearest-neighbour
   interpolation, the same approach used by ``render_reconstructed`` but kept
   separate per cone type instead of collapsed to luminance.
2. Mix those cone maps into an sRGB image using a species-specific 3x2 or 3x3
   "cone-to-display" matrix. The matrix for trichromats (human) is the
   identity-style mapping L→R, M→G, S→B with a small cross-channel mix to
   approximate display white. Dichromats (dog, cat) only have S and L cones,
   so the synthetic M channel is a fixed convex combination of the available
   cones — chosen so that an equi-cone-activation stimulus produces a neutral
   gray, which is the published behavior of dog/cat vision simulators based on
   Vienot/Brettel/Mollon (1999).

Why this is honest
------------------
- The mapping is *applied to physically simulated cone activations*, not to
  the original input RGB. So spectral effects (a red ball lighting up only
  L cones) propagate naturally and a dog's missing M-cone genuinely loses
  red/green discriminability.
- The mapping is documented as an approximation in this docstring, the CLI
  output, and the per-species matrices below. It is not claimed to be a
  validated perceptual model.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from retinal_sim.constants import WAVELENGTHS
from retinal_sim.retina.opsin import LAMBDA_MAX, govardovskii_a1
from retinal_sim.spectral.upsampler import _D65, _SRGB_TO_XYZ


# Per-species cone-to-linear-display mixing matrices.
# Rows are display channels (R, G, B). Columns are cone activations in the
# order given by ``_CONE_ORDER[species]``. Values are documented appearance
# approximations, not measured calibrations.
_CONE_ORDER: Dict[str, Tuple[str, ...]] = {
    "human": ("L_cone", "M_cone", "S_cone"),
    "dog":   ("L_cone", "S_cone"),
    "cat":   ("L_cone", "S_cone"),
}

# Hunt-Pointer-Estevez LMS → XYZ matrix (D65, equal-energy normalized so that
# LMS=(1,1,1) corresponds to the D65 white point in XYZ). Our opsin templates
# in retina/opsin.py are peak-normalized to 1, which matches the HPE
# convention. The inverse direction (XYZ→LMS) is the standard HPE D65 matrix:
#     [[ 0.4002, 0.7076, -0.0808],
#      [-0.2263, 1.1653,  0.0457],
#      [ 0.0000, 0.0000,  0.9182]]
# We invert it once at import time so the per-pixel hot path is one 3x3 mul.
_XYZ_TO_LMS_HPE_D65 = np.array(
    [
        [ 0.4002, 0.7076, -0.0808],
        [-0.2263, 1.1653,  0.0457],
        [ 0.0000, 0.0000,  0.9182],
    ],
    dtype=np.float64,
)
_LMS_TO_XYZ_HPE_D65 = np.linalg.inv(_XYZ_TO_LMS_HPE_D65)

# linear sRGB ↔ XYZ (D65). We invert the matrix already shipped in
# spectral/upsampler.py so the two stay numerically consistent.
_XYZ_TO_LINEAR_SRGB_D65 = np.linalg.inv(_SRGB_TO_XYZ.astype(np.float64))

# Von Kries chromatic adaptation: scale each cone so a D65 stimulus produces
# equal LMS responses. Without this, peak-normalized opsin curves under D65
# yield S << L,M (D65 has less blue power and the S band is narrow), and
# white-in renders blue-deficient even with the correct HPE matrix.
def _human_d65_cone_responses() -> np.ndarray:
    lam = LAMBDA_MAX["human"]
    L = govardovskii_a1(lam["L_cone"], WAVELENGTHS)
    M = govardovskii_a1(lam["M_cone"], WAVELENGTHS)
    S = govardovskii_a1(lam["S_cone"], WAVELENGTHS)
    d65 = np.asarray(_D65, dtype=np.float64)
    return np.array([float(np.trapezoid(L * d65, WAVELENGTHS)),
                     float(np.trapezoid(M * d65, WAVELENGTHS)),
                     float(np.trapezoid(S * d65, WAVELENGTHS))])


_HUMAN_D65_LMS = _human_d65_cone_responses()
_HUMAN_VON_KRIES = (1.0 / _HUMAN_D65_LMS) * _HUMAN_D65_LMS.max()  # diag scaling

# Composed LMS → linear sRGB transform for trichromats, with von Kries
# adaptation folded in.
_LMS_TO_LINEAR_SRGB_HUMAN = (
    _XYZ_TO_LINEAR_SRGB_D65 @ _LMS_TO_XYZ_HPE_D65 @ np.diag(_HUMAN_VON_KRIES)
).astype(np.float32)

# Legacy near-identity matrix retained for reference / non-human fallback only.
_HUMAN_MATRIX = np.array(
    [
        [0.85, 0.10, 0.05],
        [0.15, 0.80, 0.05],
        [0.00, 0.05, 0.95],
    ],
    dtype=np.float32,
)

# Dichromat dog/cat: only S and L cones. The "L" cone (~555 nm in dog,
# ~553 nm in cat) sits between the human M and L peaks, so we route it to
# both R and G in equal measure. S maps to B as in the trichromat case.
# The constraint is that equi-cone activation (S=L=1) yields neutral gray.
_DICHROMAT_MATRIX = np.array(
    [
        [0.50, 0.50],   # R: half of dichromat L
        [0.50, 0.50],   # G: half of dichromat L  + half of S → desaturated yellow-blue axis
        [0.00, 1.00],   # B: pure S
    ],
    dtype=np.float32,
)


def _species_matrix(species_name: str) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return (display matrix, cone order) for a species."""
    name = species_name.lower()
    if name == "human":
        return _LMS_TO_LINEAR_SRGB_HUMAN, _CONE_ORDER["human"]
    if name in ("dog", "cat"):
        return _DICHROMAT_MATRIX, _CONE_ORDER[name]
    raise ValueError(
        f"perceptual rendering not configured for species {species_name!r}; "
        f"supported: human, dog, cat"
    )


def _invert_naka_rushton(
    response: np.ndarray,
    sigma: float = 0.5,
    n: float = 0.7,
    r_max: float = 1.0,
) -> np.ndarray:
    """Recover a linear excitation from a Naka-Rushton-compressed response.

    Forward (retinal_sim/retina/transduction.py):
        r = R_max * x^n / (x^n + sigma^n)
    Inverse:
        x = sigma * (r / (R_max - r)) ** (1/n)

    Values at or above ``r_max`` are clipped slightly below to avoid a
    divide-by-zero; negative values are clamped to zero. The defaults match
    NAKA_RUSHTON_DEFAULTS for cones; if these drift in the forward path,
    update them here too. (TODO: surface N-R params on MosaicActivation so we
    can read them from the result instead of duplicating defaults.)
    """
    r = np.clip(np.asarray(response, dtype=np.float64), 0.0, r_max - 1e-6)
    ratio = r / (r_max - r)
    return (sigma * np.power(ratio, 1.0 / n)).astype(np.float32)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply the standard sRGB EOTF (piecewise) to linear-light values."""
    x = np.clip(linear, 0.0, 1.0)
    low = x <= 0.0031308
    out = np.where(low, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055)
    return out.astype(np.float32)


def _gamut_clip(linear: np.ndarray) -> np.ndarray:
    """Map out-of-gamut linear RGB into the unit cube. Hard clip for now."""
    return np.clip(linear, 0.0, 1.0)


def reconstruct_cone_maps(
    activation: object,
    grid_shape: Tuple[int, int],
    mm_range: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Build one (H, W) activation map per cone type from a MosaicActivation.

    Uses nearest-neighbour interpolation per cone type (rods are excluded).
    Returns an empty map for any cone type that has no receptors in the
    mosaic — callers can then decide how to fill it.
    """
    from scipy.interpolate import griddata

    H, W = grid_shape
    positions = np.asarray(activation.mosaic.positions, dtype=np.float64)
    types = np.asarray(activation.mosaic.types)
    responses = np.clip(
        np.asarray(activation.responses, dtype=np.float64), 0.0, 1.0
    )

    if mm_range is None and len(positions) > 0:
        xmin, xmax = float(positions[:, 0].min()), float(positions[:, 0].max())
        ymin, ymax = float(positions[:, 1].min()), float(positions[:, 1].max())
        dx = max((xmax - xmin) * 0.05, 0.05)
        dy = max((ymax - ymin) * 0.05, 0.05)
        xmin -= dx; xmax += dx
        ymin -= dy; ymax += dy
    elif mm_range is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in mm_range)
    else:
        xmin = xmax = ymin = ymax = 0.0

    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)
    grid_x, grid_y = np.meshgrid(xs, ys)

    cone_maps: Dict[str, np.ndarray] = {}
    for ctype in ("L_cone", "M_cone", "S_cone"):
        mask = types == ctype
        if not mask.any():
            cone_maps[ctype] = np.zeros((H, W), dtype=np.float32)
            continue
        recon = griddata(
            positions[mask], responses[mask], (grid_x, grid_y), method="nearest"
        )
        cone_maps[ctype] = np.clip(recon, 0.0, 1.0).astype(np.float32)
    return cone_maps


def cone_maps_to_srgb(
    cone_maps: Dict[str, np.ndarray],
    species_name: str,
    gamma: float = 2.2,  # retained for backward compat; ignored (sRGB EOTF used)
) -> np.ndarray:
    """Mix per-cone activation maps into a gamma-encoded sRGB image.

    Pipeline:
      1. Invert Naka-Rushton on each cone map → linear-ish excitations.
      2. Renormalize all cones by a *shared* scalar so the brightest
         excitation in the scene maps to ~1.0 (per-cone normalization would
         destroy color, so we deliberately use one global factor).
      3. Apply the species mixing matrix (HPE-derived for human;
         hand-rolled dichromat matrix for dog/cat).
      4. Hard-clip to gamut and apply the standard sRGB EOTF.

    Returns float32 in [0, 1] with shape (H, W, 3).
    """
    matrix, cone_order = _species_matrix(species_name)

    linearized = {c: _invert_naka_rushton(cone_maps[c]) for c in cone_order}
    stacked = np.stack([linearized[c] for c in cone_order], axis=-1)  # (H, W, K)

    # Shared brightness normalization across cones. The 99th percentile gives
    # headroom for specular highlights without letting a single bright pixel
    # crush the rest of the image.
    finite = stacked[np.isfinite(stacked)]
    if finite.size > 0:
        scale = float(np.percentile(finite, 99.0))
    else:
        scale = 0.0
    if scale > 1e-9:
        stacked = stacked / scale

    H, W, K = stacked.shape
    flat = stacked.reshape(-1, K).astype(np.float64)
    linear = flat @ matrix.T.astype(np.float64)  # (H*W, 3)
    linear = _gamut_clip(linear).reshape(H, W, 3)
    encoded = _linear_to_srgb(linear)
    return encoded


def render_perceptual_image(
    result: object,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convenience wrapper: SimulationResult → (H, W, 3) sRGB float32 image."""
    if grid_shape is None:
        input_shape = result.artifacts.get("input_shape") if hasattr(result, "artifacts") else None
        if input_shape is None:
            grid_shape = (256, 256)
        else:
            grid_shape = (int(input_shape[0]), int(input_shape[1]))
    cone_maps = reconstruct_cone_maps(result.activation, grid_shape)
    return cone_maps_to_srgb(cone_maps, result.species_name)
