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


# Per-species cone-to-linear-display mixing matrices.
# Rows are display channels (R, G, B). Columns are cone activations in the
# order given by ``_CONE_ORDER[species]``. Values are documented appearance
# approximations, not measured calibrations.
_CONE_ORDER: Dict[str, Tuple[str, ...]] = {
    "human": ("L_cone", "M_cone", "S_cone"),
    "dog":   ("L_cone", "S_cone"),
    "cat":   ("L_cone", "S_cone"),
}

# Trichromat human: a near-identity LMS→RGB with a small mix so neutral cone
# activations (L=M=S=1) produce a neutral display gray instead of pure red.
_HUMAN_MATRIX = np.array(
    [
        [0.85, 0.10, 0.05],   # R draws mainly from L
        [0.15, 0.80, 0.05],   # G draws mainly from M
        [0.00, 0.05, 0.95],   # B draws mainly from S
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
        return _HUMAN_MATRIX, _CONE_ORDER["human"]
    if name in ("dog", "cat"):
        return _DICHROMAT_MATRIX, _CONE_ORDER[name]
    raise ValueError(
        f"perceptual rendering not configured for species {species_name!r}; "
        f"supported: human, dog, cat"
    )


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
    gamma: float = 2.2,
) -> np.ndarray:
    """Mix per-cone activation maps into a gamma-encoded sRGB image.

    See module docstring for the meaning and limitations of the per-species
    mixing matrices. Returns float32 in [0, 1] with shape (H, W, 3).
    """
    matrix, cone_order = _species_matrix(species_name)
    stack = np.stack([cone_maps[c] for c in cone_order], axis=-1)  # (H, W, K)
    H, W, K = stack.shape
    flat = stack.reshape(-1, K)
    linear = flat @ matrix.T  # (H*W, 3)
    linear = np.clip(linear, 0.0, 1.0).reshape(H, W, 3)
    encoded = np.power(linear, 1.0 / gamma).astype(np.float32)
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
