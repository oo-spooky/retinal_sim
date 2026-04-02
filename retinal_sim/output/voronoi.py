"""Voronoi activation map renderer."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Receptor type → (R, G, B) base colour at full brightness.
# rods:   achromatic (white scaled to gray by response)
# S_cone: short-wavelength → blue
# M_cone: medium-wavelength → green
# L_cone: long-wavelength → red
_TYPE_BASE_COLOR: dict = {
    "rod":    (1.0, 1.0, 1.0),
    "S_cone": (0.0, 0.0, 1.0),
    "M_cone": (0.0, 1.0, 0.0),
    "L_cone": (1.0, 0.0, 0.0),
}
_FALLBACK_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def render_voronoi(
    activation: object,
    output_size: Tuple[int, int] = (256, 256),
    mm_range: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Color each Voronoi cell by receptor response intensity.

    Each output pixel is assigned to the nearest receptor, which is
    equivalent to Voronoi tessellation.  Type is encoded by hue
    (rods→gray, S→blue, M→green, L→red) and response [0, 1] is
    mapped to brightness.

    Args:
        activation:  MosaicActivation from RetinalStage.compute_response().
        output_size: (H, W) in pixels.
        mm_range:    (xmin, xmax, ymin, ymax) bounding box in mm.
                     Derived from receptor positions if None.

    Returns:
        Float32 RGB image of shape (H, W, 3) with values in [0, 1].
    """
    from scipy.spatial import cKDTree

    positions = np.asarray(activation.mosaic.positions, dtype=np.float64)  # (N, 2)
    types = activation.mosaic.types                                          # (N,) str
    responses = np.clip(
        np.asarray(activation.responses, dtype=np.float32), 0.0, 1.0
    )  # (N,)

    N = len(positions)
    H, W = output_size

    if N == 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    # Per-receptor RGB = base_colour × response
    base_rgb = np.array(
        [_TYPE_BASE_COLOR.get(str(t), _FALLBACK_COLOR) for t in types],
        dtype=np.float32,
    )  # (N, 3)
    receptor_rgb = base_rgb * responses[:, np.newaxis]  # (N, 3)

    # Bounding box (with a small margin so border receptors aren't clipped)
    if mm_range is None:
        xmin, xmax = float(positions[:, 0].min()), float(positions[:, 0].max())
        ymin, ymax = float(positions[:, 1].min()), float(positions[:, 1].max())
        dx = max((xmax - xmin) * 0.05, 0.05)
        dy = max((ymax - ymin) * 0.05, 0.05)
        xmin -= dx; xmax += dx
        ymin -= dy; ymax += dy
    else:
        xmin, xmax, ymin, ymax = float(mm_range[0]), float(mm_range[1]), \
            float(mm_range[2]), float(mm_range[3])

    # Pixel-grid coordinates in mm
    xs = np.linspace(xmin, xmax, W, dtype=np.float64)
    ys = np.linspace(ymin, ymax, H, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pixel_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (H*W, 2)

    # Nearest-receptor assignment
    _, idx = cKDTree(positions).query(pixel_pts)  # (H*W,)

    return receptor_rgb[idx].reshape(H, W, 3)
