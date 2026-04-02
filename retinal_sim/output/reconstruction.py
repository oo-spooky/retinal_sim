"""Grid reconstruction from mosaic activations."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def render_reconstructed(
    activation: object,
    grid_shape: Tuple[int, int],
    mm_range: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Inverse-map from mosaic activations back to a regular grid.

    Uses nearest-neighbour interpolation from scattered receptor positions to
    a regular pixel grid.  This shows what information is preserved/lost by
    retinal sampling — it is explicitly NOT a model of what the animal
    perceives (that requires neural processing beyond the retina).

    Args:
        activation:  MosaicActivation from RetinalStage.compute_response().
        grid_shape:  (H, W) output grid shape.
        mm_range:    (xmin, xmax, ymin, ymax) bounding box in mm.
                     Derived from receptor positions if None.

    Returns:
        Float32 luminance image of shape (H, W) with values in [0, 1].
    """
    from scipy.interpolate import griddata

    positions = np.asarray(activation.mosaic.positions, dtype=np.float64)  # (N, 2)
    responses = np.clip(
        np.asarray(activation.responses, dtype=np.float64), 0.0, 1.0
    )  # (N,)

    H, W = grid_shape

    if len(positions) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Bounding box (with margin so border receptors aren't at the image edge)
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

    xs = np.linspace(xmin, xmax, W)
    ys = np.linspace(ymin, ymax, H)
    grid_x, grid_y = np.meshgrid(xs, ys)

    recon = griddata(positions, responses, (grid_x, grid_y), method="nearest")
    return np.clip(recon, 0.0, 1.0).astype(np.float32)
