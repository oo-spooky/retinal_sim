"""Grid reconstruction from mosaic activations."""
from __future__ import annotations

import numpy as np


def render_reconstructed(activation: object, grid_shape: tuple, **kwargs) -> np.ndarray:
    """Inverse-map from mosaic activations back to a regular grid.

    This shows what information is preserved/lost by retinal sampling — it is
    explicitly NOT a model of what the animal perceives (that requires neural
    processing beyond the retina).
    """
    raise NotImplementedError
