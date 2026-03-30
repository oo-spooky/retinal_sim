"""Voronoi activation map renderer."""
from __future__ import annotations

import numpy as np


def render_voronoi(activation: object, **kwargs) -> np.ndarray:
    """Color each Voronoi cell by receptor response intensity.

    Receptor type encoded by hue: rods=gray, S-cones=blue, L-cones=green/red.
    Intensity mapped to brightness.
    """
    raise NotImplementedError
