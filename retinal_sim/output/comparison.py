"""Multi-species comparison panel renderer."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from retinal_sim.output.voronoi import _TYPE_BASE_COLOR, _FALLBACK_COLOR, render_voronoi


def render_comparison(
    activations: Dict[str, object],
    output_size: Tuple[int, int] = (256, 256),
    title: str = "Comparative rendering",
) -> object:
    """Render a side-by-side comparative retinal-activation panel.

    Args:
        activations:  Mapping of species name → MosaicActivation.
        output_size:  (H, W) in pixels for each panel.
        title:        Overall figure title for the comparative rendering.

    Returns:
        matplotlib Figure with one Voronoi subplot per species.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(activations), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (name, act) in zip(axes, activations.items()):
        img = render_voronoi(act, output_size=output_size)
        ax.imshow(img, origin="lower", interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")

    # Hide unused axes (only relevant if activations was empty)
    for ax in axes[len(activations):]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def render_mosaic_map(
    mosaic: object,
    output_size: Tuple[int, int] = (512, 512),
    title: str = "Mosaic receptor map",
) -> object:
    """Scatter-plot receptor positions colour-coded by type.

    Args:
        mosaic:      PhotoreceptorMosaic.
        output_size: (height_px, width_px) for the figure.
        title:       Figure title.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.asarray(mosaic.positions)  # (N, 2)
    types = mosaic.types                       # (N,)

    unique_types = list(dict.fromkeys(str(t) for t in types))

    figw = output_size[1] / 100.0
    figh = output_size[0] / 100.0
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=100)

    for t in unique_types:
        mask = np.array([str(tp) == t for tp in types])
        rgb = _TYPE_BASE_COLOR.get(t, _FALLBACK_COLOR)
        # Lighten for visibility on a white background
        rgba = tuple(c * 0.8 + 0.1 for c in rgb) + (0.6,)
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            c=[rgba],
            s=2,
            label=t,
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.legend(markerscale=4, loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig
