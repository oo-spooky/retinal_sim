"""Snellen E optotype generation for acuity validation.

Generates synthetic high-contrast Snellen E images at calibrated angular sizes
for use in the Phase 9 acuity simulation.

Standard Snellen E conventions
--------------------------------
- The letter occupies a 5×5 grid (vertical spine + 3 horizontal bars).
- MAR (Minimum Angle of Resolution) = 1/5 of letter height.
- Letter height = 5 × MAR.
- Four orientations test discrimination: right, left, up, down.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 5×5 binary templates (1 = black stroke, 0 = white background)
# ---------------------------------------------------------------------------

# E facing right (standard orientation):
#   ■ ■ ■ ■ ■
#   ■ . . . .
#   ■ ■ ■ . .
#   ■ . . . .
#   ■ ■ ■ ■ ■
_E_RIGHT = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.float64,
)

_TEMPLATES: dict[str, np.ndarray] = {
    "right": _E_RIGHT,
    "left": np.fliplr(_E_RIGHT),
    "up": np.rot90(_E_RIGHT, k=1),
    "down": np.rot90(_E_RIGHT, k=-1),
}


def make_snellen_e(letter_height_px: int, orientation: str = "right") -> np.ndarray:
    """Return a binary Snellen E as a float64 2-D array (0 = black, 1 = white).

    The letter exactly fills the returned array; ``letter_height_px`` must be
    a positive multiple of 5 for pixel-exact rendering.  Non-multiples of 5
    are handled by nearest-neighbour upsampling of the 5×5 template.

    Args:
        letter_height_px: Side length of the returned square array in pixels.
        orientation:      One of ``'right'``, ``'left'``, ``'up'``, ``'down'``.

    Returns:
        ``(letter_height_px, letter_height_px)`` float64 array; 0 = black, 1 = white.
    """
    if orientation not in _TEMPLATES:
        raise ValueError(
            f"orientation must be one of {list(_TEMPLATES)}, got {orientation!r}"
        )
    if letter_height_px < 1:
        raise ValueError(f"letter_height_px must be >= 1, got {letter_height_px}")

    template = _TEMPLATES[orientation]  # (5, 5)

    # Nearest-neighbour scale to (letter_height_px, letter_height_px).
    scale = letter_height_px / 5.0
    # Build index arrays for each output pixel → which 5×5 cell it maps to.
    idx = np.floor(np.arange(letter_height_px) / scale).astype(np.intp)
    idx = np.clip(idx, 0, 4)
    scaled = template[np.ix_(idx, idx)]  # (H, H)

    # Invert: template has 1=black; output has 0=black, 1=white.
    return 1.0 - scaled


def snellen_scene_rgb(
    angular_size_arcmin: float,
    patch_arcmin: float,
    image_size_px: int = 64,
    orientation: str = "right",
) -> np.ndarray:
    """Generate an RGB Snellen-E scene centred in a white background.

    The letter angular height equals ``angular_size_arcmin``; the scene spans
    ``patch_arcmin`` × ``patch_arcmin`` of visual angle.

    Args:
        angular_size_arcmin: Letter height in arcminutes.
        patch_arcmin:        Total scene extent in arcminutes (both axes).
        image_size_px:       Output image side length in pixels.
        orientation:         Letter orientation (``'right'``, ``'left'``, etc.).

    Returns:
        ``(image_size_px, image_size_px, 3)`` uint8 RGB array.
        White background (255, 255, 255); black letter (0, 0, 0).
    """
    if patch_arcmin <= 0:
        raise ValueError("patch_arcmin must be positive")
    if angular_size_arcmin <= 0:
        raise ValueError("angular_size_arcmin must be positive")
    if angular_size_arcmin > patch_arcmin:
        raise ValueError("angular_size_arcmin must not exceed patch_arcmin")

    # Letter height in pixels (nearest multiple of 5 for correct template scaling).
    letter_frac = angular_size_arcmin / patch_arcmin
    letter_raw_px = letter_frac * image_size_px
    # Round to nearest multiple of 5, minimum 5.
    letter_height_px = max(5, int(round(letter_raw_px / 5.0)) * 5)
    letter_height_px = min(letter_height_px, image_size_px)

    # Build letter patch (0 = black, 1 = white, float64).
    letter = make_snellen_e(letter_height_px, orientation)

    # Place letter centred in a white canvas.
    canvas = np.ones((image_size_px, image_size_px), dtype=np.float64)
    r0 = (image_size_px - letter_height_px) // 2
    c0 = (image_size_px - letter_height_px) // 2
    canvas[r0 : r0 + letter_height_px, c0 : c0 + letter_height_px] = letter

    # Convert to uint8 RGB.
    gray_u8 = (canvas * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
