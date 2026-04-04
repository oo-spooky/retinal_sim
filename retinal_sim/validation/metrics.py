"""Shared validation and pipeline metrics helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def receptor_pixel_coordinates(result: Any, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map receptor positions back to nearest image pixel indices."""
    height, width = image_shape
    mm_per_pixel = float(np.asarray(result.scene.mm_per_pixel).flat[0])
    positions = result.mosaic.positions
    cols = np.clip(
        np.round(positions[:, 0] / mm_per_pixel + (width - 1) / 2.0).astype(int),
        0,
        width - 1,
    )
    rows = np.clip(
        np.round(positions[:, 1] / mm_per_pixel + (height - 1) / 2.0).astype(int),
        0,
        height - 1,
    )
    return rows, cols


def stimulated_receptor_mask(result: Any, image_shape: tuple[int, int]) -> np.ndarray:
    """Return receptors whose positions fall within the scene footprint."""
    height, width = image_shape
    mm_per_pixel = float(np.asarray(result.scene.mm_per_pixel).flat[0])
    pos = result.mosaic.positions
    col_f = pos[:, 0] / mm_per_pixel + (width - 1) / 2.0
    row_f = pos[:, 1] / mm_per_pixel + (height - 1) / 2.0
    return (
        (col_f >= 0.0) & (col_f <= width - 1.0) &
        (row_f >= 0.0) & (row_f <= height - 1.0)
    )


def centered_box_mask(
    positions: np.ndarray,
    width_mm: float,
    height_mm: float | None = None,
) -> np.ndarray:
    """Return a centred rectangular mask in retinal millimetres."""
    if height_mm is None:
        height_mm = width_mm
    half_w = float(width_mm) / 2.0
    half_h = float(height_mm) / 2.0
    return (
        (np.abs(positions[:, 0]) <= half_w) &
        (np.abs(positions[:, 1]) <= half_h)
    )


def split_half_discriminability(result: Any, image_shape: tuple[int, int]) -> float:
    """Euclidean difference between left/right mean cone-response vectors."""
    _, cols = receptor_pixel_coordinates(result, image_shape)
    left_means: list[float] = []
    right_means: list[float] = []

    for receptor_type in ("L_cone", "M_cone", "S_cone"):
        type_mask = result.mosaic.types == receptor_type
        left_mask = type_mask & (cols < (image_shape[1] // 2))
        right_mask = type_mask & (cols >= (image_shape[1] // 2))
        if not np.any(left_mask) or not np.any(right_mask):
            continue
        left_means.append(float(np.mean(result.activation.responses[left_mask])))
        right_means.append(float(np.mean(result.activation.responses[right_mask])))

    if not left_means:
        return 0.0
    return float(np.linalg.norm(np.asarray(left_means) - np.asarray(right_means)))


def figure_ground_discriminability(result: Any, figure_mask: np.ndarray) -> float:
    """Return best cone-type figure/background discriminability for one result."""
    rows, cols = receptor_pixel_coordinates(result, figure_mask.shape)
    in_figure = figure_mask[rows, cols]
    best_d = 0.0

    for receptor_type in np.unique(result.mosaic.types):
        if receptor_type == "rod":
            continue
        type_mask = result.mosaic.types == receptor_type
        figure_type_mask = type_mask & in_figure
        background_type_mask = type_mask & ~in_figure
        if figure_type_mask.sum() < 5 or background_type_mask.sum() < 5:
            continue
        mean_figure = float(np.mean(result.activation.responses[figure_type_mask]))
        mean_background = float(np.mean(result.activation.responses[background_type_mask]))
        d_value = abs(mean_figure - mean_background) / (
            mean_figure + mean_background + 1e-8
        )
        best_d = max(best_d, d_value)

    return best_d


def response_contrast_by_region(
    result: Any,
    bright_mask: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    """Return normalized bright/dark receptor-response contrast inside a region."""
    rows, cols = receptor_pixel_coordinates(result, bright_mask.shape)
    bright = bright_mask[rows, cols]
    responses = result.activation.responses
    use = region_mask
    bright_use = use & bright
    dark_use = use & ~bright
    if bright_use.sum() < 5 or dark_use.sum() < 5:
        return 0.0
    bright_mean = float(np.mean(responses[bright_use]))
    dark_mean = float(np.mean(responses[dark_use]))
    return abs(bright_mean - dark_mean) / (bright_mean + dark_mean + 1e-8)


def median_by(items: Iterable[float]) -> float:
    arr = np.asarray(list(items), dtype=float)
    return float(np.median(arr)) if arr.size else 0.0


def compute_simulation_summary_metrics(
    result: Any,
    input_image: np.ndarray,
) -> Dict[str, float]:
    """Build lightweight summary metrics for a pipeline run."""
    stimulated = stimulated_receptor_mask(result, input_image.shape[:2])
    metrics: Dict[str, float] = {
        "stimulated_receptor_count": float(np.sum(stimulated)),
        "stimulated_receptor_fraction": float(np.mean(stimulated)) if stimulated.size else 0.0,
        "mean_response": float(np.mean(result.activation.responses)) if result.activation.responses.size else 0.0,
    }

    if input_image.shape[1] >= 2:
        metrics["left_right_cone_discriminability"] = split_half_discriminability(
            result, input_image.shape[:2]
        )

    if input_image.shape[0] > 0 and input_image.shape[1] > 0:
        image_gray = np.mean(input_image.astype(np.float32), axis=2)
        bright_mask = image_gray >= float(np.median(image_gray))
        radii = np.sqrt(np.sum(result.mosaic.positions ** 2, axis=1))
        if radii.size:
            inner = radii <= np.quantile(radii, 0.35)
            outer = radii >= np.quantile(radii, 0.65)
            metrics["center_contrast"] = response_contrast_by_region(result, bright_mask, inner)
            metrics["periphery_contrast"] = response_contrast_by_region(result, bright_mask, outer)

    return metrics
