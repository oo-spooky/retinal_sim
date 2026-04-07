"""Structured diagnostic builders for Phase R6 traceable output families."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from retinal_sim.output.reconstruction import render_reconstructed
from retinal_sim.output.voronoi import _TYPE_BASE_COLOR, _FALLBACK_COLOR, render_voronoi

_SELECTED_WAVELENGTHS_NM = (420.0, 530.0, 650.0)


def build_retinal_irradiance_diagnostics(
    spectral_image: Any,
    retinal_irradiance: Any,
) -> Dict[str, Any]:
    """Build deterministic retinal-irradiance summaries from pipeline state."""
    data = np.asarray(retinal_irradiance.data, dtype=np.float32)
    spectral_data = np.asarray(spectral_image.data, dtype=np.float32)
    wavelengths = np.asarray(retinal_irradiance.wavelengths, dtype=np.float64)
    delivered_mean = data.mean(axis=(0, 1))
    source_mean = spectral_data.mean(axis=(0, 1))
    safe_max = max(float(np.max(data)), 1e-8)

    selected_slices = []
    for target_nm in _SELECTED_WAVELENGTHS_NM:
        idx = int(np.argmin(np.abs(wavelengths - target_nm)))
        band = data[:, :, idx]
        selected_slices.append(
            {
                "id": f"slice_{int(round(float(wavelengths[idx])))}nm",
                "label": f"Retinal irradiance slice ({int(round(float(wavelengths[idx])))} nm)",
                "wavelength_nm": float(wavelengths[idx]),
                "mean": float(np.mean(band)),
                "min": float(np.min(band)),
                "max": float(np.max(band)),
                "image_kind": "grayscale",
                "image_data": _normalize_image(band),
            }
        )

    delivered_sample_points = []
    for idx in np.linspace(0, len(wavelengths) - 1, min(len(wavelengths), 8), dtype=int):
        delivered_sample_points.append(
            {
                "wavelength_nm": float(wavelengths[idx]),
                "source_mean": float(source_mean[idx]),
                "delivered_mean": float(delivered_mean[idx]),
            }
        )

    composite = np.stack(
        [
            data[:, :, int(np.argmin(np.abs(wavelengths - 650.0)))],
            data[:, :, int(np.argmin(np.abs(wavelengths - 530.0)))],
            data[:, :, int(np.argmin(np.abs(wavelengths - 420.0)))],
        ],
        axis=-1,
    )

    metadata = retinal_irradiance.metadata
    optical_keys = (
        "pupil_shape",
        "pupil_area_mm2",
        "reference_pupil_area_mm2",
        "pupil_throughput_scale",
        "effective_f_number",
        "effective_f_number_x",
        "effective_f_number_y",
        "anisotropy_active",
        "defocus_diopters",
        "defocus_residual_diopters",
        "lca_reference_wavelength_nm",
        "lca_anchor_wavelengths_nm",
        "lca_offset_diopters",
        "total_defocus_diopters_by_wavelength",
        "media_transmission_applied",
        "media_transmission_source",
        "media_transmission_values",
        "psf_sigma_mm_x",
        "psf_sigma_mm_y",
        "psf_sigma_px_x",
        "psf_sigma_px_y",
        "pixel_scale_mm",
    )
    optical_summary = {
        key: metadata[key]
        for key in optical_keys
        if key in metadata
    }

    return {
        "family_label": "retinal irradiance diagnostics",
        "traceability_note": (
            "This family records where scene-spectrum assumptions and anterior-eye "
            "optical effects altered the retinally delivered light before receptor sampling."
        ),
        "delivered_spectrum_summary": {
            "wavelengths_nm": wavelengths.astype(float).tolist(),
            "source_mean_by_wavelength": source_mean.astype(float).tolist(),
            "delivered_mean_by_wavelength": delivered_mean.astype(float).tolist(),
            "delivered_sample_points": delivered_sample_points,
            "peak_delivered_wavelength_nm": float(wavelengths[int(np.argmax(delivered_mean))]),
            "delivered_total_energy": float(np.sum(delivered_mean)),
        },
        "selected_wavelength_slices": selected_slices,
        "band_composite": {
            "label": "Retinal irradiance RGB-style band composite",
            "channels_nm": {"red": 650.0, "green": 530.0, "blue": 420.0},
            "image_kind": "rgb",
            "image_data": np.clip(composite / safe_max, 0.0, 1.0).astype(np.float32),
        },
        "optical_stage_summary": optical_summary,
    }


def build_photoreceptor_activation_diagnostics(
    scene: Any,
    mosaic: Any,
    activation: Any,
    receptor_rows: np.ndarray,
    receptor_cols: np.ndarray,
    stimulated_mask: np.ndarray,
    input_shape: tuple[int, int],
) -> Dict[str, Any]:
    """Build deterministic receptor-sampling and activation summaries."""
    responses = np.asarray(activation.responses, dtype=np.float32)
    types = np.asarray(mosaic.types)
    positions = np.asarray(mosaic.positions, dtype=np.float32)
    unique_types = list(dict.fromkeys(str(item) for item in types))

    per_type = []
    for receptor_type in unique_types:
        mask = np.asarray([str(item) == receptor_type for item in types], dtype=bool)
        stimulated_type = mask & stimulated_mask
        response_values = responses[mask]
        per_type.append(
            {
                "receptor_type": receptor_type,
                "count": int(np.sum(mask)),
                "stimulated_count": int(np.sum(stimulated_type)),
                "stimulated_fraction": float(np.mean(stimulated_type[mask])) if np.any(mask) else 0.0,
                "mean_response": float(np.mean(response_values)) if response_values.size else 0.0,
                "median_response": float(np.median(response_values)) if response_values.size else 0.0,
                "min_response": float(np.min(response_values)) if response_values.size else 0.0,
                "max_response": float(np.max(response_values)) if response_values.size else 0.0,
            }
        )

    footprint = {
        "input_shape": [int(input_shape[0]), int(input_shape[1])],
        "stimulated_receptor_count": int(np.sum(stimulated_mask)),
        "stimulated_receptor_fraction": float(np.mean(stimulated_mask)) if stimulated_mask.size else 0.0,
        "scene_covers_patch_fraction": float(getattr(scene, "scene_covers_patch_fraction", 0.0)),
        "retinal_extent_mm": {
            "width": float(getattr(scene, "retinal_width_mm", 0.0)),
            "height": float(getattr(scene, "retinal_height_mm", 0.0)),
        },
        "stimulated_pixel_bounds": _pixel_bounds(receptor_rows, receptor_cols, stimulated_mask),
        "position_bounds_mm": _position_bounds_mm(positions, stimulated_mask),
    }

    return {
        "family_label": "photoreceptor activation diagnostics",
        "traceability_note": (
            "This family records where receptor sampling density, receptor classes, "
            "and front-end transduction changed the retained retinal information."
        ),
        "overall_summary": {
            "n_receptors": int(len(positions)),
            "mean_response": float(np.mean(responses)) if responses.size else 0.0,
            "median_response": float(np.median(responses)) if responses.size else 0.0,
            "min_response": float(np.min(responses)) if responses.size else 0.0,
            "max_response": float(np.max(responses)) if responses.size else 0.0,
        },
        "response_summary_by_type": per_type,
        "sampling_footprint_summary": footprint,
        "mosaic_footprint_overlay": {
            "label": "Stimulated receptor footprint overlay",
            "image_kind": "rgb",
            "image_data": _build_mosaic_overlay(
                receptor_rows=receptor_rows,
                receptor_cols=receptor_cols,
                types=types,
                stimulated_mask=stimulated_mask,
                image_shape=input_shape,
            ),
        },
        "retinal_physiology_summary": activation.metadata.get("retinal_physiology", {}),
    }


def build_comparative_renderings(
    activation: Any,
    input_shape: tuple[int, int],
) -> Dict[str, Any]:
    """Build claim-calibrated human-readable rendering artifacts."""
    output_size = _render_output_size(input_shape)
    voronoi = render_voronoi(activation, output_size=output_size)
    reconstructed = render_reconstructed(activation, grid_shape=output_size)
    return {
        "family_label": "comparative renderings",
        "scope_note": (
            "These human-readable outputs are retinal-information renderings of the "
            "modeled retinal front end. They are not direct perceptual, retinal-circuit, "
            "or cortical reconstructions, and they should not be read as what the animal "
            "consciously sees."
        ),
        "items": [
            {
                "id": "comparative_activation_map",
                "label": "Comparative rendering: receptor-coded activation map",
                "description": (
                    "Voronoi rendering of receptor activations. Hue encodes receptor class "
                    "and brightness encodes modeled activation."
                ),
                "image_kind": "rgb",
                "image_data": np.asarray(voronoi, dtype=np.float32),
            },
            {
                "id": "retinal_information_rendering",
                "label": "Retinal-information rendering",
                "description": (
                    "Regular-grid rendering of retained front-end activation information "
                    "after retinal sampling."
                ),
                "image_kind": "grayscale",
                "image_data": np.asarray(reconstructed, dtype=np.float32),
            },
        ],
    }


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value - min_value <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_value) / (max_value - min_value)).astype(np.float32)


def _pixel_bounds(
    rows: np.ndarray,
    cols: np.ndarray,
    stimulated_mask: np.ndarray,
) -> Dict[str, int]:
    if not np.any(stimulated_mask):
        return {"row_min": 0, "row_max": 0, "col_min": 0, "col_max": 0}
    return {
        "row_min": int(np.min(rows[stimulated_mask])),
        "row_max": int(np.max(rows[stimulated_mask])),
        "col_min": int(np.min(cols[stimulated_mask])),
        "col_max": int(np.max(cols[stimulated_mask])),
    }


def _position_bounds_mm(
    positions: np.ndarray,
    stimulated_mask: np.ndarray,
) -> Dict[str, float]:
    active = positions[stimulated_mask] if np.any(stimulated_mask) else positions
    if active.size == 0:
        return {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
        }
    return {
        "x_min": float(np.min(active[:, 0])),
        "x_max": float(np.max(active[:, 0])),
        "y_min": float(np.min(active[:, 1])),
        "y_max": float(np.max(active[:, 1])),
    }


def _render_output_size(input_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = int(input_shape[0]), int(input_shape[1])
    longest = max(height, width, 1)
    scale = min(96 / longest, 1.0)
    return (
        max(16, int(round(height * scale))),
        max(16, int(round(width * scale))),
    )


def _build_mosaic_overlay(
    *,
    receptor_rows: np.ndarray,
    receptor_cols: np.ndarray,
    types: np.ndarray,
    stimulated_mask: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    if height <= 0 or width <= 0:
        return overlay

    for index in np.flatnonzero(stimulated_mask):
        color = np.asarray(
            _TYPE_BASE_COLOR.get(str(types[index]), _FALLBACK_COLOR),
            dtype=np.float32,
        )
        row = int(np.clip(receptor_rows[index], 0, height - 1))
        col = int(np.clip(receptor_cols[index], 0, width - 1))
        overlay[row, col] = color

    return overlay
