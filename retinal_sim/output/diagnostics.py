"""Structured diagnostic builders for Phase R6 traceable output families."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

from retinal_sim.output.reconstruction import render_reconstructed
from retinal_sim.output.voronoi import _TYPE_BASE_COLOR, _FALLBACK_COLOR, render_voronoi

_SELECTED_WAVELENGTHS_NM = (420.0, 530.0, 650.0)
_PLOT_COLORS_RGB = (
    (42, 91, 153),
    (205, 79, 57),
    (62, 138, 86),
)


def json_safe_artifact_value(value: Any) -> Any:
    """Recursively convert artifact payloads into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe_artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_artifact_value(item) for item in value]
    return value


def assert_json_safe_roundtrip(payload: Any) -> Any:
    """Return a JSON-roundtripped copy of an artifact payload.

    This makes the ndarray/list conversion contract explicit at the point where
    diagnostic-family payloads are persisted or tested.
    """
    return json.loads(json.dumps(json_safe_artifact_value(payload)))


def build_spectral_interpretation_diagnostics(
    spectral_image: Any,
    native_input_shape: Optional[tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Build deterministic spectral-interpretation summaries from pipeline state."""
    data = np.asarray(spectral_image.data, dtype=np.float32)
    wavelengths = np.asarray(spectral_image.wavelengths, dtype=np.float64)
    source_mean = data.mean(axis=(0, 1))
    metadata = dict(getattr(spectral_image, "metadata", {}))

    return {
        "family_label": "spectral interpretation diagnostics",
        "family_version": "r6_species_report_v1",
        "traceability_note": (
            "This family records how the input patch was interpreted as a spectral "
            "stimulus before species-specific optics or retinal sampling were applied."
        ),
        "native_input_patch_px": (
            [int(native_input_shape[0]), int(native_input_shape[1])]
            if native_input_shape is not None else None
        ),
        "input_mode_summary": {
            "scene_input_mode": metadata.get("scene_input_mode"),
            "scene_input_is_inferred": metadata.get("scene_input_is_inferred"),
            "scene_input_assumptions": metadata.get("scene_input_assumptions", []),
        },
        "source_spectrum_summary": {
            "wavelengths_nm": wavelengths.astype(float).tolist(),
            "source_mean_by_wavelength": source_mean.astype(float).tolist(),
            "peak_source_wavelength_nm": float(wavelengths[int(np.argmax(source_mean))]),
            "source_total_energy": float(np.sum(source_mean)),
        },
        "spectral_band_composite": {
            "id": "spectral_band_composite",
            "label": "Spectral interpretation RGB-style band composite",
            "image_kind": "rgb",
            "image_data": _band_composite_from_cube(data, wavelengths),
        },
        "source_mean_spectrum_plot": {
            "id": "source_mean_spectrum_plot",
            "label": "Mean source spectrum across the simulated patch",
            "image_kind": "rgb",
            "image_data": _plot_series_image(
                wavelengths,
                [("source", source_mean)],
                title="Source spectrum",
            ),
        },
    }


def build_optical_delivery_diagnostics(
    spectral_image: Any,
    retinal_irradiance: Any,
) -> Dict[str, Any]:
    """Build deterministic optical-delivery summaries from pipeline state."""
    spectral_data = np.asarray(spectral_image.data, dtype=np.float32)
    irradiance_data = np.asarray(retinal_irradiance.data, dtype=np.float32)
    wavelengths = np.asarray(retinal_irradiance.wavelengths, dtype=np.float64)
    metadata = dict(getattr(retinal_irradiance, "metadata", {}))

    source_mean = spectral_data.mean(axis=(0, 1))
    delivered_mean = irradiance_data.mean(axis=(0, 1))
    sigma_x = np.asarray(metadata.get("psf_sigma_px_x", []), dtype=np.float64)
    sigma_y = np.asarray(metadata.get("psf_sigma_px_y", []), dtype=np.float64)
    representative_kernel = np.asarray(
        metadata.get("representative_psf_kernel", np.zeros((1, 1), dtype=np.float32)),
        dtype=np.float32,
    )
    blur_reference_wavelength_nm = float(
        metadata.get("representative_psf_wavelength_nm", metadata.get("lca_reference_wavelength_nm", 555.0))
    )

    blur_min_wavelength_nm = None
    if sigma_x.size:
        sigma_mean = sigma_x if sigma_y.size == 0 else 0.5 * (sigma_x + sigma_y)
        blur_min_wavelength_nm = float(wavelengths[int(np.argmin(sigma_mean))])

    return {
        "family_label": "optical delivery diagnostics",
        "family_version": "r6_species_report_v1",
        "traceability_note": (
            "This family records how species-specific optics altered the inferred scene "
            "spectrum before photoreceptor sampling, including throughput, blur, and media filtering."
        ),
        "optical_delivery_summary": {
            "pupil_throughput_scale": metadata.get("pupil_throughput_scale"),
            "anisotropy_active": metadata.get("anisotropy_active"),
            "effective_f_number": metadata.get("effective_f_number"),
            "effective_f_number_x": metadata.get("effective_f_number_x"),
            "effective_f_number_y": metadata.get("effective_f_number_y"),
            "blur_min_wavelength_nm": blur_min_wavelength_nm,
            "reference_wavelength_nm": blur_reference_wavelength_nm,
            "media_transmission_source": metadata.get("media_transmission_source"),
        },
        "delivered_spectrum_plot": {
            "id": "delivered_spectrum_plot",
            "label": "Source versus delivered mean spectrum",
            "image_kind": "rgb",
            "image_data": _plot_series_image(
                wavelengths,
                [("source", source_mean), ("delivered", delivered_mean)],
                title="Optical delivery",
            ),
        },
        "psf_sigma_plot": {
            "id": "psf_sigma_plot",
            "label": "PSF sigma by wavelength",
            "image_kind": "rgb",
            "image_data": _plot_series_image(
                wavelengths,
                [("sigma_x", sigma_x), ("sigma_y", sigma_y)],
                title="PSF sigma (px)",
            ) if sigma_x.size and sigma_y.size else np.zeros((180, 320, 3), dtype=np.float32),
        },
        "representative_psf_kernel": {
            "id": "representative_psf_kernel",
            "label": f"Representative PSF kernel ({int(round(blur_reference_wavelength_nm))} nm)",
            "image_kind": "grayscale",
            "reference_wavelength_nm": blur_reference_wavelength_nm,
            "image_data": _normalize_image(np.sqrt(np.clip(representative_kernel, 0.0, None))),
        },
    }


def build_retinal_irradiance_diagnostics(
    spectral_image: Any,
    retinal_irradiance: Any,
    native_input_shape: Optional[tuple[int, int]] = None,
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
                "id": f"irradiance_slice_{int(target_nm)}nm",
                "label": f"Retinal irradiance slice ({int(target_nm)} nm)",
                "target_wavelength_nm": float(target_nm),
                "sampled_wavelength_nm": float(wavelengths[idx]),
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
        "family_version": "r6_run_bundle_v1",
        "traceability_note": (
            "This family records where scene-spectrum assumptions and anterior-eye "
            "optical effects altered the retinally delivered light before receptor sampling."
        ),
        "native_input_patch_px": (
            [int(native_input_shape[0]), int(native_input_shape[1])]
            if native_input_shape is not None else None
        ),
        "irradiance_native_px": [int(data.shape[0]), int(data.shape[1])],
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
            "id": "irradiance_band_composite",
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
    overlay_rows: Optional[np.ndarray] = None,
    overlay_cols: Optional[np.ndarray] = None,
    overlay_shape: Optional[tuple[int, int]] = None,
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

    overlay_rows = receptor_rows if overlay_rows is None else overlay_rows
    overlay_cols = receptor_cols if overlay_cols is None else overlay_cols
    overlay_shape = input_shape if overlay_shape is None else overlay_shape

    footprint = {
        "input_shape": [int(input_shape[0]), int(input_shape[1])],
        "native_input_patch_px": [int(input_shape[0]), int(input_shape[1])],
        "activation_render_px": [int(overlay_shape[0]), int(overlay_shape[1])],
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
        "family_version": "r6_run_bundle_v1",
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
            "id": "stimulated_receptor_footprint_overlay",
            "label": "Stimulated receptor footprint overlay",
            "image_kind": "rgb",
            "image_data": _build_mosaic_overlay(
                receptor_rows=overlay_rows,
                receptor_cols=overlay_cols,
                types=types,
                stimulated_mask=stimulated_mask,
                image_shape=overlay_shape,
            ),
        },
        "retinal_physiology_summary": activation.metadata.get("retinal_physiology", {}),
    }


def build_comparative_renderings(
    activation: Any,
    input_shape: tuple[int, int],
    output_shape: Optional[tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Build claim-calibrated human-readable rendering artifacts."""
    output_size = output_shape if output_shape is not None else _render_output_size(input_shape)
    voronoi = render_voronoi(activation, output_size=output_size)
    reconstructed = render_reconstructed(activation, grid_shape=output_size)
    return {
        "family_label": "comparative renderings",
        "family_version": "r6_run_bundle_v1",
        "scope_note": (
            "These human-readable outputs are retinal-information renderings of the "
            "modeled retinal front end. They are not direct perceptual, retinal-circuit, "
            "or cortical reconstructions, and they should not be read as what the animal "
            "consciously sees."
        ),
        "native_input_patch_px": [int(input_shape[0]), int(input_shape[1])],
        "activation_render_px": [int(output_size[0]), int(output_size[1])],
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


def _band_composite_from_cube(data: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Return a simple RGB-style composite from representative wavelength bands."""
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    data = np.asarray(data, dtype=np.float32)
    channels = []
    for target_nm in (650.0, 530.0, 420.0):
        idx = int(np.argmin(np.abs(wavelengths - target_nm)))
        channels.append(data[:, :, idx])
    composite = np.stack(channels, axis=-1)
    safe_max = max(float(np.max(composite)), 1e-8)
    return np.clip(composite / safe_max, 0.0, 1.0).astype(np.float32)


def _plot_series_image(
    wavelengths: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    title: str,
    width: int = 320,
    height: int = 180,
) -> np.ndarray:
    """Render a lightweight line-plot image for bundle diagnostics."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return np.zeros((height, width, 3), dtype=np.float32)

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    canvas = Image.new("RGB", (width, height), color=(250, 250, 247))
    draw = ImageDraw.Draw(canvas)

    plot_left = 42
    plot_right = width - 12
    plot_top = 24
    plot_bottom = height - 28
    plot_width = max(plot_right - plot_left, 1)
    plot_height = max(plot_bottom - plot_top, 1)

    draw.rectangle(
        [(plot_left, plot_top), (plot_right, plot_bottom)],
        outline=(170, 170, 170),
        fill=(255, 255, 255),
    )
    draw.text((12, 6), title, fill=(34, 34, 34))
    draw.text((plot_left, height - 20), f"{int(round(wavelengths[0]))} nm", fill=(85, 85, 85))
    draw.text((plot_right - 42, height - 20), f"{int(round(wavelengths[-1]))} nm", fill=(85, 85, 85))

    cleaned_series: list[tuple[str, np.ndarray]] = []
    max_value = 0.0
    for label, values in series:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size != wavelengths.size:
            continue
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned_series.append((label, arr))
        max_value = max(max_value, float(np.max(arr)) if arr.size else 0.0)

    max_value = max(max_value, 1e-8)
    for row in range(5):
        y = plot_top + int(round(row * plot_height / 4.0))
        draw.line([(plot_left, y), (plot_right, y)], fill=(235, 235, 235), width=1)

    wavelength_min = float(wavelengths[0])
    wavelength_span = max(float(wavelengths[-1] - wavelengths[0]), 1e-8)

    for index, (label, values) in enumerate(cleaned_series):
        color = _PLOT_COLORS_RGB[index % len(_PLOT_COLORS_RGB)]
        points = []
        for x_nm, y_value in zip(wavelengths, values):
            x = plot_left + int(round(((float(x_nm) - wavelength_min) / wavelength_span) * plot_width))
            y = plot_bottom - int(round((float(y_value) / max_value) * plot_height))
            points.append((x, int(np.clip(y, plot_top, plot_bottom))))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        legend_x = plot_left + index * 94
        draw.line([(legend_x, 16), (legend_x + 18, 16)], fill=color, width=3)
        draw.text((legend_x + 24, 9), label.replace("_", " "), fill=(44, 44, 44))

    return np.asarray(canvas, dtype=np.float32) / 255.0


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
