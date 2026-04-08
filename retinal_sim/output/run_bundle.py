"""Helpers for writing explorer run bundles and diagnostics artifacts."""
from __future__ import annotations

import json
import os
import textwrap
from html import escape
from pathlib import Path
from typing import Any, Optional

import numpy as np

from retinal_sim.output.diagnostics import assert_json_safe_roundtrip

_COMPARATIVE_ITEM_FILENAMES = {
    "comparative_activation_map": "activation_map.png",
    "retinal_information_rendering": "retinal_information_rendering.png",
}
_PLOT_FILENAMES = {
    "spectral_band_composite": "spectral_band_composite.png",
    "source_mean_spectrum_plot": "source_mean_spectrum_plot.png",
    "delivered_spectrum_plot": "delivered_spectrum_plot.png",
    "psf_sigma_plot": "psf_sigma_plot.png",
    "representative_psf_kernel": "representative_psf_kernel.png",
}


def save_panel(panel: np.ndarray, path: Path) -> None:
    """Persist an RGB panel image."""
    save_image(panel, path)


def save_image(image: np.ndarray, path: Path) -> None:
    """Persist an RGB or grayscale image."""
    from PIL import Image

    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_json(payload: dict, path: Path) -> dict:
    """Persist JSON after enforcing a JSON-safe roundtrip."""
    safe_payload = assert_json_safe_roundtrip(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    return safe_payload


def save_diagnostics_bundle(
    diagnostics_dir: Path,
    *,
    results: dict,
    run_metadata: dict,
    species_extras: Optional[dict[str, dict[str, Any]]] = None,
) -> dict:
    """Persist a lightweight machine-readable diagnostics bundle."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_metadata": run_metadata,
        "species": {},
    }
    species_extras = species_extras or {}

    for species_name, result in results.items():
        species_dir = diagnostics_dir / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

        spectral = result.artifacts.get("spectral_interpretation_diagnostics", {})
        optical = result.artifacts.get("optical_delivery_diagnostics", {})
        irradiance = result.artifacts.get("retinal_irradiance_diagnostics", {})
        activation = result.artifacts.get("photoreceptor_activation_diagnostics", {})
        comparative = result.artifacts.get("comparative_renderings", {})
        activation_render_px = result.artifacts.get("artifact_render_shape")
        if activation_render_px is None:
            activation_render_px = comparative.get("activation_render_px")
        summary_payload = {
            "species": species_name,
            "scene_input_mode": result.metadata.get("scene_input_mode"),
            "scene_input_assumptions": result.metadata.get("scene_input_assumptions", []),
            "native_input_patch_px": result.artifacts.get("input_shape"),
            "activation_render_px": activation_render_px,
            "irradiance_native_px": list(
                np.asarray(
                    result.artifacts.get("retinal_irradiance_diagnostics", {}).get(
                        "irradiance_native_px", []
                    ),
                    dtype=int,
                ).tolist()
            ),
            "diagnostic_families": {
                "spectral_interpretation_diagnostics": {
                    "family_label": spectral.get("family_label"),
                    "family_version": spectral.get("family_version"),
                    "traceability_note": spectral.get("traceability_note"),
                },
                "optical_delivery_diagnostics": {
                    "family_label": optical.get("family_label"),
                    "family_version": optical.get("family_version"),
                    "traceability_note": optical.get("traceability_note"),
                },
                "retinal_irradiance_diagnostics": {
                    "family_label": irradiance.get("family_label"),
                    "family_version": irradiance.get("family_version"),
                    "traceability_note": irradiance.get("traceability_note"),
                },
                "photoreceptor_activation_diagnostics": {
                    "family_label": activation.get("family_label"),
                    "family_version": activation.get("family_version"),
                    "traceability_note": activation.get("traceability_note"),
                },
                "comparative_renderings": {
                    "family_label": comparative.get("family_label"),
                    "family_version": comparative.get("family_version"),
                    "scope_note": comparative.get("scope_note"),
                },
            },
            "spectral_interpretation_summary": {
                "input_mode_summary": spectral.get("input_mode_summary", {}),
                "source_spectrum_summary": spectral.get("source_spectrum_summary", {}),
            },
            "optical_delivery_summary": {
                "optical_delivery_summary": optical.get("optical_delivery_summary", {}),
            },
            "retinal_irradiance_summary": {
                "delivered_spectrum_summary": irradiance.get("delivered_spectrum_summary", {}),
                "optical_stage_summary": irradiance.get("optical_stage_summary", {}),
            },
            "photoreceptor_activation_summary": {
                "overall_summary": activation.get("overall_summary", {}),
                "sampling_footprint_summary": activation.get("sampling_footprint_summary", {}),
                "retinal_physiology_summary": activation.get("retinal_physiology_summary", {}),
            },
        }

        summary_path = species_dir / "summary.json"
        save_json(summary_payload, summary_path)

        overlay_path = species_dir / "activation_overlay.png"
        overlay_image = activation.get("mosaic_footprint_overlay", {}).get("image_data")
        if overlay_image is not None:
            save_image(np.asarray(overlay_image, dtype=np.float32), overlay_path)

        band_composite_path = species_dir / "irradiance_band_composite.png"
        band_composite = irradiance.get("band_composite", {}).get("image_data")
        if band_composite is not None:
            save_image(np.asarray(band_composite, dtype=np.float32), band_composite_path)

        manifest_entry = {
            "summary_json": summary_path.relative_to(diagnostics_dir).as_posix(),
            "activation_overlay_png": (
                overlay_path.relative_to(diagnostics_dir).as_posix()
                if overlay_image is not None
                else None
            ),
            "irradiance_band_composite_png": (
                band_composite_path.relative_to(diagnostics_dir).as_posix()
                if band_composite is not None
                else None
            ),
        }
        manifest_entry.update(species_extras.get(species_name, {}))
        manifest["species"][species_name] = manifest_entry

    return save_json(manifest, diagnostics_dir / "manifest.json")


def _prefix_relative_path(prefix: str, value: str | None) -> str | None:
    if value is None:
        return None
    return (Path(prefix) / value).as_posix()


def _relative_posix(from_dir: Path, to_path: Path) -> str:
    return Path(os.path.relpath(to_path, from_dir)).as_posix()


def _format_metric_html(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_metric_html(item) for item in value)
    return escape(str(value))


def _render_list(items: list[str]) -> str:
    if not items:
        return "<ul><li>No additional assumptions recorded.</li></ul>"
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"


def write_run_bundle(
    bundle_dir: Path,
    *,
    input_preview: np.ndarray,
    comparison_panel: np.ndarray,
    results: dict,
    input_path: Path,
    species_list: list[str],
    scene_meta: dict,
    crop_info: dict,
    native_input_shape: tuple[int, int],
    activation_render_shape: tuple[int, int],
    perceptual_renders: dict[str, np.ndarray],
    primary_output_path: str | None = None,
    extra_diagnostics_dir: str | None = None,
) -> dict:
    """Write the end-user-facing run bundle."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = bundle_dir / "comparison.png"
    input_patch_path = bundle_dir / "input_patch.png"
    save_panel(comparison_panel, comparison_path)
    save_image(input_preview, input_patch_path)

    species_root = bundle_dir / "species"
    species_root.mkdir(parents=True, exist_ok=True)

    species_summaries: dict[str, dict[str, Any]] = {}
    diagnostics_extras: dict[str, dict[str, Any]] = {}
    for species_name in species_list:
        species_summary = _write_species_report(
            bundle_dir=bundle_dir,
            species_root=species_root,
            species_name=species_name,
            result=results[species_name],
            input_preview=input_preview,
            perceptual_render=perceptual_renders[species_name],
            crop_info=crop_info,
            scene_meta=scene_meta,
        )
        species_summaries[species_name] = species_summary["overview_summary"]
        diagnostics_extras[species_name] = species_summary["diagnostics_manifest_entry"]

    diagnostics_dir = bundle_dir / "diagnostics"
    diagnostics_manifest = save_diagnostics_bundle(
        diagnostics_dir,
        results=results,
        run_metadata={
            "input_path": str(input_path),
            "species": list(species_list),
            "scene_input_mode": scene_meta["scene_input_mode"],
            "scene_input_is_inferred": scene_meta["scene_input_is_inferred"],
            "scene_input_assumptions": scene_meta["scene_input_assumptions"],
            "patch_geometry": crop_info,
            "output_path": str(comparison_path),
            "activation_render_px": list(activation_render_shape),
            "native_input_patch_px": [int(native_input_shape[0]), int(native_input_shape[1])],
        },
        species_extras=diagnostics_extras,
    )

    summary_payload = _build_run_summary(
        input_path=input_path,
        species_list=species_list,
        scene_meta=scene_meta,
        crop_info=crop_info,
        native_input_shape=native_input_shape,
        activation_render_shape=activation_render_shape,
        primary_output_path=primary_output_path,
        extra_diagnostics_dir=extra_diagnostics_dir,
        diagnostics_manifest=diagnostics_manifest,
        species_summaries=species_summaries,
    )
    save_json(summary_payload, bundle_dir / "summary.json")
    (bundle_dir / "index.html").write_text(
        _render_index_html(summary_payload),
        encoding="utf-8",
    )
    return summary_payload


def _write_species_report(
    *,
    bundle_dir: Path,
    species_root: Path,
    species_name: str,
    result: Any,
    input_preview: np.ndarray,
    perceptual_render: np.ndarray,
    crop_info: dict,
    scene_meta: dict,
) -> dict[str, Any]:
    species_dir = species_root / species_name
    species_dir.mkdir(parents=True, exist_ok=True)

    spectral = result.artifacts.get("spectral_interpretation_diagnostics", {})
    optical = result.artifacts.get("optical_delivery_diagnostics", {})
    irradiance = result.artifacts.get("retinal_irradiance_diagnostics", {})
    activation = result.artifacts.get("photoreceptor_activation_diagnostics", {})
    comparative = result.artifacts.get("comparative_renderings", {})

    relative_input_patch = _relative_posix(species_dir, bundle_dir / "input_patch.png")
    saved_assets = {
        "input_patch_png": relative_input_patch,
        "spectral_band_composite_png": None,
        "source_mean_spectrum_plot_png": None,
        "delivered_spectrum_plot_png": None,
        "psf_sigma_plot_png": None,
        "representative_psf_kernel_png": None,
        "irradiance_band_composite_png": None,
        "irradiance_slice_pngs": {},
        "activation_overlay_png": None,
        "perceptual_appearance_png": None,
        "hero_two_up_png": None,
        "story_plate_png": None,
        "activation_map_png": None,
        "retinal_information_rendering_png": None,
    }

    for family_key, filename in _PLOT_FILENAMES.items():
        if family_key in spectral:
            image_data = spectral.get(family_key, {}).get("image_data")
        else:
            image_data = optical.get(family_key, {}).get("image_data")
        if image_data is None:
            continue
        image_path = species_dir / filename
        save_image(np.asarray(image_data, dtype=np.float32), image_path)
        saved_assets[f"{family_key}_png"] = image_path.name

    band_composite = irradiance.get("band_composite", {}).get("image_data")
    if band_composite is not None:
        band_path = species_dir / "irradiance_band_composite.png"
        save_image(np.asarray(band_composite, dtype=np.float32), band_path)
        saved_assets["irradiance_band_composite_png"] = band_path.name

    for slice_item in irradiance.get("selected_wavelength_slices", []):
        image_data = slice_item.get("image_data")
        if image_data is None:
            continue
        filename = f"{slice_item['id']}.png"
        slice_path = species_dir / filename
        save_image(np.asarray(image_data, dtype=np.float32), slice_path)
        saved_assets["irradiance_slice_pngs"][slice_item["id"]] = filename

    overlay = activation.get("mosaic_footprint_overlay", {}).get("image_data")
    if overlay is not None:
        overlay_path = species_dir / "activation_overlay.png"
        save_image(np.asarray(overlay, dtype=np.float32), overlay_path)
        saved_assets["activation_overlay_png"] = overlay_path.name

    perceptual_path = species_dir / "perceptual_appearance_approximation.png"
    save_image(np.asarray(perceptual_render, dtype=np.float32), perceptual_path)
    saved_assets["perceptual_appearance_png"] = perceptual_path.name

    retinal_information_image = None
    for item in comparative.get("items", []):
        image_data = item.get("image_data")
        filename = _COMPARATIVE_ITEM_FILENAMES.get(item.get("id", ""))
        if image_data is None or filename is None:
            continue
        item_path = species_dir / filename
        save_image(np.asarray(image_data, dtype=np.float32), item_path)
        if item.get("id") == "comparative_activation_map":
            saved_assets["activation_map_png"] = item_path.name
        elif item.get("id") == "retinal_information_rendering":
            saved_assets["retinal_information_rendering_png"] = item_path.name
            retinal_information_image = np.asarray(image_data, dtype=np.float32)

    if retinal_information_image is None:
        retinal_information_image = np.zeros_like(np.asarray(perceptual_render, dtype=np.float32)[..., 0])

    hero_image = _compose_hero_two_up(
        species_name=species_name,
        perceptual_render=np.asarray(perceptual_render, dtype=np.float32),
        retinal_information_rendering=retinal_information_image,
    )
    hero_path = species_dir / "hero_two_up.png"
    save_image(hero_image, hero_path)
    saved_assets["hero_two_up_png"] = hero_path.name

    optical_detail_panel = _compose_optical_detail_panel(
        delivered_spectrum_plot=optical.get("delivered_spectrum_plot", {}).get("image_data"),
        psf_sigma_plot=optical.get("psf_sigma_plot", {}).get("image_data"),
        representative_psf_kernel=optical.get("representative_psf_kernel", {}).get("image_data"),
    )
    story_takeaways = _build_story_takeaways(
        species_name=species_name,
        scene_meta=scene_meta,
        optical=optical,
        activation=activation,
        comparative=comparative,
    )
    story_plate = _compose_story_plate(
        species_name=species_name,
        scene_input_mode=scene_meta["scene_input_mode"],
        crop_info=crop_info,
        input_patch=np.asarray(input_preview, dtype=np.float32),
        spectral_band_composite=spectral.get("spectral_band_composite", {}).get("image_data"),
        irradiance_band_composite=band_composite,
        optical_detail_panel=optical_detail_panel,
        activation_overlay=overlay,
        hero_two_up=hero_image,
        takeaways=story_takeaways,
    )
    story_plate_path = species_dir / "story_plate.png"
    save_image(story_plate, story_plate_path)
    saved_assets["story_plate_png"] = story_plate_path.name

    summary_payload = _build_species_summary(
        species_name=species_name,
        result=result,
        crop_info=crop_info,
        scene_meta=scene_meta,
        saved_assets=saved_assets,
        input_patch_relative_path=relative_input_patch,
        story_takeaways=story_takeaways,
    )
    save_json(summary_payload, species_dir / "summary.json")
    (species_dir / "index.html").write_text(
        _render_species_html(summary_payload),
        encoding="utf-8",
    )

    diagnostics_manifest_entry = {
        "species_report_html": _relative_posix(bundle_dir / "diagnostics", species_dir / "index.html"),
        "species_report_summary_json": _relative_posix(bundle_dir / "diagnostics", species_dir / "summary.json"),
        "input_patch_png": _relative_posix(bundle_dir / "diagnostics", bundle_dir / "input_patch.png"),
        "hero_two_up_png": _relative_posix(bundle_dir / "diagnostics", species_dir / "hero_two_up.png"),
        "story_plate_png": _relative_posix(bundle_dir / "diagnostics", species_dir / "story_plate.png"),
        "activation_map_png": _relative_posix(bundle_dir / "diagnostics", species_dir / "activation_map.png"),
        "retinal_information_rendering_png": _relative_posix(
            bundle_dir / "diagnostics",
            species_dir / "retinal_information_rendering.png",
        ),
        "irradiance_slice_pngs": {
            key: _relative_posix(bundle_dir / "diagnostics", species_dir / value)
            for key, value in saved_assets["irradiance_slice_pngs"].items()
        },
        "spectral_stage_assets": {
            "spectral_band_composite_png": _relative_posix(
                bundle_dir / "diagnostics",
                species_dir / saved_assets["spectral_band_composite_png"],
            ) if saved_assets["spectral_band_composite_png"] is not None else None,
            "source_mean_spectrum_plot_png": _relative_posix(
                bundle_dir / "diagnostics",
                species_dir / saved_assets["source_mean_spectrum_plot_png"],
            ) if saved_assets["source_mean_spectrum_plot_png"] is not None else None,
        },
        "optical_stage_assets": {
            "delivered_spectrum_plot_png": _relative_posix(
                bundle_dir / "diagnostics",
                species_dir / saved_assets["delivered_spectrum_plot_png"],
            ) if saved_assets["delivered_spectrum_plot_png"] is not None else None,
            "psf_sigma_plot_png": _relative_posix(
                bundle_dir / "diagnostics",
                species_dir / saved_assets["psf_sigma_plot_png"],
            ) if saved_assets["psf_sigma_plot_png"] is not None else None,
            "representative_psf_kernel_png": _relative_posix(
                bundle_dir / "diagnostics",
                species_dir / saved_assets["representative_psf_kernel_png"],
            ) if saved_assets["representative_psf_kernel_png"] is not None else None,
        },
    }
    if saved_assets["irradiance_band_composite_png"] is not None:
        diagnostics_manifest_entry["species_irradiance_band_composite_png"] = _relative_posix(
            bundle_dir / "diagnostics",
            species_dir / saved_assets["irradiance_band_composite_png"],
        )
    if saved_assets["activation_overlay_png"] is not None:
        diagnostics_manifest_entry["species_activation_overlay_png"] = _relative_posix(
            bundle_dir / "diagnostics",
            species_dir / saved_assets["activation_overlay_png"],
        )

    overview_summary = {
        "species": species_name,
        "native_input_patch_px": result.artifacts.get("input_shape"),
        "activation_render_px": result.artifacts.get("artifact_render_shape")
        or comparative.get("activation_render_px"),
        "irradiance_native_px": irradiance.get("irradiance_native_px"),
        "stimulated_receptor_count": result.summary_metrics.get(
            "stimulated_receptor_count",
            activation.get("sampling_footprint_summary", {}).get("stimulated_receptor_count"),
        ),
        "stimulated_receptor_fraction": result.summary_metrics.get(
            "stimulated_receptor_fraction",
            activation.get("sampling_footprint_summary", {}).get("stimulated_receptor_fraction"),
        ),
        "mean_response": result.summary_metrics.get(
            "mean_response",
            activation.get("overall_summary", {}).get("mean_response"),
        ),
        "left_right_cone_discriminability": result.summary_metrics.get("left_right_cone_discriminability"),
        "center_contrast": result.summary_metrics.get("center_contrast"),
        "periphery_contrast": result.summary_metrics.get("periphery_contrast"),
        "retinal_model_scope": activation.get("retinal_physiology_summary", {}).get("model_scope"),
        "diagnostics": {
            "summary_json": _prefix_relative_path("diagnostics", f"{species_name}/summary.json"),
            "activation_overlay_png": _prefix_relative_path(
                "diagnostics",
                f"{species_name}/activation_overlay.png" if overlay is not None else None,
            ),
            "irradiance_band_composite_png": _prefix_relative_path(
                "diagnostics",
                f"{species_name}/irradiance_band_composite.png" if band_composite is not None else None,
            ),
            "manifest_entry": f"diagnostics/manifest.json#species/{species_name}",
        },
        "report_html": f"species/{species_name}/index.html",
        "report_summary_json": f"species/{species_name}/summary.json",
        "report_assets": {
            "input_patch_png": "input_patch.png",
            "spectral_band_composite_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["spectral_band_composite_png"],
            ),
            "source_mean_spectrum_plot_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["source_mean_spectrum_plot_png"],
            ),
            "delivered_spectrum_plot_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["delivered_spectrum_plot_png"],
            ),
            "psf_sigma_plot_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["psf_sigma_plot_png"],
            ),
            "representative_psf_kernel_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["representative_psf_kernel_png"],
            ),
            "irradiance_band_composite_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["irradiance_band_composite_png"],
            ),
            "irradiance_slice_pngs": {
                key: _prefix_relative_path(f"species/{species_name}", value)
                for key, value in saved_assets["irradiance_slice_pngs"].items()
            },
            "activation_overlay_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["activation_overlay_png"],
            ),
            "perceptual_appearance_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["perceptual_appearance_png"],
            ),
            "hero_two_up_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["hero_two_up_png"],
            ),
            "story_plate_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["story_plate_png"],
            ),
            "activation_map_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["activation_map_png"],
            ),
            "retinal_information_rendering_png": _prefix_relative_path(
                f"species/{species_name}",
                saved_assets["retinal_information_rendering_png"],
            ),
        },
    }

    return {
        "overview_summary": overview_summary,
        "diagnostics_manifest_entry": diagnostics_manifest_entry,
    }


def _build_species_summary(
    *,
    species_name: str,
    result: Any,
    crop_info: dict,
    scene_meta: dict,
    saved_assets: dict[str, Any],
    input_patch_relative_path: str,
    story_takeaways: list[str],
) -> dict[str, Any]:
    spectral = result.artifacts.get("spectral_interpretation_diagnostics", {})
    optical = result.artifacts.get("optical_delivery_diagnostics", {})
    irradiance = result.artifacts.get("retinal_irradiance_diagnostics", {})
    activation = result.artifacts.get("photoreceptor_activation_diagnostics", {})
    comparative = result.artifacts.get("comparative_renderings", {})

    retinal_information_items = []
    for item in comparative.get("items", []):
        filename = _COMPARATIVE_ITEM_FILENAMES.get(item.get("id", ""))
        if filename is None:
            continue
        retinal_information_items.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "description": item.get("description"),
                "image_path": filename,
            }
        )

    selected_slices = []
    for slice_item in irradiance.get("selected_wavelength_slices", []):
        selected_slices.append(
            {
                "id": slice_item.get("id"),
                "label": slice_item.get("label"),
                "target_wavelength_nm": slice_item.get("target_wavelength_nm"),
                "sampled_wavelength_nm": slice_item.get("sampled_wavelength_nm"),
                "mean": slice_item.get("mean"),
                "min": slice_item.get("min"),
                "max": slice_item.get("max"),
                "image_path": saved_assets["irradiance_slice_pngs"].get(slice_item.get("id")),
            }
        )

    stage_story = _build_stage_story(
        species_name=species_name,
        scene_meta=scene_meta,
        crop_info=crop_info,
        spectral=spectral,
        optical=optical,
        irradiance=irradiance,
        activation=activation,
        comparative=comparative,
        saved_assets=saved_assets,
        story_takeaways=story_takeaways,
    )

    return {
        "species": species_name,
        "title": f"{species_name} retinal-front-end report",
        "scope_note": (
            "This page traces the modeled retinal front end from scene assumptions to "
            "spectral interpretation, optical delivery, receptor sampling, and final "
            "retinal-information outputs. It is not a direct perceptual, retinal-circuit, "
            "or cortical report."
        ),
        "scene_input_mode": scene_meta["scene_input_mode"],
        "scene_input_is_inferred": scene_meta["scene_input_is_inferred"],
        "scene_input_assumptions": scene_meta["scene_input_assumptions"],
        "patch_geometry": crop_info,
        "native_input_patch_px": result.artifacts.get("input_shape"),
        "activation_render_px": result.artifacts.get("artifact_render_shape")
        or comparative.get("activation_render_px"),
        "irradiance_native_px": irradiance.get("irradiance_native_px"),
        "paths": {
            "overview_html": "../../index.html",
            "summary_json": "summary.json",
            "input_patch_png": input_patch_relative_path,
            "hero_two_up_png": "hero_two_up.png",
            "story_plate_png": "story_plate.png",
            "diagnostics_summary_json": f"../../diagnostics/{species_name}/summary.json",
            "diagnostics_manifest_json": "../../diagnostics/manifest.json",
        },
        "hero": {
            "hero_two_up_png": saved_assets["hero_two_up_png"],
            "perceptual_appearance_png": saved_assets["perceptual_appearance_png"],
            "retinal_information_rendering_png": saved_assets["retinal_information_rendering_png"],
            "appearance_caption": "Appearance approximation",
            "retinal_caption": "Retained retinal-front-end information",
        },
        "story_plate": {
            "story_plate_png": saved_assets["story_plate_png"],
            "takeaways": story_takeaways,
        },
        "diagnostic_families": {
            "spectral_interpretation_diagnostics": {
                "family_label": spectral.get("family_label"),
                "family_version": spectral.get("family_version"),
                "traceability_note": spectral.get("traceability_note"),
            },
            "optical_delivery_diagnostics": {
                "family_label": optical.get("family_label"),
                "family_version": optical.get("family_version"),
                "traceability_note": optical.get("traceability_note"),
            },
            "retinal_irradiance_diagnostics": {
                "family_label": irradiance.get("family_label"),
                "family_version": irradiance.get("family_version"),
                "traceability_note": irradiance.get("traceability_note"),
            },
            "photoreceptor_activation_diagnostics": {
                "family_label": activation.get("family_label"),
                "family_version": activation.get("family_version"),
                "traceability_note": activation.get("traceability_note"),
            },
            "comparative_renderings": {
                "family_label": comparative.get("family_label"),
                "family_version": comparative.get("family_version"),
                "scope_note": comparative.get("scope_note"),
            },
        },
        "scene_and_patch": {
            "input_patch_png": input_patch_relative_path,
            "crop_info": crop_info,
            "scene_input_mode": scene_meta["scene_input_mode"],
            "scene_input_is_inferred": scene_meta["scene_input_is_inferred"],
            "scene_input_assumptions": scene_meta["scene_input_assumptions"],
        },
        "spectral_interpretation": {
            "traceability_note": spectral.get("traceability_note"),
            "spectral_band_composite_png": saved_assets["spectral_band_composite_png"],
            "source_mean_spectrum_plot_png": saved_assets["source_mean_spectrum_plot_png"],
            "input_mode_summary": spectral.get("input_mode_summary", {}),
            "source_spectrum_summary": spectral.get("source_spectrum_summary", {}),
        },
        "optical_delivery": {
            "traceability_note": optical.get("traceability_note"),
            "delivered_spectrum_plot_png": saved_assets["delivered_spectrum_plot_png"],
            "psf_sigma_plot_png": saved_assets["psf_sigma_plot_png"],
            "representative_psf_kernel_png": saved_assets["representative_psf_kernel_png"],
            "optical_delivery_summary": optical.get("optical_delivery_summary", {}),
        },
        "retinal_irradiance": {
            "traceability_note": irradiance.get("traceability_note"),
            "band_composite_png": saved_assets["irradiance_band_composite_png"],
            "selected_wavelength_slices": selected_slices,
            "optical_stage_summary": irradiance.get("optical_stage_summary", {}),
            "delivered_spectrum_summary": irradiance.get("delivered_spectrum_summary", {}),
        },
        "sampling_and_activation": {
            "traceability_note": activation.get("traceability_note"),
            "activation_overlay_png": saved_assets["activation_overlay_png"],
            "overall_summary": activation.get("overall_summary", {}),
            "sampling_footprint_summary": activation.get("sampling_footprint_summary", {}),
            "response_summary_by_type": activation.get("response_summary_by_type", []),
            "retinal_physiology_summary": activation.get("retinal_physiology_summary", {}),
        },
        "retinal_information_outputs": {
            "scope_note": comparative.get("scope_note"),
            "activation_map_png": saved_assets["activation_map_png"],
            "retinal_information_rendering_png": saved_assets["retinal_information_rendering_png"],
            "perceptual_appearance_png": saved_assets["perceptual_appearance_png"],
            "hero_two_up_png": saved_assets["hero_two_up_png"],
            "items": retinal_information_items,
        },
        "stage_story": stage_story,
    }


def _build_stage_story(
    *,
    species_name: str,
    scene_meta: dict,
    crop_info: dict,
    spectral: dict,
    optical: dict,
    irradiance: dict,
    activation: dict,
    comparative: dict,
    saved_assets: dict[str, Any],
    story_takeaways: list[str],
) -> list[dict[str, Any]]:
    source_summary = spectral.get("source_spectrum_summary", {})
    optical_summary = optical.get("optical_delivery_summary", {})
    sampling_summary = activation.get("sampling_footprint_summary", {})
    response_summary = activation.get("overall_summary", {})
    retina_scope = activation.get("retinal_physiology_summary", {})

    return [
        {
            "id": "scene_patch",
            "title": "Scene and patch",
            "what_happens": (
                "The input scene is cropped to the simulated central patch so every downstream "
                "artifact refers to the same field of view."
            ),
            "species_takeaway": (
                "This step is shared across species in this run, which means later differences "
                f"in the {species_name} report come from the modeled biology rather than a different crop."
            ),
            "final_correlation": (
                "The final composite should be read as different processing of the same patch, "
                "not different source content."
            ),
            "asset_paths": _asset_links([("Input patch", saved_assets["input_patch_png"])]),
            "key_metrics": {
                "Scene input mode": scene_meta["scene_input_mode"],
                "Spectrum source": "RGB-inferred" if scene_meta["scene_input_is_inferred"] else "Measured spectrum",
                "Full scene (deg)": f"{crop_info['full_angular_width_deg']:.2f} x {crop_info['full_angular_height_deg']:.2f}",
                "Simulated patch (deg)": f"{crop_info['cropped_angular_width_deg']:.2f} x {crop_info['cropped_angular_height_deg']:.2f}",
                "Crop applied": "yes" if crop_info["cropped"] else "no",
            },
            "raw_diagnostics_paths": _raw_links(
                [
                    ("Species summary JSON", "summary.json"),
                    ("Diagnostics summary JSON", f"../../diagnostics/{species_name}/summary.json"),
                ]
            ),
            "raw_metadata": {
                "scene_input_assumptions": scene_meta["scene_input_assumptions"],
                "crop_info": crop_info,
            },
        },
        {
            "id": "spectral_interpretation",
            "title": "Spectral interpretation",
            "what_happens": (
                "The cropped patch is interpreted as a spectral stimulus, either by RGB-to-spectrum "
                "inference or by directly accepting a measured spectral cube."
            ),
            "species_takeaway": (
                "This spectral starting point is shared across species for this run. "
                f"The {species_name} page uses it as the common input before optics diverge."
            ),
            "final_correlation": (
                "If the final outputs differ from other species, the divergence begins after this "
                "shared spectral interpretation stage."
            ),
            "asset_paths": _asset_links(
                [
                    ("Spectral band composite", saved_assets["spectral_band_composite_png"]),
                    ("Mean source spectrum", saved_assets["source_mean_spectrum_plot_png"]),
                ]
            ),
            "key_metrics": {
                "Input mode": scene_meta["scene_input_mode"],
                "Inference status": "RGB-inferred spectrum" if scene_meta["scene_input_is_inferred"] else "Measured spectrum",
                "Peak source wavelength (nm)": source_summary.get("peak_source_wavelength_nm"),
                "Total source energy": source_summary.get("source_total_energy"),
            },
            "raw_diagnostics_paths": _raw_links(
                [
                    ("Species summary JSON", "summary.json"),
                    ("Diagnostics manifest", "../../diagnostics/manifest.json"),
                ]
            ),
            "raw_metadata": {
                "traceability_note": spectral.get("traceability_note"),
                "input_mode_summary": spectral.get("input_mode_summary", {}),
                "source_spectrum_summary": source_summary,
            },
        },
        {
            "id": "optical_delivery",
            "title": "Optical delivery",
            "what_happens": (
                "Species-specific optics filter and blur the spectral scene before it reaches the retina. "
                "The delivered spectrum and PSF metadata show where throughput, anisotropy, and blur entered the pipeline."
            ),
            "species_takeaway": _optical_takeaway(species_name, optical_summary),
            "final_correlation": (
                "These optical differences shape the contrast and spectral balance that the receptors sample, "
                "which is why they materially influence the final composite."
            ),
            "asset_paths": _asset_links(
                [
                    ("Delivered spectrum", saved_assets["delivered_spectrum_plot_png"]),
                    ("PSF sigma by wavelength", saved_assets["psf_sigma_plot_png"]),
                    ("Representative PSF kernel", saved_assets["representative_psf_kernel_png"]),
                    ("Retinal irradiance band composite", saved_assets["irradiance_band_composite_png"]),
                    *[
                        (item["label"], saved_assets["irradiance_slice_pngs"].get(item["id"]))
                        for item in irradiance.get("selected_wavelength_slices", [])
                    ],
                ]
            ),
            "key_metrics": {
                "Pupil throughput scale": optical_summary.get("pupil_throughput_scale"),
                "Anisotropy active": optical_summary.get("anisotropy_active"),
                "Effective f-number": optical_summary.get("effective_f_number"),
                "Blur minimum wavelength (nm)": optical_summary.get("blur_min_wavelength_nm"),
                "Media transmission source": optical_summary.get("media_transmission_source"),
            },
            "raw_diagnostics_paths": _raw_links(
                [
                    ("Species summary JSON", "summary.json"),
                    ("Diagnostics summary JSON", f"../../diagnostics/{species_name}/summary.json"),
                ]
            ),
            "raw_metadata": {
                "traceability_note": optical.get("traceability_note"),
                "optical_delivery_summary": optical_summary,
                "optical_stage_summary": irradiance.get("optical_stage_summary", {}),
            },
        },
        {
            "id": "receptor_sampling",
            "title": "Receptor sampling and activation",
            "what_happens": (
                "The delivered retinal image is sampled by the species-specific mosaic and pushed through the "
                "retinal-front-end transduction model, producing receptor activations."
            ),
            "species_takeaway": _sampling_takeaway(species_name, activation),
            "final_correlation": (
                "The final outputs retain this sampling pattern, so receptor density, class mix, and activation "
                "strength directly affect the structure of the composite above."
            ),
            "asset_paths": _asset_links(
                [("Stimulated receptor footprint overlay", saved_assets["activation_overlay_png"])]
            ),
            "key_metrics": {
                "Stimulated receptor count": sampling_summary.get("stimulated_receptor_count"),
                "Stimulated receptor fraction": sampling_summary.get("stimulated_receptor_fraction"),
                "Total receptors": response_summary.get("n_receptors"),
                "Mean response": response_summary.get("mean_response"),
                "Model scope": retina_scope.get("model_scope"),
            },
            "raw_diagnostics_paths": _raw_links(
                [
                    ("Species summary JSON", "summary.json"),
                    ("Diagnostics summary JSON", f"../../diagnostics/{species_name}/summary.json"),
                ]
            ),
            "raw_metadata": {
                "traceability_note": activation.get("traceability_note"),
                "sampling_footprint_summary": sampling_summary,
                "response_summary_by_type": activation.get("response_summary_by_type", []),
                "retinal_physiology_summary": retina_scope,
            },
        },
        {
            "id": "final_outputs",
            "title": "Final outputs",
            "what_happens": (
                "The report ends with two human-readable views: an appearance approximation and a "
                "retinal-information rendering. Both summarize the accumulated effects of the earlier stages."
            ),
            "species_takeaway": story_takeaways[-1],
            "final_correlation": (
                "These outputs integrate the scene assumptions, optical delivery, and receptor sampling shown above. "
                "They remain model-scope-safe summaries of the retinal front end rather than direct perceptual claims."
            ),
            "asset_paths": _asset_links(
                [
                    ("Hero two-up", saved_assets["hero_two_up_png"]),
                    ("Appearance approximation", saved_assets["perceptual_appearance_png"]),
                    ("Activation map", saved_assets["activation_map_png"]),
                    ("Retinal-information rendering", saved_assets["retinal_information_rendering_png"]),
                ]
            ),
            "key_metrics": {
                "Activation render grid": comparative.get("activation_render_px"),
                "Retinal-output scope": "Retinal front end only",
                "Appearance panel": "Appearance approximation",
                "Retinal-information panel": "Retained retinal-front-end information",
            },
            "raw_diagnostics_paths": _raw_links(
                [
                    ("Species summary JSON", "summary.json"),
                    ("Diagnostics manifest", "../../diagnostics/manifest.json"),
                ]
            ),
            "raw_metadata": {
                "scope_note": comparative.get("scope_note"),
                "rendered_items": comparative.get("items", []),
            },
        },
    ]


def _build_run_summary(
    *,
    input_path: Path,
    species_list: list[str],
    scene_meta: dict,
    crop_info: dict,
    native_input_shape: tuple[int, int],
    activation_render_shape: tuple[int, int],
    primary_output_path: str | None,
    extra_diagnostics_dir: str | None,
    diagnostics_manifest: dict,
    species_summaries: dict[str, dict[str, Any]],
) -> dict:
    return {
        "bundle_label": "explorer_run_bundle",
        "input_path": str(input_path),
        "species": list(species_list),
        "scene_input_mode": scene_meta["scene_input_mode"],
        "scene_input_is_inferred": scene_meta["scene_input_is_inferred"],
        "scene_input_assumptions": scene_meta["scene_input_assumptions"],
        "patch_geometry": crop_info,
        "native_input_patch_px": [int(native_input_shape[0]), int(native_input_shape[1])],
        "activation_render_px": [int(activation_render_shape[0]), int(activation_render_shape[1])],
        "irradiance_native_px": [int(native_input_shape[0]), int(native_input_shape[1])],
        "scope_note": (
            "This bundle summarizes a retinal-front-end comparison. The outputs expose traceable "
            "scene assumptions, spectral interpretation, optical delivery, receptor sampling, and "
            "retinal-information renderings. They are not direct conscious-perception, retinal-circuit, "
            "or cortical reconstructions."
        ),
        "paths": {
            "comparison_png": "comparison.png",
            "input_patch_png": "input_patch.png",
            "summary_json": "summary.json",
            "index_html": "index.html",
            "diagnostics_manifest_json": "diagnostics/manifest.json",
            "species_reports_dir": "species",
            "extra_output_path": primary_output_path,
            "extra_diagnostics_dir": extra_diagnostics_dir,
        },
        "species_summaries": species_summaries,
        "diagnostics_manifest": diagnostics_manifest,
    }


def _render_index_html(summary: dict) -> str:
    assumptions = _render_list(summary.get("scene_input_assumptions", []))
    crop = summary["patch_geometry"]

    species_cards = []
    for species_name, payload in summary["species_summaries"].items():
        assets = payload.get("report_assets", {})
        species_cards.append(
            "<div class=\"species-card\">"
            f"<h3>{escape(species_name.title())}</h3>"
            "<p>This standalone walkthrough shows the full five-stage story for this species, from patch assumptions through final outputs.</p>"
            f"<p><a class=\"cta\" href=\"{escape(payload.get('report_html', '#'))}\">Open {escape(species_name)} report</a></p>"
            "<p class=\"secondary-links\">"
            f"<a href=\"{escape(assets.get('story_plate_png') or '#')}\">story plate</a> | "
            f"<a href=\"{escape(assets.get('hero_two_up_png') or '#')}\">hero image</a> | "
            f"<a href=\"{escape(payload.get('report_summary_json') or '#')}\">summary JSON</a>"
            "</p>"
            "</div>"
        )

    species_rows = ""
    for species_name, payload in summary["species_summaries"].items():
        assets = payload.get("report_assets", {})
        species_rows += (
            "<tr>"
            f"<td>{escape(species_name)}</td>"
            f"<td>{_format_metric_html(payload.get('stimulated_receptor_count'))}</td>"
            f"<td>{_format_metric_html(payload.get('stimulated_receptor_fraction'))}</td>"
            f"<td>{_format_metric_html(payload.get('mean_response'))}</td>"
            f"<td>{_format_metric_html(payload.get('left_right_cone_discriminability'))}</td>"
            f"<td>{_format_metric_html(payload.get('center_contrast'))}</td>"
            f"<td>{_format_metric_html(payload.get('periphery_contrast'))}</td>"
            f"<td><a href=\"{escape(payload.get('report_html') or '#')}\">report</a></td>"
            f"<td><a href=\"{escape(assets.get('story_plate_png') or '#')}\">story plate</a></td>"
            f"<td><a href=\"{escape(assets.get('hero_two_up_png') or '#')}\">hero</a></td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>retinal_sim explorer run</title>
  <style>{_BUNDLE_STYLE}</style>
</head>
<body>
<div class="shell">
<header class="masthead">
  <p class="eyebrow">Run bundle</p>
  <h1>retinal_sim explorer run</h1>
  <p class="lead">A landing page for the comparison strip plus guided per-species reports. Use the individual species pages to follow where differences enter the pipeline.</p>
</header>

<div class="callout scope"><strong>Scope note:</strong> {escape(summary["scope_note"])}</div>

<section class="section">
  <h2>Overview</h2>
  <div class="two-up">
    <div class="card">
      <p><strong>Scene input mode:</strong> <code>{escape(summary["scene_input_mode"])}</code></p>
      <p><strong>Species:</strong> {escape(", ".join(summary["species"]))}</p>
      <p><strong>Patch geometry:</strong> full scene {crop["full_angular_width_deg"]:.2f} x {crop["full_angular_height_deg"]:.2f} deg, simulated patch {crop["cropped_angular_width_deg"]:.2f} x {crop["cropped_angular_height_deg"]:.2f} deg.</p>
      <p><strong>Cropping applied:</strong> {"yes" if crop["cropped"] else "no"}</p>
      <p><strong>Native input patch:</strong> {summary["native_input_patch_px"][1]} x {summary["native_input_patch_px"][0]} px</p>
      <p><strong>Activation render grid:</strong> {summary["activation_render_px"][1]} x {summary["activation_render_px"][0]} px</p>
    </div>
    <div class="card image-card"><img src="{escape(summary["paths"]["input_patch_png"])}" alt="Input patch preview"></div>
  </div>
</section>

<section class="section">
  <h2>Scene assumptions</h2>
  <div class="callout">{assumptions}</div>
</section>

<section class="section">
  <h2>Open species reports</h2>
  <div class="species-grid">{"".join(species_cards)}</div>
</section>

<section class="section">
  <h2>At a glance comparison</h2>
  <div class="callout">
    <p>The comparison strip remains an overview. The primary experience is the per-species report set above, where each page breaks the result into scene, spectral, optical, sampling, and final-output stages.</p>
    <p><img src="{escape(summary["paths"]["comparison_png"])}" alt="Retinal comparison panel"></p>
  </div>
</section>

<section class="section">
  <h2>Species summaries</h2>
  <table>
    <thead>
      <tr>
        <th>Species</th>
        <th>Stimulated receptors</th>
        <th>Stimulated fraction</th>
        <th>Mean response</th>
        <th>Left/right discriminability</th>
        <th>Center contrast</th>
        <th>Periphery contrast</th>
        <th>Species report</th>
        <th>Story plate</th>
        <th>Hero image</th>
      </tr>
    </thead>
    <tbody>{species_rows}</tbody>
  </table>
</section>

<section class="section">
  <h2>Traceability bundle</h2>
  <div class="callout">
    <p><a href="{escape(summary["paths"]["diagnostics_manifest_json"])}">Diagnostics manifest</a></p>
    <p>The diagnostics directory remains the machine-readable family. The species pages translate those same families into user-facing walkthroughs without changing the underlying scope claims.</p>
  </div>
</section>
</div>
</body>
</html>"""


def _render_species_html(summary: dict) -> str:
    sections = "".join(_render_stage_section(stage) for stage in summary["stage_story"])
    how_to_read = (
        "Read this page top to bottom. The hero shows the final outputs first, then each stage explains "
        "what the simulator is doing, what is species-specific at that step, and how that difference carries into the final result."
    )
    story_takeaways_html = "".join(
        f"<li>{escape(str(item))}</li>"
        for item in summary.get("story_plate", {}).get("takeaways", [])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(summary["title"])}</title>
  <style>{_BUNDLE_STYLE}</style>
</head>
<body>
<div class="shell species-shell">
<p><a href="{escape(summary["paths"]["overview_html"])}">Back to overview</a></p>
<header class="masthead species-masthead">
  <p class="eyebrow">{escape(summary["species"].title())} report</p>
  <h1>{escape(summary["title"])}</h1>
  <p class="lead">A guided walkthrough of how this species-specific retinal-front-end simulation moves from the shared scene patch to the final report outputs.</p>
</header>

<div class="callout scope"><strong>Scope note:</strong> {escape(summary["scope_note"])}</div>

<section class="section">
  <h2>Hero</h2>
  <div class="hero-card">
    <img src="{escape(summary["paths"]["hero_two_up_png"])}" alt="{escape(summary["species"])} hero two-up">
  </div>
  <div class="callout note">
    <strong>How to read this page:</strong> {escape(how_to_read)}
  </div>
</section>

<section class="section">
  <h2>Story plate</h2>
  <div class="hero-card">
    <img src="{escape(summary["paths"]["story_plate_png"])}" alt="{escape(summary["species"])} story plate">
  </div>
  <div class="callout">
    <p>This one-page PNG is the shareable summary for the {escape(summary["species"])} run.</p>
    <ul>{story_takeaways_html}</ul>
  </div>
</section>

{sections}

<section class="section">
  <h2>Machine-readable links</h2>
  <div class="callout">
    <p><a href="{escape(summary["paths"]["summary_json"])}">Species summary JSON</a></p>
    <p><a href="{escape(summary["paths"]["diagnostics_summary_json"])}">Diagnostics summary JSON</a></p>
    <p><a href="{escape(summary["paths"]["diagnostics_manifest_json"])}">Diagnostics manifest</a></p>
  </div>
</section>
</div>
</body>
</html>"""


def _render_stage_section(stage: dict[str, Any]) -> str:
    assets = stage.get("asset_paths", [])
    asset_html = ""
    if assets:
        asset_html = "<div class=\"gallery\">" + "".join(
            (
                "<div class=\"gallery-item\">"
                f"<p class=\"output-item-title\">{escape(str(asset['label']))}</p>"
                f"<img src=\"{escape(str(asset['path']))}\" alt=\"{escape(str(asset['label']))}\">"
                "</div>"
            )
            for asset in assets
        ) + "</div>"

    raw_links = stage.get("raw_diagnostics_paths", [])
    raw_links_html = "".join(
        f"<li><a href=\"{escape(str(item['path']))}\">{escape(str(item['label']))}</a></li>"
        for item in raw_links
    )
    raw_metadata = assert_json_safe_roundtrip(stage.get("raw_metadata", {}))

    return (
        f"<section class=\"section\">"
        f"<h2>{escape(str(stage['title']))}</h2>"
        "<div class=\"stage-card\">"
        f"<p>{escape(str(stage['what_happens']))}</p>"
        f"<p><strong>Why it matters for this species:</strong> {escape(str(stage['species_takeaway']))}</p>"
        f"<p><strong>How it shows up in the final outputs:</strong> {escape(str(stage['final_correlation']))}</p>"
        f"{_render_metrics_table(stage.get('key_metrics', {}))}"
        f"{asset_html}"
        "<details>"
        "<summary>Raw metadata and JSON links</summary>"
        f"<ul>{raw_links_html}</ul>"
        f"<pre>{escape(json.dumps(raw_metadata, indent=2))}</pre>"
        "</details>"
        "</div>"
        "</section>"
    )


def _render_metrics_table(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "<p>No key metrics recorded for this stage.</p>"
    rows = "".join(
        "<tr>"
        f"<th>{escape(str(label))}</th>"
        f"<td>{_format_metric_html(value)}</td>"
        "</tr>"
        for label, value in metrics.items()
    )
    return f"<table class=\"metrics\"><tbody>{rows}</tbody></table>"


def _asset_links(items: list[tuple[str, str | None]]) -> list[dict[str, str]]:
    return [
        {"label": label, "path": path}
        for label, path in items
        if path
    ]


def _raw_links(items: list[tuple[str, str | None]]) -> list[dict[str, str]]:
    return [
        {"label": label, "path": path}
        for label, path in items
        if path
    ]


def _optical_takeaway(species_name: str, optical_summary: dict[str, Any]) -> str:
    anisotropy = optical_summary.get("anisotropy_active")
    throughput = optical_summary.get("pupil_throughput_scale")
    blur_wavelength = optical_summary.get("blur_min_wavelength_nm")
    if anisotropy:
        return (
            f"{species_name.title()} uses an anisotropic optical model here, so blur is directionally structured "
            "before receptor sampling even begins."
        )
    if throughput is not None and float(throughput) != 1.0:
        return (
            f"{species_name.title()} changes the delivered light level through pupil throughput "
            f"(scale {float(throughput):.3f}), which shifts how much retinal signal reaches the mosaic."
        )
    if blur_wavelength is not None:
        return (
            f"{species_name.title()} reaches its minimum blur near {float(blur_wavelength):.0f} nm in this run, "
            "which helps explain spectral-balance differences later in the report."
        )
    return (
        f"{species_name.title()} applies its own optical delivery model at this stage, even when the starting patch is shared."
    )


def _sampling_takeaway(species_name: str, activation: dict[str, Any]) -> str:
    response_by_type = activation.get("response_summary_by_type", [])
    receptor_types = [item.get("receptor_type") for item in response_by_type if item.get("count")]
    stimulated = activation.get("sampling_footprint_summary", {}).get("stimulated_receptor_count")
    return (
        f"{species_name.title()} samples the delivered image with receptor classes {', '.join(str(item) for item in receptor_types)}; "
        f"{_format_metric_html(stimulated)} receptors were stimulated in this patch."
    )


def _build_story_takeaways(
    *,
    species_name: str,
    scene_meta: dict,
    optical: dict,
    activation: dict,
    comparative: dict,
) -> list[str]:
    source_kind = "RGB-inferred spectrum" if scene_meta["scene_input_is_inferred"] else "measured spectrum"
    optical_summary = optical.get("optical_delivery_summary", {})
    activation_summary = activation.get("sampling_footprint_summary", {})
    optical_line = _optical_takeaway(species_name, optical_summary)
    sampling_line = (
        f"{species_name.title()} retains {activation_summary.get('stimulated_receptor_count', 'n/a')} stimulated receptors in this patch, "
        "and the final report keeps that retinal-front-end scope explicit."
    )
    return [
        f"This run starts from a {source_kind} in {scene_meta['scene_input_mode']}.",
        optical_line,
        sampling_line if comparative else f"{species_name.title()} final outputs remain retinal-front-end summaries.",
    ]


def _as_rgb(image: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) if image is not None else np.zeros((1, 1), dtype=np.float32)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    return arr


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(_as_rgb(image), dtype=np.float32)
    pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    resized = pil.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _compose_titled_tile(
    image: np.ndarray,
    *,
    title: str,
    subtitle: str = "",
    tile_size: tuple[int, int] = (360, 230),
) -> np.ndarray:
    from PIL import Image, ImageDraw

    width, height = tile_size
    canvas = Image.new("RGB", (width, height), color=(255, 250, 244))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=16, outline=(213, 197, 180), fill=(255, 250, 244))

    title_lines = textwrap.wrap(title, width=30)[:2]
    subtitle_lines = textwrap.wrap(subtitle, width=42)[:2] if subtitle else []
    y = 12
    for line in title_lines:
        draw.text((16, y), line, fill=(43, 39, 35))
        y += 16
    for line in subtitle_lines:
        draw.text((16, y), line, fill=(96, 87, 78))
        y += 14

    image_top = max(y + 8, 62)
    image_height = height - image_top - 12
    image_width = width - 24
    resized = _resize_image(image, (image_width, image_height))
    pil_image = Image.fromarray(np.clip(resized * 255.0, 0, 255).astype(np.uint8))
    canvas.paste(pil_image, (12, image_top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _compose_hero_two_up(
    *,
    species_name: str,
    perceptual_render: np.ndarray,
    retinal_information_rendering: np.ndarray,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    left_tile = _compose_titled_tile(
        perceptual_render,
        title="Appearance approximation",
        subtitle="Human-readable approximation only",
        tile_size=(520, 350),
    )
    right_tile = _compose_titled_tile(
        retinal_information_rendering,
        title="Retained retinal-front-end information",
        subtitle="Not a direct perceptual or cortical reconstruction",
        tile_size=(520, 350),
    )

    canvas = Image.new("RGB", (1100, 520), color=(244, 238, 230))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), f"{species_name.title()} final outputs", fill=(30, 29, 28))
    draw.text(
        (28, 42),
        "Left: appearance approximation. Right: retained retinal-front-end information.",
        fill=(89, 79, 68),
    )
    canvas.paste(Image.fromarray(np.clip(left_tile * 255.0, 0, 255).astype(np.uint8)), (28, 88))
    canvas.paste(Image.fromarray(np.clip(right_tile * 255.0, 0, 255).astype(np.uint8)), (552, 88))
    draw.rounded_rectangle((16, 12, 1084, 504), radius=24, outline=(191, 177, 161), width=2)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _compose_optical_detail_panel(
    *,
    delivered_spectrum_plot: np.ndarray | None,
    psf_sigma_plot: np.ndarray | None,
    representative_psf_kernel: np.ndarray | None,
) -> np.ndarray:
    top = _compose_titled_tile(
        _as_rgb(delivered_spectrum_plot),
        title="Delivered spectrum",
        subtitle="Source versus delivered mean spectrum",
        tile_size=(360, 190),
    )
    bottom_left = _compose_titled_tile(
        _as_rgb(psf_sigma_plot),
        title="PSF sigma",
        subtitle="Blur scale by wavelength",
        tile_size=(175, 160),
    )
    bottom_right = _compose_titled_tile(
        _as_rgb(representative_psf_kernel),
        title="Representative PSF",
        subtitle="Reference kernel",
        tile_size=(175, 160),
    )
    top_rgb = _as_rgb(top)
    bottom_left_rgb = _as_rgb(bottom_left)
    bottom_right_rgb = _as_rgb(bottom_right)
    panel = np.ones((370, 360, 3), dtype=np.float32) * np.array([1.0, 0.98, 0.96], dtype=np.float32)
    panel[:190, :, :] = top_rgb
    panel[205:365, :175, :] = bottom_left_rgb
    panel[205:365, 185:360, :] = bottom_right_rgb
    return panel


def _compose_story_plate(
    *,
    species_name: str,
    scene_input_mode: str,
    crop_info: dict,
    input_patch: np.ndarray,
    spectral_band_composite: np.ndarray | None,
    irradiance_band_composite: np.ndarray | None,
    optical_detail_panel: np.ndarray,
    activation_overlay: np.ndarray | None,
    hero_two_up: np.ndarray,
    takeaways: list[str],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    tile_size = (360, 230)
    tiles = [
        _compose_titled_tile(input_patch, title="Scene and patch", subtitle="Shared cropped input", tile_size=tile_size),
        _compose_titled_tile(_as_rgb(spectral_band_composite), title="Spectral interpretation", subtitle="Shared spectral starting point", tile_size=tile_size),
        _compose_titled_tile(_as_rgb(irradiance_band_composite), title="Retinal irradiance", subtitle="Delivered after species optics", tile_size=tile_size),
        _compose_titled_tile(optical_detail_panel, title="Optical delivery", subtitle="Throughput, blur, and PSF detail", tile_size=tile_size),
        _compose_titled_tile(_as_rgb(activation_overlay), title="Receptor sampling", subtitle="Stimulated footprint overlay", tile_size=tile_size),
        _compose_titled_tile(hero_two_up, title="Final outputs", subtitle="Appearance approximation plus retinal-information rendering", tile_size=tile_size),
    ]

    canvas = Image.new("RGB", (1160, 860), color=(247, 243, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), f"{species_name.title()} story plate", fill=(31, 28, 25))
    draw.text(
        (28, 44),
        (
            f"{scene_input_mode} | patch {crop_info['cropped_angular_width_deg']:.2f} x "
            f"{crop_info['cropped_angular_height_deg']:.2f} deg"
        ),
        fill=(89, 79, 68),
    )

    x_positions = (28, 400, 772)
    y_positions = (90, 340)
    for index, tile in enumerate(tiles):
        row = index // 3
        col = index % 3
        pil_tile = Image.fromarray(np.clip(tile * 255.0, 0, 255).astype(np.uint8))
        canvas.paste(pil_tile, (x_positions[col], y_positions[row]))

    footer_top = 598
    draw.rounded_rectangle((24, footer_top, 1136, 836), radius=18, outline=(201, 188, 173), fill=(255, 250, 244))
    draw.text((42, footer_top + 18), "Key takeaways", fill=(41, 38, 34))
    for idx, takeaway in enumerate(takeaways[:3]):
        wrapped = textwrap.wrap(takeaway, width=44)
        y = footer_top + 48 + idx * 58
        draw.text((50, y), f"{idx + 1}. {wrapped[0]}", fill=(83, 73, 63))
        for offset, line in enumerate(wrapped[1:], start=1):
            draw.text((68, y + offset * 14), line, fill=(83, 73, 63))

    draw.rounded_rectangle((16, 12, 1144, 844), radius=26, outline=(191, 177, 161), width=2)
    return np.asarray(canvas, dtype=np.float32) / 255.0


_BUNDLE_STYLE = """
:root {
  --bg: #f4efe8;
  --paper: #fffaf4;
  --panel: #f8f2ea;
  --panel-strong: #efe4d4;
  --line: #d6c6b6;
  --ink: #1f1c19;
  --muted: #62574c;
  --accent: #184e69;
  --accent-soft: #d8ebf5;
}
body { margin: 0; background: radial-gradient(circle at top, #faf6ef 0%, var(--bg) 58%, #ece3d7 100%); color: var(--ink); font-family: "Trebuchet MS", "Segoe UI", sans-serif; line-height: 1.55; }
.shell { max-width: 1160px; margin: 0 auto; padding: 28px 24px 56px; }
.species-shell { max-width: 1180px; }
.masthead { padding: 18px 0 8px; }
.species-masthead { padding-top: 8px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.78rem; color: var(--muted); margin: 0 0 8px; }
h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; color: #211d19; }
h1 { margin: 0 0 10px; font-size: clamp(2.2rem, 4vw, 3rem); }
h2 { margin: 0 0 14px; font-size: 1.55rem; }
h3 { margin: 0 0 8px; font-size: 1.15rem; }
.lead { max-width: 72ch; color: var(--muted); margin: 0; }
.section { margin-top: 28px; }
.callout, .card, .hero-card, .stage-card, .species-card { background: var(--paper); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 8px 22px rgba(79, 57, 33, 0.06); }
.callout, .card, .stage-card, .species-card { padding: 16px 18px; }
.hero-card { padding: 14px; overflow: hidden; }
.scope { background: var(--accent-soft); border-color: #afcddd; }
.note { background: #fff5df; border-color: #e7cf9b; }
.two-up { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; align-items: start; }
.species-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; }
.cta { display: inline-block; padding: 10px 14px; border-radius: 999px; background: var(--accent); color: #fff; text-decoration: none; font-weight: 700; }
.secondary-links { font-size: 0.94rem; color: var(--muted); }
.stage-card p { margin: 0 0 12px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-top: 16px; }
.gallery-item { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px; }
.output-item-title { font-weight: 700; margin: 0 0 8px; color: #2f2922; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; background: var(--paper); border-radius: 14px; overflow: hidden; }
th, td { border: 1px solid var(--line); padding: 9px 10px; text-align: left; font-size: 0.95rem; vertical-align: top; }
th { background: var(--panel-strong); }
.metrics { margin-top: 12px; }
.metrics th { width: 34%; }
img { width: 100%; max-width: 100%; border-radius: 12px; border: 1px solid var(--line); display: block; background: #fff; }
code { background: #efe7dc; padding: 1px 6px; border-radius: 5px; }
pre { white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
a { color: var(--accent); }
details { margin-top: 14px; padding-top: 8px; border-top: 1px solid var(--line); }
summary { cursor: pointer; font-weight: 700; color: #3a342d; }
ul { margin: 10px 0; padding-left: 20px; }
"""
