"""Helpers for writing explorer run bundles and diagnostics artifacts."""
from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any, Optional

import numpy as np

from retinal_sim.output.diagnostics import assert_json_safe_roundtrip

_COMPARATIVE_ITEM_FILENAMES = {
    "comparative_activation_map": "activation_map.png",
    "retinal_information_rendering": "retinal_information_rendering.png",
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
    """Persist a lightweight irradiance/activation traceability bundle."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_metadata": run_metadata,
        "species": {},
    }
    species_extras = species_extras or {}

    for species_name, result in results.items():
        species_dir = diagnostics_dir / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

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
            "retinal_irradiance_summary": {
                "traceability_note": irradiance.get("traceability_note"),
                "delivered_spectrum_summary": irradiance.get("delivered_spectrum_summary", {}),
                "optical_stage_summary": irradiance.get("optical_stage_summary", {}),
            },
            "photoreceptor_activation_summary": {
                "traceability_note": activation.get("traceability_note"),
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
    crop_info: dict,
    scene_meta: dict,
) -> dict[str, Any]:
    species_dir = species_root / species_name
    species_dir.mkdir(parents=True, exist_ok=True)

    irradiance = result.artifacts.get("retinal_irradiance_diagnostics", {})
    activation = result.artifacts.get("photoreceptor_activation_diagnostics", {})
    comparative = result.artifacts.get("comparative_renderings", {})

    relative_input_patch = _relative_posix(species_dir, bundle_dir / "input_patch.png")
    saved_assets = {
        "input_patch_png": relative_input_patch,
        "irradiance_band_composite_png": None,
        "irradiance_slice_pngs": {},
        "activation_overlay_png": None,
        "activation_map_png": None,
        "retinal_information_rendering_png": None,
    }

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

    summary_payload = _build_species_summary(
        species_name=species_name,
        result=result,
        crop_info=crop_info,
        scene_meta=scene_meta,
        saved_assets=saved_assets,
        input_patch_relative_path=relative_input_patch,
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
        "activation_map_png": _relative_posix(bundle_dir / "diagnostics", species_dir / "activation_map.png"),
        "retinal_information_rendering_png": _relative_posix(
            bundle_dir / "diagnostics",
            species_dir / "retinal_information_rendering.png",
        ),
        "irradiance_slice_pngs": {
            key: _relative_posix(bundle_dir / "diagnostics", species_dir / value)
            for key, value in saved_assets["irradiance_slice_pngs"].items()
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
        },
        "report_html": f"species/{species_name}/index.html",
        "report_summary_json": f"species/{species_name}/summary.json",
        "report_assets": {
            "input_patch_png": "input_patch.png",
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
) -> dict[str, Any]:
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

    return {
        "species": species_name,
        "title": f"{species_name} retinal-front-end report",
        "scope_note": (
            "This page traces the modeled retinal front end from scene assumptions to "
            "retinally delivered irradiance, receptor sampling, and retinal-information "
            "renderings. It is not a direct perceptual, retinal-circuit, or cortical report."
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
            "diagnostics_summary_json": f"../../diagnostics/{species_name}/summary.json",
            "diagnostics_manifest_json": "../../diagnostics/manifest.json",
        },
        "scene_and_patch": {
            "input_patch_png": input_patch_relative_path,
            "crop_info": crop_info,
            "scene_input_mode": scene_meta["scene_input_mode"],
            "scene_input_is_inferred": scene_meta["scene_input_is_inferred"],
            "scene_input_assumptions": scene_meta["scene_input_assumptions"],
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
            "items": retinal_information_items,
        },
    }


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
            "This bundle summarizes a retinal-front-end comparison. The outputs expose "
            "traceable retinal irradiance diagnostics, photoreceptor activation "
            "diagnostics, and retinal-information renderings. They are not direct "
            "conscious-perception, retinal-circuit, or cortical reconstructions."
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
        diagnostics = payload.get("diagnostics", {})
        species_cards.append(
            "<div class=\"species-card\">"
            f"<h3>{escape(species_name.title())}</h3>"
            "<p>This standalone report walks through the modeled stage story for this species.</p>"
            f"<p><a class=\"cta\" href=\"{escape(payload.get('report_html', '#'))}\">Open {escape(species_name)} report</a></p>"
            "<p class=\"secondary-links\">"
            f"<a href=\"{escape(diagnostics.get('summary_json') or '#')}\">diagnostics summary</a> | "
            f"<a href=\"{escape(summary['paths']['diagnostics_manifest_json'])}\">manifest</a>"
            "</p>"
            "</div>"
        )

    species_rows = ""
    for species_name, payload in summary["species_summaries"].items():
        diagnostics = payload.get("diagnostics", {})
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
            f"<td><a href=\"{escape(diagnostics.get('summary_json') or '#')}\">summary</a></td>"
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
<h1>retinal_sim explorer run</h1>
<div class="panel note"><strong>Scope note:</strong> {escape(summary["scope_note"])}</div>

<h2>Overview</h2>
<div class="panel split">
  <div>
    <p><strong>Scene input mode:</strong> <code>{escape(summary["scene_input_mode"])}</code></p>
    <p><strong>Species:</strong> {escape(", ".join(summary["species"]))}</p>
    <p><strong>Patch geometry:</strong> full scene {crop["full_angular_width_deg"]:.2f} x {crop["full_angular_height_deg"]:.2f} deg, simulated patch {crop["cropped_angular_width_deg"]:.2f} x {crop["cropped_angular_height_deg"]:.2f} deg.</p>
    <p><strong>Cropping applied:</strong> {"yes" if crop["cropped"] else "no"}</p>
    <p><strong>Native input patch:</strong> {summary["native_input_patch_px"][1]} x {summary["native_input_patch_px"][0]} px</p>
    <p><strong>Activation render grid:</strong> {summary["activation_render_px"][1]} x {summary["activation_render_px"][0]} px</p>
  </div>
  <div><img src="{escape(summary["paths"]["input_patch_png"])}" alt="Input patch preview"></div>
</div>

<h2>Scene assumptions</h2>
<div class="panel">{assumptions}</div>

<h2>Open species reports</h2>
<div class="species-grid">{"".join(species_cards)}</div>

<h2>At a glance comparison</h2>
<div class="panel">
  <p>The comparison strip remains an overview. The primary experience is the per-species report set above.</p>
  <p><img src="{escape(summary["paths"]["comparison_png"])}" alt="Retinal comparison panel"></p>
</div>

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
      <th>Diagnostics</th>
    </tr>
  </thead>
  <tbody>{species_rows}</tbody>
</table>

<h2>Traceability bundle</h2>
<div class="panel">
  <p><a href="{escape(summary["paths"]["diagnostics_manifest_json"])}">Diagnostics manifest</a></p>
  <p>The diagnostics directory remains the machine-readable family. The per-species pages turn those same stages into a user-facing walkthrough.</p>
</div>
</body>
</html>"""


def _render_species_html(summary: dict) -> str:
    scene = summary["scene_and_patch"]
    irradiance = summary["retinal_irradiance"]
    sampling = summary["sampling_and_activation"]
    outputs = summary["retinal_information_outputs"]

    slice_gallery = "".join(
        (
            "<div class=\"gallery-item\">"
            f"<p class=\"output-item-title\">{escape(item['label'])}</p>"
            f"<img src=\"{escape(item['image_path'])}\" alt=\"{escape(item['label'])}\">"
            f"<p>Target {item['target_wavelength_nm']:.0f} nm, sampled {item['sampled_wavelength_nm']:.0f} nm</p>"
            "</div>"
        )
        for item in irradiance.get("selected_wavelength_slices", [])
        if item.get("image_path")
    )
    info_gallery = "".join(
        (
            "<div class=\"gallery-item\">"
            f"<p class=\"output-item-title\">{escape(item['label'])}</p>"
            f"<img src=\"{escape(item['image_path'])}\" alt=\"{escape(item['label'])}\">"
            f"<p>{escape(str(item.get('description', '')))}</p>"
            "</div>"
        )
        for item in outputs.get("items", [])
        if item.get("image_path")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(summary["title"])}</title>
  <style>{_BUNDLE_STYLE}</style>
</head>
<body>
<p><a href="{escape(summary["paths"]["overview_html"])}">Back to overview</a></p>
<h1>{escape(summary["title"])}</h1>
<div class="panel note"><strong>Scope note:</strong> {escape(summary["scope_note"])}</div>

<h2>Scene and patch</h2>
<div class="panel split">
  <div>
    <p><strong>Scene input mode:</strong> <code>{escape(scene["scene_input_mode"])}</code></p>
    <p><strong>Spectrum source:</strong> {"RGB-inferred" if summary["scene_input_is_inferred"] else "measured spectrum"}</p>
    <p><strong>Patch geometry:</strong> full scene {summary["patch_geometry"]["full_angular_width_deg"]:.2f} x {summary["patch_geometry"]["full_angular_height_deg"]:.2f} deg, simulated patch {summary["patch_geometry"]["cropped_angular_width_deg"]:.2f} x {summary["patch_geometry"]["cropped_angular_height_deg"]:.2f} deg.</p>
    {_render_list(scene.get("scene_input_assumptions", []))}
  </div>
  <div><img src="{escape(scene["input_patch_png"])}" alt="Input patch"></div>
</div>

<h2>Retinal irradiance</h2>
<div class="panel">
  <p>{escape(str(irradiance.get("traceability_note", "")))}</p>
  <div class="split">
    <div>
      <p class="output-item-title">Retinal irradiance band composite</p>
      <img src="{escape(str(irradiance.get("band_composite_png", "")))}" alt="Retinal irradiance band composite">
    </div>
    <div>
      <p class="output-item-title">Optical stage summary</p>
      <pre>{escape(json.dumps(irradiance.get("optical_stage_summary", {}), indent=2))}</pre>
    </div>
  </div>
  <div class="gallery">{slice_gallery}</div>
</div>

<h2>Sampling and activation</h2>
<div class="panel">
  <p>{escape(str(sampling.get("traceability_note", "")))}</p>
  <div class="split">
    <div>
      <p class="output-item-title">Stimulated receptor footprint overlay</p>
      <img src="{escape(str(sampling.get("activation_overlay_png", "")))}" alt="Stimulated receptor footprint overlay">
    </div>
    <div>
      <p class="output-item-title">Sampling footprint summary</p>
      <pre>{escape(json.dumps(sampling.get("sampling_footprint_summary", {}), indent=2))}</pre>
    </div>
  </div>
  <div class="split">
    <div>
      <p class="output-item-title">Overall activation summary</p>
      <pre>{escape(json.dumps(sampling.get("overall_summary", {}), indent=2))}</pre>
    </div>
    <div>
      <p class="output-item-title">Retinal physiology scope</p>
      <pre>{escape(json.dumps(sampling.get("retinal_physiology_summary", {}), indent=2))}</pre>
    </div>
  </div>
</div>

<h2>Retinal-information outputs</h2>
<div class="panel">
  <p>{escape(str(outputs.get("scope_note", "")))}</p>
  <div class="gallery">{info_gallery}</div>
</div>

<div class="panel">
  <p><a href="{escape(summary["paths"]["diagnostics_summary_json"])}">Diagnostics summary JSON</a></p>
  <p><a href="{escape(summary["paths"]["diagnostics_manifest_json"])}">Diagnostics manifest</a></p>
</div>
</body>
</html>"""


_BUNDLE_STYLE = """
body { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #222; line-height: 1.5; }
h1, h2, h3 { border-bottom: 1px solid #ddd; padding-bottom: 6px; }
.panel { background: #fafafa; border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px; margin: 14px 0; }
.note { background: #eef6ff; border-color: #c7dbf4; }
.split { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; align-items: start; }
.species-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
.species-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px; }
.cta { display: inline-block; padding: 8px 12px; border-radius: 999px; background: #173b67; color: #fff; text-decoration: none; font-weight: 600; }
.secondary-links { font-size: 0.92em; color: #555; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-top: 14px; }
.gallery-item { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; }
.output-item-title { font-weight: 600; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: left; font-size: 0.94em; }
th { background: #f4f4f4; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }
code { background: #f6f6f6; padding: 0 4px; border-radius: 3px; }
pre { white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 12px; }
a { color: #173b67; }
"""
