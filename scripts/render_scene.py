"""Render an input scene as human, dog, and cat retinal patches.

Usage::

    python scripts/render_scene.py path/to/photo.jpg \
        --species human dog cat \
        --scene-width-m 0.5 \
        --distance-m 2.0 \
        --patch-deg 5 \
        --output reports/render.png

This is a thin wrapper around ``RetinalSimulator.compare_species`` followed by
``output.perceptual.render_perceptual_image``. The output is a side-by-side
comparison panel: input patch preview on the left, then one perceptual
rendering per requested species.

The perceptual mapping is an appearance approximation, not a perceptual model.
See ``retinal_sim/output/perceptual.py`` for what is and is not claimed.

Notes
-----
- Large ``--patch-deg`` values will dramatically slow the optical convolution
  and mosaic generation. The 2-degree validation patch is the default for a
  reason; treat values above ~5 deg as exploratory.
- ``--scene-width-m`` and ``--distance-m`` together determine the angular
  subtense of the input scene. If either angular axis exceeds ``--patch-deg``,
  this helper crops to the central patch so the displayed input matches the
  patch actually simulated by the retina pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from retinal_sim.output.perceptual import render_perceptual_image
from retinal_sim.pipeline import RetinalSimulator
from retinal_sim.spectral.upsampler import (
    SCENE_INPUT_MODES,
    SpectralImage,
    scene_input_metadata,
)

_RGB_PREVIEW_CHANNELS_NM = {"red": 650.0, "green": 530.0, "blue": 420.0}


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to load images. Install it with `pip install Pillow`."
        ) from exc
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_spectral_image(path: Path) -> SpectralImage:
    """Load measured spectral input from ``.npz`` or ``.npy``."""
    payload = np.load(path, allow_pickle=False)

    if isinstance(payload, np.ndarray):
        data = np.asarray(payload, dtype=np.float32)
        wavelengths = None
    else:
        if "data" not in payload:
            raise SystemExit(
                "Measured-spectrum input requires a .npz file with a `data` array "
                "of shape (H, W, N_lambda)."
            )
        data = np.asarray(payload["data"], dtype=np.float32)
        wavelengths = (
            np.asarray(payload["wavelengths"], dtype=np.float64)
            if "wavelengths" in payload
            else None
        )

    if data.ndim != 3:
        raise SystemExit(
            "Measured-spectrum input must have shape (H, W, N_lambda); "
            f"got {data.shape!r}."
        )

    if wavelengths is None:
        from retinal_sim.constants import WAVELENGTHS

        if data.shape[2] != len(WAVELENGTHS):
            raise SystemExit(
                "Measured-spectrum input without explicit wavelengths must use the "
                f"canonical wavelength grid length {len(WAVELENGTHS)}; "
                f"got {data.shape[2]} bands."
            )
        wavelengths = np.asarray(WAVELENGTHS, dtype=np.float64)
    elif wavelengths.shape != (data.shape[2],):
        raise SystemExit(
            "Measured-spectrum wavelengths must match the spectral band axis; "
            f"got wavelengths shape {wavelengths.shape!r} for data {data.shape!r}."
        )

    return SpectralImage(data=data, wavelengths=wavelengths, metadata={})


def _load_input(path: Path, input_mode: str) -> np.ndarray | SpectralImage:
    """Load RGB or measured-spectral input based on the chosen semantics."""
    if input_mode == "measured_spectrum":
        return _load_spectral_image(path)
    return _load_image(path)


def _array_shape(input_data: np.ndarray | SpectralImage) -> tuple[int, int]:
    if isinstance(input_data, SpectralImage):
        return tuple(int(v) for v in input_data.data.shape[:2])
    return tuple(int(v) for v in np.asarray(input_data).shape[:2])


def _crop_ndarray(data: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Return a centered crop using the same fraction on both image axes."""
    height, width = data.shape[:2]
    crop_w = max(1, int(round(width * crop_fraction)))
    crop_h = max(1, int(round(height * crop_fraction)))
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    return data[y0 : y0 + crop_h, x0 : x0 + crop_w, ...]


def _crop_input_to_patch(
    input_data: np.ndarray | SpectralImage,
    *,
    scene_width_m: float,
    distance_m: float,
    patch_deg: float,
) -> tuple[np.ndarray | SpectralImage, dict]:
    """Crop to the simulator's square patch if either angular axis is too large."""
    height, width = _array_shape(input_data)
    scene_height_m = scene_width_m * height / width if width > 0 else scene_width_m
    full_angular_width_deg = 2.0 * np.degrees(np.arctan((scene_width_m / 2.0) / distance_m))
    full_angular_height_deg = 2.0 * np.degrees(np.arctan((scene_height_m / 2.0) / distance_m))
    crop_fraction = min(
        1.0,
        patch_deg / max(full_angular_width_deg, 1e-9),
        patch_deg / max(full_angular_height_deg, 1e-9),
    )

    if crop_fraction < 1.0:
        if isinstance(input_data, SpectralImage):
            cropped_data = _crop_ndarray(input_data.data, crop_fraction)
            cropped_input: np.ndarray | SpectralImage = SpectralImage(
                data=np.asarray(cropped_data, dtype=np.float32),
                wavelengths=np.asarray(input_data.wavelengths, dtype=np.float64).copy(),
                metadata=dict(getattr(input_data, "metadata", {})),
            )
        else:
            cropped_input = _crop_ndarray(np.asarray(input_data), crop_fraction).astype(np.float32)
        cropped_height, cropped_width = _array_shape(cropped_input)
    else:
        cropped_input = input_data
        cropped_height, cropped_width = height, width

    return cropped_input, {
        "cropped": bool(crop_fraction < 1.0),
        "crop_fraction": float(crop_fraction),
        "full_angular_width_deg": float(full_angular_width_deg),
        "full_angular_height_deg": float(full_angular_height_deg),
        "cropped_angular_width_deg": float(full_angular_width_deg * crop_fraction),
        "cropped_angular_height_deg": float(full_angular_height_deg * crop_fraction),
        "full_scene_width_m": float(scene_width_m),
        "full_scene_height_m": float(scene_height_m),
        "cropped_scene_width_m": float(scene_width_m * crop_fraction),
        "cropped_scene_height_m": float(scene_height_m * crop_fraction),
        "input_width_px": int(width),
        "input_height_px": int(height),
        "cropped_width_px": int(cropped_width),
        "cropped_height_px": int(cropped_height),
        "patch_deg": float(patch_deg),
        "distance_m": float(distance_m),
    }


def _spectral_preview(spectral_image: SpectralImage) -> np.ndarray:
    """Return a simple RGB-style preview for measured spectral inputs."""
    data = np.asarray(spectral_image.data, dtype=np.float32)
    wavelengths = np.asarray(spectral_image.wavelengths, dtype=np.float64)
    channels = []
    for target_nm in (
        _RGB_PREVIEW_CHANNELS_NM["red"],
        _RGB_PREVIEW_CHANNELS_NM["green"],
        _RGB_PREVIEW_CHANNELS_NM["blue"],
    ):
        idx = int(np.argmin(np.abs(wavelengths - target_nm)))
        channels.append(data[:, :, idx])
    composite = np.stack(channels, axis=-1)
    safe_max = max(float(np.max(composite)), 1e-8)
    return np.clip(composite / safe_max, 0.0, 1.0).astype(np.float32)


def _input_preview_image(input_data: np.ndarray | SpectralImage) -> np.ndarray:
    if isinstance(input_data, SpectralImage):
        return _spectral_preview(input_data)
    return np.asarray(input_data, dtype=np.float32)


def _save_panel(panel: np.ndarray, path: Path) -> None:
    from PIL import Image

    arr = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a small text label across the top of an image (best-effort)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([(0, 0), (pil.width, 18)], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(pil, dtype=np.float32) / 255.0


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _save_image(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _save_diagnostics_bundle(
    diagnostics_dir: Path,
    *,
    results: dict,
    run_metadata: dict,
) -> None:
    """Persist a lightweight irradiance/activation traceability bundle."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_metadata": run_metadata,
        "species": {},
    }

    for species_name, result in results.items():
        species_dir = diagnostics_dir / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

        irradiance = result.artifacts.get("retinal_irradiance_diagnostics", {})
        activation = result.artifacts.get("photoreceptor_activation_diagnostics", {})
        summary_payload = {
            "species": species_name,
            "scene_input_mode": result.metadata.get("scene_input_mode"),
            "scene_input_assumptions": result.metadata.get("scene_input_assumptions", []),
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
        _save_json(summary_payload, summary_path)

        overlay_path = species_dir / "activation_overlay.png"
        overlay_image = activation.get("mosaic_footprint_overlay", {}).get("image_data")
        if overlay_image is not None:
            _save_image(np.asarray(overlay_image, dtype=np.float32), overlay_path)

        band_composite_path = species_dir / "irradiance_band_composite.png"
        band_composite = irradiance.get("band_composite", {}).get("image_data")
        if band_composite is not None:
            _save_image(np.asarray(band_composite, dtype=np.float32), band_composite_path)

        manifest["species"][species_name] = {
            "summary_json": str(summary_path.relative_to(diagnostics_dir)),
            "activation_overlay_png": (
                str(overlay_path.relative_to(diagnostics_dir))
                if overlay_image is not None else None
            ),
            "irradiance_band_composite_png": (
                str(band_composite_path.relative_to(diagnostics_dir))
                if band_composite is not None else None
            ),
        }

    _save_json(manifest, diagnostics_dir / "manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render an image as human/dog/cat retinas would receive it.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help=(
            "Input file. Use a PIL-readable RGB image for RGB modes, or a .npz/.npy "
            "spectral cube for measured_spectrum."
        ),
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=["human", "dog", "cat"],
        choices=["human", "dog", "cat"],
    )
    parser.add_argument(
        "--scene-width-m",
        type=float,
        default=0.5,
        help="Physical width of the scene in metres.",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        default=2.0,
        help="Viewing distance in metres.",
    )
    parser.add_argument(
        "--patch-deg",
        type=float,
        default=2.0,
        help="Simulated retinal patch extent in degrees.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stimulus-scale", type=float, default=0.01)
    parser.add_argument(
        "--input-mode",
        choices=sorted(SCENE_INPUT_MODES),
        default="reflectance_under_d65",
        help="Declared scene-spectrum semantics passed through to RetinalSimulator.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Optional directory for per-species irradiance/activation artifacts.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/render_scene.png"))
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Input image not found: {args.image}", file=sys.stderr)
        return 2

    input_data = _load_input(args.image, args.input_mode)
    input_h, input_w = _array_shape(input_data)
    print(f"Loaded {args.image} ({input_w}x{input_h})")

    cropped_input, crop_info = _crop_input_to_patch(
        input_data,
        scene_width_m=args.scene_width_m,
        distance_m=args.distance_m,
        patch_deg=args.patch_deg,
    )
    scene_meta = scene_input_metadata(args.input_mode)
    sim_scene_width_m = crop_info["cropped_scene_width_m"]
    preview = _input_preview_image(cropped_input)
    preview_h, preview_w = preview.shape[:2]

    print(
        "Scene semantics: "
        f"{scene_meta['scene_input_mode']} "
        f"({'RGB-inferred' if scene_meta['scene_input_is_inferred'] else 'measured spectrum'})"
    )
    for assumption in scene_meta["scene_input_assumptions"]:
        print(f"  - {assumption}")
    print(
        "Patch geometry: "
        f"full scene {crop_info['full_angular_width_deg']:.2f}x"
        f"{crop_info['full_angular_height_deg']:.2f} deg, "
        f"simulated patch {crop_info['cropped_angular_width_deg']:.2f}x"
        f"{crop_info['cropped_angular_height_deg']:.2f} deg."
    )
    if crop_info["cropped"]:
        print(
            "Cropping input to central "
            f"{crop_info['cropped_width_px']}x{crop_info['cropped_height_px']} "
            f"(scene_width_m -> {sim_scene_width_m:.4f})."
        )
    else:
        print("Input already fits inside the simulated patch; no crop applied.")

    sim = RetinalSimulator(
        args.species[0],
        patch_extent_deg=args.patch_deg,
        stimulus_scale=args.stimulus_scale,
        seed=args.seed,
    )
    print(f"Running pipeline for: {', '.join(args.species)}")
    results = sim.compare_species(
        cropped_input,
        species_list=args.species,
        scene_width_m=sim_scene_width_m,
        viewing_distance_m=args.distance_m,
        input_mode=args.input_mode,
    )

    input_label = "input (patch)"
    if args.input_mode == "measured_spectrum":
        input_label = "input spectrum (patch)"
    panels = [_label(preview, input_label)]
    for species_name in args.species:
        rendered = render_perceptual_image(results[species_name], grid_shape=(preview_h, preview_w))
        panels.append(_label(rendered, species_name))

    panel = np.concatenate(panels, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_panel(panel, args.output)
    print(f"Saved comparison to {args.output}")

    if args.diagnostics_dir is not None:
        run_metadata = {
            "input_path": str(args.image),
            "output_path": str(args.output),
            "species": list(args.species),
            "scene_input_mode": scene_meta["scene_input_mode"],
            "scene_input_is_inferred": scene_meta["scene_input_is_inferred"],
            "scene_input_assumptions": scene_meta["scene_input_assumptions"],
            "patch_geometry": crop_info,
        }
        _save_diagnostics_bundle(
            args.diagnostics_dir,
            results=results,
            run_metadata=run_metadata,
        )
        print(f"Saved diagnostics to {args.diagnostics_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
