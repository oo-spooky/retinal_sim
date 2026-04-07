"""Render an input image as it would appear to human, dog, and cat retinas.

Usage::

    python scripts/render_scene.py path/to/photo.jpg \\
        --species human dog cat \\
        --scene-width-m 0.5 \\
        --distance-m 2.0 \\
        --patch-deg 5 \\
        --output reports/render.png

This is a thin wrapper around ``RetinalSimulator.compare_species`` followed by
``output.perceptual.render_perceptual_image``. The output is a side-by-side
comparison panel: original image on the left, then one perceptual rendering
per requested species. The perceptual mapping is an *appearance approximation*,
not a perceptual model — see ``retinal_sim/output/perceptual.py`` for what is
and is not claimed.

Notes
-----
- Large ``--patch-deg`` values will dramatically slow the optical convolution
  and mosaic generation. The 2-degree validation patch is the default for a
  reason; treat values above ~5° as exploratory.
- ``--scene-width-m`` and ``--distance-m`` together determine the angular
  subtense of the input image. If the image subtends more than ``--patch-deg``,
  the central patch is what gets simulated; the surrounding pixels do not
  reach the retina in the simulation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from retinal_sim.output.perceptual import render_perceptual_image
from retinal_sim.pipeline import RetinalSimulator


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to load images. Install it with `pip install Pillow`."
        ) from exc
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render an image as human/dog/cat retinas would receive it.",
    )
    parser.add_argument("image", type=Path, help="Input image (any PIL-readable format).")
    parser.add_argument(
        "--species",
        nargs="+",
        default=["human", "dog", "cat"],
        choices=["human", "dog", "cat"],
    )
    parser.add_argument("--scene-width-m", type=float, default=0.5,
                        help="Physical width of the scene in metres.")
    parser.add_argument("--distance-m", type=float, default=2.0,
                        help="Viewing distance in metres.")
    parser.add_argument("--patch-deg", type=float, default=2.0,
                        help="Simulated retinal patch extent in degrees.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stimulus-scale", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("reports/render_scene.png"))
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Input image not found: {args.image}", file=sys.stderr)
        return 2

    rgb = _load_image(args.image)
    H, W, _ = rgb.shape
    print(f"Loaded {args.image} ({W}x{H})")

    # Crop the input to the simulated patch so the displayed panel matches what
    # the retina actually sees. Without this the simulator only models the
    # central ``patch_deg`` of the scene but the rendered cone activations are
    # then stretched across the full original image, which makes high-acuity
    # species (human) look blurrier than they should and low-density species
    # look like coarse Voronoi tiles. Cropping first keeps the panel
    # apples-to-apples.
    full_angular_w_deg = 2.0 * np.degrees(
        np.arctan((args.scene_width_m / 2.0) / args.distance_m)
    )
    crop_frac = min(1.0, args.patch_deg / full_angular_w_deg)
    if crop_frac < 1.0:
        crop_w = max(1, int(round(W * crop_frac)))
        crop_h = max(1, int(round(H * crop_frac)))
        x0 = (W - crop_w) // 2
        y0 = (H - crop_h) // 2
        rgb = rgb[y0:y0 + crop_h, x0:x0 + crop_w, :]
        H, W, _ = rgb.shape
        sim_scene_width_m = args.scene_width_m * crop_frac
        print(
            f"Scene subtends {full_angular_w_deg:.2f}° but patch is "
            f"{args.patch_deg:.2f}°; cropping input to central "
            f"{crop_w}x{crop_h} (scene_width_m -> {sim_scene_width_m:.4f})."
        )
    else:
        sim_scene_width_m = args.scene_width_m

    sim = RetinalSimulator(
        args.species[0],
        patch_extent_deg=args.patch_deg,
        stimulus_scale=args.stimulus_scale,
        seed=args.seed,
    )
    print(f"Running pipeline for: {', '.join(args.species)}")
    results = sim.compare_species(
        rgb,
        species_list=args.species,
        scene_width_m=sim_scene_width_m,
        viewing_distance_m=args.distance_m,
        input_mode="reflectance_under_d65",
    )

    panels = [_label(rgb, "input (patch)")]
    for sp in args.species:
        rendered = render_perceptual_image(results[sp], grid_shape=(H, W))
        panels.append(_label(rendered, sp))

    panel = np.concatenate(panels, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_panel(panel, args.output)
    print(f"Saved comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
