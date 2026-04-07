"""Dichromat confusion validation via full pipeline simulation (Phase 10).

Algorithm
---------
For a given foreground/background colour pair:

1.  Generate a pseudoisochromatic dot pattern (circular figure on textured
    background) via :func:`~retinal_sim.validation.ishihara.make_dot_pattern`.
2.  Run the image through the full pipeline:
        SpectralUpsampler → OpticalStage → MosaicGenerator → RetinalStage
    using a photoreceptor mosaic sized to the scene patch.
3.  Map each receptor's retinal position back to a pixel coordinate and
    look up whether it falls inside the figure disc.
4.  Compute the figure/ground discriminability metric for the dominant cone
    type::

        D = |mean(r_fig) − mean(r_bg)| / (mean(r_fig) + mean(r_bg) + ε)

    D ≈ 0 → figure indistinguishable from background.
    D → 1 → maximum contrast between figure and background.
5.  Average D over ``n_seeds`` independent mosaic realisations.

Predicted confusion: D < ``threshold`` (default 0.10).

For a confusion pair (returned by
:func:`~retinal_sim.validation.ishihara.find_confusion_pair`):

* A **human** trichromat model should yield D ≫ threshold (sees the figure).
* A **dog** dichromat model should yield D ≪ threshold (cannot see the figure).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from retinal_sim.constants import WAVELENGTHS
from retinal_sim.optical.stage import OpticalStage
from retinal_sim.retina.mosaic import MosaicGenerator
from retinal_sim.retina.stage import MosaicActivation, RetinalStage
from retinal_sim.species.config import SpeciesConfig
from retinal_sim.spectral.upsampler import SpectralUpsampler
from retinal_sim.validation.ishihara import make_dot_pattern
from retinal_sim.validation.datasets import confusion_pairs, control_pairs, as_uint8_pair

# Default discriminability threshold for "confused" (cannot see figure).
_DEFAULT_THRESHOLD = 0.10
# Minimum number of dominant-cone receptors required in each region for a
# reliable discriminability estimate.
_MIN_CONES_PER_REGION = 5


class DichromatValidator:
    """Predict figure/ground discriminability for a species on a dot pattern.

    Args:
        species_name:    One of ``'human'``, ``'dog'``, ``'cat'``.
        seed:            Base random seed for mosaic generation.
        n_seeds:         Number of independent mosaic realisations to average.
        stimulus_scale:  Multiplicative scale applied to spectral irradiance
                         before Naka-Rushton transduction.  The default (0.01)
                         keeps typical excitations near the half-saturation
                         constant (σ ≈ 0.5), preserving the full dynamic range
                         of the nonlinearity.  This is equivalent to viewing
                         the pattern under dim (1 %) illumination, or
                         alternatively correcting for the fact that the
                         SpectralUpsampler outputs unit-reflectance values
                         while the Naka-Rushton σ is calibrated to physical
                         photon-catch units.
    """

    def __init__(
        self,
        species_name: str,
        seed: int = 42,
        n_seeds: int = 2,
        stimulus_scale: float = 0.01,
    ) -> None:
        self._species_name = species_name
        self._base_seed = seed
        self._n_seeds = n_seeds
        self._stimulus_scale = stimulus_scale
        self._cfg = SpeciesConfig.load(species_name)
        self._upsampler = SpectralUpsampler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discriminability(
        self,
        fg_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        patch_size_deg: float = 2.0,
        image_size_px: int = 64,
        n_dots: int = 200,
    ) -> float:
        """Return mean figure/ground discriminability averaged over n_seeds.

        D = |mean_r_fig − mean_r_bg| / (mean_r_fig + mean_r_bg + ε)

        Larger D → figure clearly visible.  D ≈ 0 → figure not detectable.

        Args:
            fg_rgb:        Figure colour, shape (3,) uint8.
            bg_rgb:        Background colour, shape (3,) uint8.
            patch_size_deg: Angular extent of the patch in degrees.
            image_size_px:  Image side length in pixels.
            n_dots:         Number of texture dots per region in the pattern.

        Returns:
            D in [0, 1].
        """
        d_values = [
            self._discriminability_one_seed(
                fg_rgb,
                bg_rgb,
                patch_size_deg,
                image_size_px,
                n_dots,
                self._base_seed + k,
            )
            for k in range(self._n_seeds)
        ]
        return float(np.mean(d_values))

    def is_confused(
        self,
        fg_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        patch_size_deg: float = 2.0,
        image_size_px: int = 64,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> bool:
        """Return True if the species cannot detect the figure (D < threshold).

        Args:
            fg_rgb:        Figure colour, shape (3,) uint8.
            bg_rgb:        Background colour, shape (3,) uint8.
            patch_size_deg: Patch angular size (degrees).
            image_size_px:  Image resolution.
            threshold:      Discriminability below which the species is
                            considered confused.

        Returns:
            True if D < threshold.
        """
        return self.discriminability(fg_rgb, bg_rgb, patch_size_deg, image_size_px) < threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discriminability_one_seed(
        self,
        fg_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        patch_size_deg: float,
        image_size_px: int,
        n_dots: int,
        seed: int,
    ) -> float:
        """Compute D for a single mosaic realisation.

        Returns 0.0 if the mosaic is too sparse (< _MIN_CONES_PER_REGION in
        either the figure or background region).
        """
        # Generate the dot pattern and figure mask.
        image, figure_mask = make_dot_pattern(
            fg_rgb, bg_rgb, image_size_px, n_dots, seed=seed
        )

        # Run the full pipeline on this image.
        activation = self._run_pipeline(image, patch_size_deg, image_size_px, seed)
        if activation is None:
            return 0.0

        # --- Map receptor mm positions → figure/background ----------------
        focal_mm = self._cfg.optical.focal_length_mm
        patch_half_mm = focal_mm * np.tan(np.radians(patch_size_deg / 2.0))
        pixel_scale_mm = (2.0 * patch_half_mm) / image_size_px

        pos = activation.mosaic.positions  # (N, 2) retinal mm
        # Centre of image = (0, 0) mm.
        col_f = pos[:, 0] / pixel_scale_mm + (image_size_px - 1) / 2.0
        row_f = pos[:, 1] / pixel_scale_mm + (image_size_px - 1) / 2.0
        col_i = np.clip(np.round(col_f).astype(int), 0, image_size_px - 1)
        row_i = np.clip(np.round(row_f).astype(int), 0, image_size_px - 1)

        in_figure = figure_mask[row_i, col_i]

        # --- Best-cone-type discriminability ------------------------------
        # For each cone type compute D = |mean_fig - mean_bg| / sum, then
        # return the maximum across types.  Using the BEST cone type is more
        # sensitive than the dominant type alone: the confusion pair differs
        # in human M-cone response (not L-cone), so restricting to the
        # dominant (L) cone would miss the chromatic signal entirely.
        best_d = 0.0
        for ctype in np.unique(activation.mosaic.types):
            if ctype == "rod":
                continue
            type_mask = activation.mosaic.types == ctype
            fig_mask = type_mask & in_figure
            bg_mask = type_mask & ~in_figure
            if fig_mask.sum() < _MIN_CONES_PER_REGION or bg_mask.sum() < _MIN_CONES_PER_REGION:
                continue
            r_fig = float(np.mean(activation.responses[fig_mask]))
            r_bg = float(np.mean(activation.responses[bg_mask]))
            d = abs(r_fig - r_bg) / (r_fig + r_bg + 1e-8)
            if d > best_d:
                best_d = d

        return best_d

    def _run_pipeline(
        self,
        rgb_image: np.ndarray,
        patch_size_deg: float,
        image_size_px: int,
        seed: int,
    ) -> Optional[MosaicActivation]:
        """Run SpectralUpsampler → OpticalStage → RetinalStage on *rgb_image*.

        Args:
            rgb_image:     (H, W, 3) uint8.
            patch_size_deg: Angular size of the patch in degrees.
            image_size_px:  Must match H and W of *rgb_image*.
            seed:           Mosaic random seed.

        Returns:
            MosaicActivation, or None if the patch is too small for a mosaic.
        """
        focal_mm = self._cfg.optical.focal_length_mm
        patch_half_mm = focal_mm * np.tan(np.radians(patch_size_deg / 2.0))
        pixel_scale_mm = (2.0 * patch_half_mm) / image_size_px

        # 1. Spectral upsampling + stimulus scaling.
        # Scale irradiance to keep photoreceptor excitations near the NR
        # half-saturation constant.  Without scaling, unit-reflectance inputs
        # produce excitations >> σ (≈ 0.5), saturating all responses to R_max
        # and eliminating chromatic discriminability.
        spectral = self._upsampler.upsample(
            rgb_image,
            input_mode="reflectance_under_d65",
        )
        spectral.data = (spectral.data * self._stimulus_scale).astype(np.float32)

        # 2. Optical stage.
        optical = OpticalStage(self._cfg.optical)

        class _Scene:
            pass

        scene = _Scene()
        scene.mm_per_pixel = (pixel_scale_mm, pixel_scale_mm)  # type: ignore[attr-defined]
        scene.defocus_residual_diopters = 0.0  # type: ignore[attr-defined]

        irradiance = optical.apply(spectral, scene=scene)
        irradiance.metadata["pixel_scale_mm"] = pixel_scale_mm

        # 3. Photoreceptor mosaic.
        retinal_params = dataclasses.replace(
            self._cfg.retinal, patch_extent_deg=patch_size_deg
        )
        mosaic_gen = MosaicGenerator(retinal_params, self._cfg.optical, WAVELENGTHS)
        mosaic = mosaic_gen.generate(seed=seed)

        if mosaic.n_receptors == 0:
            return None

        # 4. Retinal stage.
        retinal_stage = RetinalStage(retinal_params, self._cfg.optical)
        return retinal_stage.compute_response(mosaic, irradiance)


# ---------------------------------------------------------------------------
# Module-level helper (mirrors acuity.py)
# ---------------------------------------------------------------------------

def _dominant_cone_type(mosaic) -> str:
    """Return the most common cone type in *mosaic* (excluding rods)."""
    cone_mask = mosaic.types != "rod"
    if not np.any(cone_mask):
        return mosaic.types[0]
    cone_types = mosaic.types[cone_mask]
    unique, counts = np.unique(cone_types, return_counts=True)
    return str(unique[np.argmax(counts)])


def evaluate_stimulus_matrix(
    species_list: list[str],
    *,
    patch_size_deg: float = 2.0,
    image_size_px: int = 48,
    n_seeds: int = 2,
    seed: int = 42,
) -> dict[str, dict[str, list[float]]]:
    """Evaluate fixed confusion/control stimulus panels for multiple species."""
    validators = {
        species: DichromatValidator(species, seed=seed, n_seeds=n_seeds)
        for species in species_list
    }
    results: dict[str, dict[str, list[float]]] = {
        species: {"confusion_dog": [], "confusion_cat": [], "control": []}
        for species in species_list
    }

    for dataset_name, items in (
        ("confusion_dog", confusion_pairs("dog")),
        ("confusion_cat", confusion_pairs("cat")),
        ("control", control_pairs()),
    ):
        for item in items:
            fg_rgb, bg_rgb = as_uint8_pair(item)
            for species, validator in validators.items():
                d_value = validator.discriminability(
                    fg_rgb,
                    bg_rgb,
                    patch_size_deg=patch_size_deg,
                    image_size_px=image_size_px,
                )
                results[species][dataset_name].append(float(d_value))

    return results
