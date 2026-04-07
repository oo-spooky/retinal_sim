"""Snellen acuity prediction via full pipeline simulation (Phase 9).

Algorithm
---------
For each candidate letter angular size *A* (arcminutes):

1.  Build a 64×64 RGB Snellen-E image spanning 2.5A arcminutes, in two
    orientations (right-facing and left-facing).
2.  Run both images through the full pipeline:
        SpectralUpsampler → OpticalStage → RetinalStage
    using a photoreceptor mosaic sized to the same 2.5A arcmin patch.
3.  Restrict to the dominant (most numerous) cone type.
4.  Compute Pearson-correlation distance between the two response
    vectors: ``D = 1 - corr(r1, r2)`` where corr is Pearson correlation.
5.  Average D over ``n_seeds`` independent mosaic realisations.

Predicted acuity = smallest *A* where averaged D > ``threshold`` (default 0.05).

Notes on expected accuracy
--------------------------
The PoC model captures optics (PSF) and photoreceptor sampling but omits
post-receptoral neural factors.  Published behavioral acuity is therefore
expected to be 2–5× worse (larger) than the model prediction for dog/cat,
because ganglion-cell convergence further limits spatial resolution.  The
model prediction for human is expected to be close to the 1 arcmin behavioral
limit because human foveal acuity is primarily optics-limited.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import numpy as np

from retinal_sim.constants import WAVELENGTHS
from retinal_sim.optical.stage import OpticalStage
from retinal_sim.retina.mosaic import MosaicGenerator
from retinal_sim.retina.stage import RetinalStage
from retinal_sim.species.config import SpeciesConfig
from retinal_sim.spectral.upsampler import SpectralUpsampler
from retinal_sim.validation.snellen import snellen_scene_rgb

# Image size for all acuity simulations.
_IMG_PX = 64
# Patch width = PATCH_FACTOR * letter_angular_size.
_PATCH_FACTOR = 2.5
# Default discriminability threshold (Pearson-distance scale).
_DEFAULT_THRESHOLD = 0.05
# Minimum number of dominant-cone receptors inside the letter bounding box
# required for a reliable discriminability estimate.
_MIN_CONES_IN_LETTER = 15
# Angular sizes tested by default (arcminutes).
_DEFAULT_SIZES = [1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]


class AcuityValidator:
    """Predict Snellen grating acuity for a given species via the full pipeline.

    Args:
        species_name:  One of ``'human'``, ``'dog'``, ``'cat'``.
        seed:          Base random seed for mosaic generation.
        n_seeds:       Number of mosaic seeds to average over per test size.
    """

    def __init__(
        self,
        species_name: str,
        seed: int = 42,
        n_seeds: int = 2,
    ) -> None:
        self._species_name = species_name
        self._base_seed = seed
        self._n_seeds = n_seeds
        self._cfg = SpeciesConfig.load(species_name)
        self._upsampler = SpectralUpsampler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discriminability(
        self,
        angular_size_arcmin: float,
        orientation1: str = "right",
        orientation2: str = "left",
    ) -> float:
        """Return mean discriminability D in [0, 1] averaged over n_seeds mosaics.

        D = mean|r1 - r2| / (mean(r1 + r2) + eps), where r1/r2 are the
        dominant-cone responses to two letter orientations.

        Larger D → letters are more distinguishable.  D ≈ 0 → indistinguishable.
        """
        patch_arcmin = _PATCH_FACTOR * angular_size_arcmin

        d_values = []
        for k in range(self._n_seeds):
            seed = self._base_seed + k
            d = self._discriminability_one_seed(
                angular_size_arcmin, patch_arcmin, orientation1, orientation2, seed
            )
            d_values.append(d)
        return float(np.mean(d_values))

    def predict_acuity(
        self,
        test_sizes_arcmin: Optional[Sequence[float]] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> float:
        """Return predicted acuity limit (arcminutes).

        Searches test sizes from smallest to largest and returns the first
        size at which mean discriminability > ``threshold``.  If no size
        clears the threshold the largest tested size is returned.

        Args:
            test_sizes_arcmin: Letter angular heights to test, sorted ascending.
                               Defaults to ``[1.5, 2, 3, 5, 8, 12, 20]``.
            threshold:         Discriminability threshold (default 0.05).

        Returns:
            Predicted acuity in arcminutes (MAR = predicted / 5).
        """
        sizes = sorted(test_sizes_arcmin or _DEFAULT_SIZES)
        for size in sizes:
            if self.discriminability(size) >= threshold:
                return float(size)
        return float(sizes[-1])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discriminability_one_seed(
        self,
        angular_size_arcmin: float,
        patch_arcmin: float,
        orientation1: str,
        orientation2: str,
        seed: int,
    ) -> float:
        """Compute discriminability for a single mosaic realisation.

        Metric: Pearson-correlation distance D = 1 - corr(r1, r2) restricted
        to dominant-cone receptors whose positions fall inside the letter's
        angular bounding box.  Restricting to the letter region avoids dilution
        by the large number of background cones (which respond identically to
        both orientations and reduce apparent discriminability).

        Returns 0.0 when fewer than _MIN_CONES_IN_LETTER cones fall inside the
        letter box (too few samples for a reliable estimate).
        """
        act1 = self._run_pipeline(angular_size_arcmin, patch_arcmin, orientation1, seed)
        act2 = self._run_pipeline(angular_size_arcmin, patch_arcmin, orientation2, seed)

        if act1 is None or act2 is None:
            return 0.0

        # Restrict to dominant cone type.
        dom = _dominant_cone_type(act1.mosaic)
        type_mask = act1.mosaic.types == dom

        # Further restrict to cones inside the letter's angular bounding box.
        # The letter is centred at the patch origin (0, 0) in retinal mm.
        letter_half_mm = (
            self._cfg.optical.focal_length_mm
            * np.tan(np.radians(angular_size_arcmin / 2.0 / 60.0))
        )
        pos = act1.mosaic.positions  # (N, 2) float32
        in_box = (
            (np.abs(pos[:, 0]) <= letter_half_mm)
            & (np.abs(pos[:, 1]) <= letter_half_mm)
        )
        letter_mask = type_mask & in_box

        r1 = act1.responses[letter_mask]
        r2 = act2.responses[letter_mask]

        if len(r1) < _MIN_CONES_IN_LETTER:
            return 0.0

        # Pearson-correlation distance: D = 1 - corr(r1, r2).
        r1_c = r1 - r1.mean()
        r2_c = r2 - r2.mean()
        denom = float(np.sqrt(np.dot(r1_c, r1_c) * np.dot(r2_c, r2_c) + 1e-12))
        corr = float(np.dot(r1_c, r2_c)) / denom
        return float(1.0 - corr)

    def _run_pipeline(
        self,
        angular_size_arcmin: float,
        patch_arcmin: float,
        orientation: str,
        seed: int,
    ):
        """Run SpectralUpsampler → OpticalStage → RetinalStage for one image.

        Returns a MosaicActivation, or None if the patch is too small to
        generate a mosaic (< 1 cone).
        """
        patch_deg = patch_arcmin / 60.0
        focal_length_mm = self._cfg.optical.focal_length_mm

        # Retinal extent of the patch (mm).
        retinal_half_mm = focal_length_mm * np.tan(np.radians(patch_deg / 2.0))
        retinal_width_mm = 2.0 * retinal_half_mm
        pixel_scale_mm = retinal_width_mm / _IMG_PX

        # 1. Generate RGB image.
        rgb = snellen_scene_rgb(angular_size_arcmin, patch_arcmin, _IMG_PX, orientation)

        # 2. Spectral upsampling.
        spectral = self._upsampler.upsample(
            rgb,
            input_mode="reflectance_under_d65",
        )

        # 3. Optical stage (PSF).
        optical = OpticalStage(self._cfg.optical)

        class _Scene:
            """Minimal scene-like object supplying pixel scale + zero defocus."""
            pass

        scene = _Scene()
        scene.mm_per_pixel = (pixel_scale_mm, pixel_scale_mm)  # type: ignore[attr-defined]
        scene.defocus_residual_diopters = 0.0  # type: ignore[attr-defined]

        irradiance = optical.apply(spectral, scene=scene)
        irradiance.metadata["pixel_scale_mm"] = pixel_scale_mm

        # 4. Photoreceptor mosaic sized to this patch.
        retinal_params = dataclasses.replace(
            self._cfg.retinal, patch_extent_deg=patch_deg
        )

        mosaic_gen = MosaicGenerator(retinal_params, self._cfg.optical, WAVELENGTHS)
        mosaic = mosaic_gen.generate(seed=seed)

        if mosaic.n_receptors == 0:
            return None

        # 5. Retinal stage.
        retinal_stage = RetinalStage(retinal_params, self._cfg.optical)
        return retinal_stage.compute_response(mosaic, irradiance)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _dominant_cone_type(mosaic) -> str:
    """Return the most common cone type in *mosaic* (excluding rods)."""
    cone_mask = mosaic.types != "rod"
    if not np.any(cone_mask):
        return mosaic.types[0]  # fallback: no cones → use whatever is there
    cone_types = mosaic.types[cone_mask]
    unique, counts = np.unique(cone_types, return_counts=True)
    return str(unique[np.argmax(counts)])
