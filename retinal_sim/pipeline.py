"""RetinalSimulator pipeline orchestrator (Phase 12).

Wires together the full simulation chain:
    SpeciesConfig → SceneGeometry → SpectralUpsampler → OpticalStage → RetinalStage

Usage::

    sim = RetinalSimulator("human")
    result = sim.simulate(rgb_image, scene_width_m=0.3, viewing_distance_m=6.0)

    results = sim.compare_species(
        rgb_image, ["human", "dog", "cat"],
        scene_width_m=0.3, viewing_distance_m=6.0,
    )
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from retinal_sim.optical.stage import OpticalStage, RetinalIrradiance
from retinal_sim.retina.mosaic import MosaicGenerator, PhotoreceptorMosaic
from retinal_sim.retina.stage import MosaicActivation, RetinalStage
from retinal_sim.scene.geometry import SceneGeometry, SceneDescription
from retinal_sim.species.config import SpeciesConfig
from retinal_sim.spectral.upsampler import SpectralImage, SpectralUpsampler
from retinal_sim.constants import WAVELENGTHS
from retinal_sim.validation.metrics import (
    compute_simulation_summary_metrics,
    receptor_pixel_coordinates,
    stimulated_receptor_mask,
)


@dataclass
class SimulationResult:
    """Container for all pipeline stage outputs."""
    scene: SceneDescription
    spectral_image: SpectralImage
    retinal_irradiance: RetinalIrradiance
    mosaic: PhotoreceptorMosaic
    activation: MosaicActivation
    species_name: str = ""
    metadata: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    summary_metrics: dict = field(default_factory=dict)


class RetinalSimulator:
    """Top-level pipeline orchestrator.

    Args:
        species: Species name ('human', 'dog', 'cat') or a SpeciesConfig object.
        patch_extent_deg: Angular size of the simulated retinal patch (degrees).
        light_level: 'photopic' | 'mesopic' | 'scotopic'
        stimulus_scale: Multiplicative scale for spectral irradiance before
            transduction.  Keeps excitations near the Naka-Rushton
            half-saturation constant (see SCRATCHPAD Phase 10 gotchas).
        seed: Base random seed for mosaic generation.
    """

    def __init__(
        self,
        species: Union[str, SpeciesConfig],
        patch_extent_deg: float = 2.0,
        light_level: str = "photopic",
        stimulus_scale: float = 0.01,
        seed: int = 42,
    ) -> None:
        if isinstance(species, str):
            self._cfg = SpeciesConfig.load(species)
        else:
            self._cfg = species

        self._patch_extent_deg = patch_extent_deg
        self._light_level = light_level
        self._stimulus_scale = stimulus_scale
        self._seed = seed

        # Shared spectral upsampler (stateless, species-independent).
        self._upsampler = SpectralUpsampler()

    @property
    def species_name(self) -> str:
        return self._cfg.name

    @property
    def config(self) -> SpeciesConfig:
        return self._cfg

    def simulate(
        self,
        input_image: np.ndarray,
        scene_width_m: Optional[float] = None,
        viewing_distance_m: float = float("inf"),
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Run the full pipeline on a single image for this simulator's species.

        Args:
            input_image: (H, W, 3) uint8 RGB image.
            scene_width_m: Physical width of the scene (metres). If None, the
                image is treated as spanning exactly the patch extent.
            viewing_distance_m: Eye-to-scene distance (metres).
            seed: Mosaic random seed override (defaults to self._seed).

        Returns:
            SimulationResult with all intermediate outputs.
        """
        seed = seed if seed is not None else self._seed
        H, W = input_image.shape[:2]
        optical = self._cfg.optical
        retinal = self._cfg.retinal

        # --- Scene geometry ---
        if scene_width_m is not None:
            geom = SceneGeometry(scene_width_m, viewing_distance_m)
            scene = geom.compute((H, W), optical)
        else:
            # No physical scene: create a synthetic SceneDescription spanning
            # the full patch extent.
            scene = self._default_scene(H, W, optical)

        # --- Spectral upsampling ---
        spectral = self._upsampler.upsample(input_image)
        spectral.data = (spectral.data * self._stimulus_scale).astype(np.float32)

        # --- Optical stage ---
        optical_stage = OpticalStage(optical)
        irradiance = optical_stage.apply(spectral, scene=scene)

        # Embed pixel scale for RetinalStage fallback.
        if hasattr(scene, "mm_per_pixel") and scene.mm_per_pixel:
            irradiance.metadata["pixel_scale_mm"] = float(
                np.asarray(scene.mm_per_pixel).flat[0]
            )

        # --- Retinal stage ---
        retinal_params = dataclasses.replace(
            retinal, patch_extent_deg=self._patch_extent_deg,
        )
        retinal_stage = RetinalStage(retinal_params, optical)
        mosaic = retinal_stage.generate_mosaic(seed=seed)
        activation = retinal_stage.compute_response(mosaic, irradiance, scene)

        artifacts = self._build_artifacts(
            scene=scene,
            mosaic=mosaic,
            input_image=input_image,
        )
        temp_result = SimulationResult(
            scene=scene,
            spectral_image=spectral,
            retinal_irradiance=irradiance,
            mosaic=mosaic,
            activation=activation,
            species_name=self._cfg.name,
            artifacts=artifacts,
        )
        summary_metrics = compute_simulation_summary_metrics(temp_result, input_image)

        return SimulationResult(
            scene=scene,
            spectral_image=spectral,
            retinal_irradiance=irradiance,
            mosaic=mosaic,
            activation=activation,
            species_name=self._cfg.name,
            artifacts=artifacts,
            summary_metrics=summary_metrics,
        )

    def compare_species(
        self,
        input_image: np.ndarray,
        species_list: List[str],
        scene_width_m: Optional[float] = None,
        viewing_distance_m: float = float("inf"),
        seed: Optional[int] = None,
    ) -> Dict[str, SimulationResult]:
        """Run the same image through multiple species pipelines.

        Creates a fresh RetinalSimulator per species, sharing the same
        stimulus_scale, patch_extent_deg, and seed.

        Args:
            input_image: (H, W, 3) uint8 RGB image.
            species_list: List of species names.
            scene_width_m: Physical width of the scene (metres).
            viewing_distance_m: Eye-to-scene distance (metres).
            seed: Mosaic random seed override.

        Returns:
            Dict mapping species name → SimulationResult.
        """
        results: Dict[str, SimulationResult] = {}
        for sp in species_list:
            sim = RetinalSimulator(
                sp,
                patch_extent_deg=self._patch_extent_deg,
                light_level=self._light_level,
                stimulus_scale=self._stimulus_scale,
                seed=self._seed,
            )
            results[sp] = sim.simulate(
                input_image, scene_width_m, viewing_distance_m,
                seed=seed,
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_scene(self, H: int, W: int, optical) -> SceneDescription:
        """Synthetic SceneDescription when no physical scene dimensions given.

        Places the image at infinity, covering exactly the configured patch
        extent.  Used when the caller just wants to feed an image without
        worrying about scene geometry.
        """
        focal_mm = optical.focal_length_mm
        patch_half_mm = focal_mm * np.tan(
            np.radians(self._patch_extent_deg / 2.0)
        )
        retinal_w_mm = 2.0 * patch_half_mm
        retinal_h_mm = retinal_w_mm * H / W if W > 0 else retinal_w_mm

        mm_per_px_x = retinal_w_mm / W if W > 0 else 0.0
        mm_per_px_y = retinal_h_mm / H if H > 0 else 0.0

        return SceneDescription(
            scene_width_m=0.0,
            scene_height_m=0.0,
            viewing_distance_m=float("inf"),
            angular_width_deg=self._patch_extent_deg,
            angular_height_deg=self._patch_extent_deg * H / W if W > 0 else self._patch_extent_deg,
            retinal_width_mm=float(retinal_w_mm),
            retinal_height_mm=float(retinal_h_mm),
            mm_per_pixel=(mm_per_px_x, mm_per_px_y),
            accommodation_demand_diopters=0.0,
            defocus_residual_diopters=0.0,
            clipped=False,
            scene_covers_patch_fraction=1.0,
        )

    def _build_artifacts(
        self,
        scene: SceneDescription,
        mosaic: PhotoreceptorMosaic,
        input_image: np.ndarray,
    ) -> dict:
        """Build lightweight derived artifacts used by validation and demos."""
        class _ResultLike:
            def __init__(self, scene_obj, mosaic_obj) -> None:
                self.scene = scene_obj
                self.mosaic = mosaic_obj

        result_like = _ResultLike(scene, mosaic)
        rows, cols = receptor_pixel_coordinates(result_like, input_image.shape[:2])
        stimulated_mask = stimulated_receptor_mask(result_like, input_image.shape[:2])
        return {
            "input_shape": tuple(int(v) for v in input_image.shape),
            "receptor_rows": rows,
            "receptor_cols": cols,
            "stimulated_receptor_mask": stimulated_mask,
        }
