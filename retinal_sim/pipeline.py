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
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from retinal_sim.output.diagnostics import (
    build_optical_delivery_diagnostics,
    build_comparative_renderings,
    build_photoreceptor_activation_diagnostics,
    build_retinal_irradiance_diagnostics,
    build_spectral_interpretation_diagnostics,
)
from retinal_sim.optical.stage import OpticalStage, RetinalIrradiance
from retinal_sim.retina.mosaic import MosaicGenerator, PhotoreceptorMosaic
from retinal_sim.retina.stage import MosaicActivation, RetinalStage
from retinal_sim.scene.geometry import SceneGeometry, SceneDescription
from retinal_sim.species.config import SpeciesConfig
from retinal_sim.spectral.upsampler import (
    DEFAULT_SCENE_INPUT_MODE,
    RGB_SCENE_INPUT_MODES,
    SpectralImage,
    SpectralUpsampler,
    normalize_scene_input_mode,
    scene_input_metadata,
)
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
        self._default_input_mode = DEFAULT_SCENE_INPUT_MODE

    @property
    def species_name(self) -> str:
        return self._cfg.name

    @property
    def config(self) -> SpeciesConfig:
        return self._cfg

    def simulate(
        self,
        input_image: Union[np.ndarray, SpectralImage],
        scene_width_m: Optional[float] = None,
        viewing_distance_m: float = float("inf"),
        seed: Optional[int] = None,
        input_mode: Optional[str] = None,
        artifact_render_longest_edge_px: Optional[int] = None,
    ) -> SimulationResult:
        """Run the full pipeline on a single image for this simulator's species.

        Args:
            input_image: RGB image for RGB modes, or SpectralImage for
                ``measured_spectrum``.
            scene_width_m: Physical width of the scene (metres). If None, the
                image is treated as spanning exactly the patch extent.
            viewing_distance_m: Eye-to-scene distance (metres).
            seed: Mosaic random seed override (defaults to self._seed).
            input_mode: One of ``reflectance_under_d65``, ``display_rgb``, or
                ``measured_spectrum``. If omitted for RGB input, defaults to
                ``reflectance_under_d65`` with a compatibility warning.
            artifact_render_longest_edge_px: Optional longest edge in pixels for
                activation-derived artifact families. This changes render density
                only; irradiance artifacts remain at native input resolution.

        Returns:
            SimulationResult with all intermediate outputs.
        """
        seed = seed if seed is not None else self._seed
        mode = self._resolve_input_mode(input_image, input_mode)
        H, W = self._input_shape(input_image)
        optical = self._cfg.optical
        retinal = self._cfg.retinal
        scene_meta = scene_input_metadata(mode)

        # --- Scene geometry ---
        if scene_width_m is not None:
            geom = SceneGeometry(scene_width_m, viewing_distance_m)
            scene = geom.compute((H, W), optical)
        else:
            # No physical scene: create a synthetic SceneDescription spanning
            # the full patch extent.
            scene = self._default_scene(H, W, optical)

        # --- Spectral construction / bypass ---
        if mode == "measured_spectrum":
            spectral = self._copy_spectral_image(input_image)
        else:
            spectral = self._upsampler.upsample(input_image, input_mode=mode)
        spectral.data = (spectral.data * self._stimulus_scale).astype(np.float32)
        spectral.metadata.update(scene_meta)

        # --- Optical stage ---
        optical_stage = OpticalStage(optical)
        irradiance = optical_stage.apply(spectral, scene=scene)
        irradiance.metadata.update(scene_meta)

        # Embed pixel scale for RetinalStage fallback.
        if hasattr(scene, "mm_per_pixel") and scene.mm_per_pixel:
            irradiance.metadata["pixel_scale_mm"] = float(
                np.asarray(scene.mm_per_pixel).flat[0]
            )

        # --- Retinal stage ---
        retinal_params = dataclasses.replace(
            retinal, patch_extent_deg=self._patch_extent_deg,
        )
        retinal_meta = {"retinal_physiology": retinal_params.physiology_metadata()}
        retinal_stage = RetinalStage(retinal_params, optical)
        mosaic = retinal_stage.generate_mosaic(seed=seed)
        activation = retinal_stage.compute_response(mosaic, irradiance, scene)

        artifacts = self._build_artifacts(
            scene=scene,
            spectral_image=spectral,
            retinal_irradiance=irradiance,
            mosaic=mosaic,
            activation=activation,
            input_image=input_image,
            artifact_render_longest_edge_px=artifact_render_longest_edge_px,
        )
        temp_result = SimulationResult(
            scene=scene,
            spectral_image=spectral,
            retinal_irradiance=irradiance,
            mosaic=mosaic,
            activation=activation,
            species_name=self._cfg.name,
            metadata={**scene_meta, **retinal_meta},
            artifacts=artifacts,
        )
        summary_metrics = compute_simulation_summary_metrics(
            temp_result,
            spectral.data if isinstance(input_image, SpectralImage) else input_image,
        )

        return SimulationResult(
            scene=scene,
            spectral_image=spectral,
            retinal_irradiance=irradiance,
            mosaic=mosaic,
            activation=activation,
            species_name=self._cfg.name,
            metadata={**scene_meta, **retinal_meta},
            artifacts=artifacts,
            summary_metrics=summary_metrics,
        )

    def compare_species(
        self,
        input_image: Union[np.ndarray, SpectralImage],
        species_list: List[str],
        scene_width_m: Optional[float] = None,
        viewing_distance_m: float = float("inf"),
        seed: Optional[int] = None,
        input_mode: Optional[str] = None,
        artifact_render_longest_edge_px: Optional[int] = None,
    ) -> Dict[str, SimulationResult]:
        """Run the same image through multiple species pipelines.

        Creates a fresh RetinalSimulator per species, sharing the same
        stimulus_scale, patch_extent_deg, and seed.

        Args:
            input_image: RGB image or SpectralImage.
            species_list: List of species names.
            scene_width_m: Physical width of the scene (metres).
            viewing_distance_m: Eye-to-scene distance (metres).
            seed: Mosaic random seed override.
            input_mode: Explicit scene-spectrum semantics.
            artifact_render_longest_edge_px: Optional longest edge in pixels for
                activation-derived artifact families.

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
                input_mode=input_mode,
                artifact_render_longest_edge_px=artifact_render_longest_edge_px,
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
        spectral_image: SpectralImage,
        retinal_irradiance: RetinalIrradiance,
        mosaic: PhotoreceptorMosaic,
        activation: MosaicActivation,
        input_image: Union[np.ndarray, SpectralImage],
        artifact_render_longest_edge_px: Optional[int] = None,
    ) -> dict:
        """Build lightweight derived artifacts used by validation and demos."""
        class _ResultLike:
            def __init__(self, scene_obj, mosaic_obj) -> None:
                self.scene = scene_obj
                self.mosaic = mosaic_obj

        result_like = _ResultLike(scene, mosaic)
        image_shape = self._input_shape(input_image)
        artifact_render_shape = (
            self._artifact_render_shape(image_shape, artifact_render_longest_edge_px)
            if artifact_render_longest_edge_px is not None
            else None
        )
        rows, cols = receptor_pixel_coordinates(result_like, image_shape)
        stimulated_mask = stimulated_receptor_mask(result_like, image_shape)
        render_rows = render_cols = None
        if artifact_render_shape is not None:
            render_rows, render_cols = self._scene_pixel_coordinates(
                scene,
                mosaic.positions,
                artifact_render_shape,
            )
        return {
            "input_shape": tuple(int(v) for v in image_shape),
            "artifact_render_shape": (
                tuple(int(v) for v in artifact_render_shape)
                if artifact_render_shape is not None else None
            ),
            "receptor_rows": rows,
            "receptor_cols": cols,
            "render_receptor_rows": render_rows,
            "render_receptor_cols": render_cols,
            "stimulated_receptor_mask": stimulated_mask,
            "spectral_interpretation_diagnostics": build_spectral_interpretation_diagnostics(
                spectral_image=spectral_image,
                native_input_shape=image_shape,
            ),
            "optical_delivery_diagnostics": build_optical_delivery_diagnostics(
                spectral_image=spectral_image,
                retinal_irradiance=retinal_irradiance,
            ),
            "retinal_irradiance_diagnostics": build_retinal_irradiance_diagnostics(
                spectral_image=spectral_image,
                retinal_irradiance=retinal_irradiance,
                native_input_shape=image_shape,
            ),
            "photoreceptor_activation_diagnostics": build_photoreceptor_activation_diagnostics(
                scene=scene,
                mosaic=mosaic,
                activation=activation,
                receptor_rows=rows,
                receptor_cols=cols,
                stimulated_mask=stimulated_mask,
                input_shape=image_shape,
                overlay_rows=render_rows,
                overlay_cols=render_cols,
                overlay_shape=artifact_render_shape,
            ),
            "comparative_renderings": build_comparative_renderings(
                activation=activation,
                input_shape=image_shape,
                output_shape=artifact_render_shape,
            ),
        }

    def _artifact_render_shape(
        self,
        input_shape: tuple[int, int],
        longest_edge_px: Optional[int],
    ) -> tuple[int, int]:
        """Return the activation-artifact render grid for a native input shape."""
        height, width = int(input_shape[0]), int(input_shape[1])
        if longest_edge_px is None:
            return (height, width)
        longest_edge_px = max(int(longest_edge_px), 1)
        longest = max(height, width, 1)
        scale = float(longest_edge_px) / float(longest)
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _scene_pixel_coordinates(
        self,
        scene: SceneDescription,
        positions: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map receptor positions onto an arbitrary scene-aligned output grid."""
        height, width = int(image_shape[0]), int(image_shape[1])
        width_mm = float(getattr(scene, "retinal_width_mm", 0.0))
        height_mm = float(getattr(scene, "retinal_height_mm", 0.0))

        if width_mm <= 0.0 or height_mm <= 0.0:
            mm_per_pixel = float(np.asarray(scene.mm_per_pixel).flat[0])
            mm_per_pixel_x = mm_per_pixel_y = mm_per_pixel
        else:
            mm_per_pixel_x = width_mm / max(width, 1)
            mm_per_pixel_y = height_mm / max(height, 1)

        cols = np.clip(
            np.round(positions[:, 0] / mm_per_pixel_x + (width - 1) / 2.0).astype(int),
            0,
            width - 1,
        )
        rows = np.clip(
            np.round(positions[:, 1] / mm_per_pixel_y + (height - 1) / 2.0).astype(int),
            0,
            height - 1,
        )
        return rows, cols

    def _resolve_input_mode(
        self,
        input_image: Union[np.ndarray, SpectralImage],
        input_mode: Optional[str],
    ) -> str:
        """Validate input type against explicit scene semantics."""
        is_spectral = isinstance(input_image, SpectralImage)
        if input_mode is None:
            if is_spectral:
                raise ValueError(
                    "SpectralImage input requires input_mode='measured_spectrum'; "
                    "implicit defaults apply only to RGB ndarray input."
                )
            warnings.warn(
                "RetinalSimulator.simulate() defaulting RGB input to "
                "'reflectance_under_d65'. Pass input_mode explicitly to avoid "
                "this compatibility fallback.",
                FutureWarning,
                stacklevel=3,
            )
            return self._default_input_mode

        mode = normalize_scene_input_mode(input_mode)
        if mode == "measured_spectrum" and not is_spectral:
            raise ValueError(
                "input_mode='measured_spectrum' requires a SpectralImage input."
            )
        if mode in RGB_SCENE_INPUT_MODES and is_spectral:
            raise ValueError(
                f"input_mode={mode!r} requires an RGB ndarray input, not SpectralImage."
            )
        return mode

    def _input_shape(
        self,
        input_image: Union[np.ndarray, SpectralImage],
    ) -> tuple[int, int]:
        """Return image height/width for RGB or measured spectral inputs."""
        if isinstance(input_image, SpectralImage):
            return tuple(int(v) for v in input_image.data.shape[:2])
        return tuple(int(v) for v in np.asarray(input_image).shape[:2])

    def _copy_spectral_image(
        self,
        spectral_image: Union[np.ndarray, SpectralImage],
    ) -> SpectralImage:
        """Return a detached SpectralImage copy for measured-spectrum runs."""
        if not isinstance(spectral_image, SpectralImage):
            raise ValueError(
                "Expected SpectralImage input for input_mode='measured_spectrum'."
            )
        return SpectralImage(
            data=np.asarray(spectral_image.data, dtype=np.float32).copy(),
            wavelengths=np.asarray(spectral_image.wavelengths, dtype=np.float64).copy(),
            metadata=dict(getattr(spectral_image, "metadata", {})),
        )
