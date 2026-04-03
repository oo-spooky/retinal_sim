"""Phase 11 tests: distance-dependent resolution.

Validation criterion (architecture §11e):
- For a fixed physical object, receptor counts inside its retinal image
  should scale approximately as 1 / d² with viewing distance.
- Retinal image extent scales as 1/d (via arctan geometry).
- Larger focal lengths produce larger retinal images at the same distance.

The tests use uniform-density mosaics (density frozen at the area centralis
value) so that 1/d² scaling is purely geometric, not confounded by the
eccentricity-dependent density gradient.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np
import pytest

from retinal_sim.retina.mosaic import MosaicGenerator, PhotoreceptorMosaic
from retinal_sim.scene.geometry import SceneGeometry
from retinal_sim.species.config import SpeciesConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBJECT_SIZE_M = 0.20          # 20 cm object
DISTANCES_M = (1.0, 10.0, 100.0)
IMAGE_SHAPE = (256, 256)
SEED = 42
MIN_COUNT_FOR_RATIO = 5
RATIO_TOLERANCE = 0.15        # 15 % relative error on count ratios
AREA_RATIO_TOL = 0.02         # 2 % for analytic retinal-area ratios
SPECIES = ["human", "dog", "cat"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _angular_width_deg(scene_width_m: float, distance_m: float) -> float:
    return float(2.0 * np.degrees(np.arctan(scene_width_m / (2.0 * distance_m))))


def _uniform_density_retinal_params(retinal_params):
    """Freeze densities at the area-centralis value to isolate geometry scaling.

    Phase 11 validates that retinal image area ∝ 1/d².  Using the native
    eccentricity-dependent density would confound the test with density
    variation (especially for species with steep gradients like human).
    """
    center_cone_density = retinal_params.cone_density_fn(0.0, 0.0)
    center_rod_density = retinal_params.rod_density_fn(0.0, 0.0)

    def cone_density_fn(ecc_mm: float, angle: float = 0.0,
                        _d: dict = dict(center_cone_density)) -> Dict[str, float]:
        return dict(_d)

    def rod_density_fn(ecc_mm: float, angle: float = 0.0,
                       _r: float = float(center_rod_density)) -> float:
        return _r

    return dataclasses.replace(
        retinal_params,
        cone_density_fn=cone_density_fn,
        rod_density_fn=rod_density_fn,
    )


def _count_receptors_within_extent(
    mosaic: PhotoreceptorMosaic,
    retinal_width_mm: float,
    retinal_height_mm: float,
) -> int:
    half_w = retinal_width_mm / 2.0
    half_h = retinal_height_mm / 2.0
    pos = mosaic.positions
    inside = (np.abs(pos[:, 0]) <= half_w) & (np.abs(pos[:, 1]) <= half_h)
    return int(inside.sum())


def _build_mosaic(species: str, patch_deg: float, seed: int = SEED):
    """Build a uniform-density mosaic covering *patch_deg* for *species*."""
    cfg = SpeciesConfig.load(species)
    rp = dataclasses.replace(
        _uniform_density_retinal_params(cfg.retinal),
        patch_extent_deg=patch_deg,
    )
    mosaic = MosaicGenerator(rp, cfg.optical).generate(seed=seed)
    return mosaic, cfg


def _distance_counts(species: str) -> Dict[float, int]:
    """Receptor counts at each distance for *species* with uniform density."""
    largest_patch_deg = _angular_width_deg(OBJECT_SIZE_M, DISTANCES_M[0])
    mosaic, cfg = _build_mosaic(species, largest_patch_deg)

    counts: Dict[float, int] = {}
    for distance_m in DISTANCES_M:
        scene = SceneGeometry(
            scene_width_m=OBJECT_SIZE_M,
            scene_height_m=OBJECT_SIZE_M,
            viewing_distance_m=distance_m,
        ).compute(IMAGE_SHAPE, cfg.optical)
        counts[distance_m] = _count_receptors_within_extent(
            mosaic, scene.retinal_width_mm, scene.retinal_height_mm,
        )
    return counts


def _scene_at_distance(species: str, distance_m: float):
    """Return SceneDescription for the standard object at *distance_m*."""
    cfg = SpeciesConfig.load(species)
    return SceneGeometry(
        scene_width_m=OBJECT_SIZE_M,
        scene_height_m=OBJECT_SIZE_M,
        viewing_distance_m=distance_m,
    ).compute(IMAGE_SHAPE, cfg.optical)


# ===================================================================
# Test classes
# ===================================================================


class TestRetinalImageGeometry:
    """Validate that retinal image extent follows expected scaling laws."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_retinal_width_positive(self, species):
        """Retinal image width is positive for all standard distances."""
        for d in DISTANCES_M:
            scene = _scene_at_distance(species, d)
            assert scene.retinal_width_mm > 0, f"d={d}"

    @pytest.mark.parametrize("species", SPECIES)
    def test_retinal_width_decreases_with_distance(self, species):
        """Farther objects produce smaller retinal images."""
        widths = [_scene_at_distance(species, d).retinal_width_mm for d in DISTANCES_M]
        for i in range(len(widths) - 1):
            assert widths[i] > widths[i + 1], (
                f"{species}: widths not decreasing: {list(zip(DISTANCES_M, widths))}"
            )

    @pytest.mark.parametrize("species", SPECIES)
    def test_retinal_height_decreases_with_distance(self, species):
        """Farther objects produce smaller retinal images (height)."""
        heights = [_scene_at_distance(species, d).retinal_height_mm for d in DISTANCES_M]
        for i in range(len(heights) - 1):
            assert heights[i] > heights[i + 1]

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("near_d,far_d", [(1.0, 10.0), (10.0, 100.0), (1.0, 100.0)])
    def test_retinal_area_scales_inverse_square(self, species, near_d, far_d):
        """Retinal image AREA should scale as 1/d² across distance pairs."""
        near = _scene_at_distance(species, near_d)
        far = _scene_at_distance(species, far_d)
        area_near = near.retinal_width_mm * near.retinal_height_mm
        area_far = far.retinal_width_mm * far.retinal_height_mm
        actual_ratio = area_near / area_far
        expected_ratio = (far_d / near_d) ** 2
        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_err < AREA_RATIO_TOL, (
            f"{species}: area ratio {near_d}m/{far_d}m = {actual_ratio:.4f}, "
            f"expected {expected_ratio:.2f}, rel err {rel_err:.4%}"
        )

    @pytest.mark.parametrize("species", SPECIES)
    def test_retinal_width_matches_arctan_formula(self, species):
        """retinal_width_mm = 2 * f * tan(arctan(w / 2d))."""
        cfg = SpeciesConfig.load(species)
        f = cfg.optical.focal_length_mm
        for d in DISTANCES_M:
            scene = _scene_at_distance(species, d)
            expected = 2.0 * f * np.tan(np.arctan(OBJECT_SIZE_M / (2.0 * d)))
            np.testing.assert_allclose(
                scene.retinal_width_mm, expected, rtol=1e-10,
                err_msg=f"{species} d={d}",
            )

    @pytest.mark.parametrize("species", SPECIES)
    def test_square_object_gives_square_retinal_image(self, species):
        """A square physical object should give equal retinal width and height."""
        for d in DISTANCES_M:
            scene = _scene_at_distance(species, d)
            np.testing.assert_allclose(
                scene.retinal_width_mm, scene.retinal_height_mm, rtol=1e-10,
                err_msg=f"{species} d={d}",
            )


class TestCrossSpeciesRetinalExtent:
    """Validate cross-species retinal image relationships."""

    @pytest.mark.parametrize("distance_m", DISTANCES_M)
    def test_retinal_width_scales_with_focal_length(self, distance_m):
        """Species with longer focal lengths produce larger retinal images."""
        configs = {s: SpeciesConfig.load(s) for s in SPECIES}
        widths = {s: _scene_at_distance(s, distance_m).retinal_width_mm for s in SPECIES}
        focals = {s: configs[s].optical.focal_length_mm for s in SPECIES}

        # Sort species by focal length
        sorted_species = sorted(SPECIES, key=lambda s: focals[s])
        sorted_widths = [widths[s] for s in sorted_species]
        for i in range(len(sorted_widths) - 1):
            assert sorted_widths[i] <= sorted_widths[i + 1], (
                f"d={distance_m}: species sorted by focal length "
                f"{sorted_species}, but widths {sorted_widths} not monotonic"
            )

    @pytest.mark.parametrize("distance_m", DISTANCES_M)
    def test_retinal_width_ratio_matches_focal_length_ratio(self, distance_m):
        """Retinal width ratio between species should equal focal length ratio."""
        configs = {s: SpeciesConfig.load(s) for s in SPECIES}
        widths = {s: _scene_at_distance(s, distance_m).retinal_width_mm for s in SPECIES}
        focals = {s: configs[s].optical.focal_length_mm for s in SPECIES}

        for s1 in SPECIES:
            for s2 in SPECIES:
                if s1 >= s2:
                    continue
                expected_ratio = focals[s1] / focals[s2]
                actual_ratio = widths[s1] / widths[s2]
                np.testing.assert_allclose(
                    actual_ratio, expected_ratio, rtol=0.01,
                    err_msg=f"{s1} vs {s2} at d={distance_m}m",
                )


class TestDistanceDependentReceptorCount:
    """Core Phase 11 validation: receptor count ∝ 1/d² for uniform density."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_receptor_count_monotonically_decreases(self, species):
        """Receptor count should strictly decrease with distance."""
        counts = _distance_counts(species)
        assert counts[1.0] > counts[10.0] > counts[100.0], (
            f"{species}: expected monotonic decrease, got {counts}"
        )

    @pytest.mark.parametrize("species", SPECIES)
    @pytest.mark.parametrize("near_d,far_d", [(1.0, 10.0), (10.0, 100.0), (1.0, 100.0)])
    def test_receptor_count_ratio_matches_inverse_square(self, species, near_d, far_d):
        """Count ratio between two distances should match (d_far/d_near)²."""
        counts = _distance_counts(species)
        near_count = counts[near_d]
        far_count = counts[far_d]

        if near_count < MIN_COUNT_FOR_RATIO or far_count < MIN_COUNT_FOR_RATIO:
            pytest.skip(
                f"{species}: insufficient receptors at "
                f"{near_d:g}m/{far_d:g}m ({near_count}/{far_count})"
            )

        actual_ratio = near_count / far_count
        expected_ratio = (far_d / near_d) ** 2
        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio

        assert rel_err < RATIO_TOLERANCE, (
            f"{species}: {near_d:g}m/{far_d:g}m count ratio {actual_ratio:.2f}, "
            f"expected {expected_ratio:.2f}, rel err {rel_err:.2%}, counts={counts}"
        )

    @pytest.mark.parametrize("species", SPECIES)
    def test_nearest_distance_has_most_receptors(self, species):
        """The closest distance should always have the most receptors."""
        counts = _distance_counts(species)
        assert counts[DISTANCES_M[0]] == max(counts.values())

    @pytest.mark.parametrize("species", SPECIES)
    def test_farthest_distance_has_fewest_receptors(self, species):
        """The farthest distance should always have the fewest receptors."""
        counts = _distance_counts(species)
        assert counts[DISTANCES_M[-1]] == min(counts.values())

    @pytest.mark.parametrize("species", SPECIES)
    def test_nonzero_count_at_all_distances(self, species):
        """At least some receptors should fall within the retinal image at all distances."""
        counts = _distance_counts(species)
        for d, c in counts.items():
            assert c > 0, f"{species}: zero receptors at d={d}m"


class TestReceptorCountAcrossSpecies:
    """Cross-species receptor count relationships at the same distance."""

    @pytest.mark.parametrize("distance_m", DISTANCES_M)
    def test_count_correlates_with_density_times_area(self, distance_m):
        """Species with higher (density × retinal_area) should have more receptors."""
        results = {}
        for species in SPECIES:
            cfg = SpeciesConfig.load(species)
            scene = _scene_at_distance(species, distance_m)
            area_mm2 = scene.retinal_width_mm * scene.retinal_height_mm
            center_cone = cfg.retinal.cone_density_fn(0.0, 0.0)
            center_rod = cfg.retinal.rod_density_fn(0.0, 0.0)
            total_density = sum(center_cone.values()) + center_rod
            results[species] = {
                "expected": total_density * area_mm2,
                "area": area_mm2,
                "density": total_density,
            }

        # Sort by expected count — actual mosaic counts should follow the same ordering
        sorted_species = sorted(SPECIES, key=lambda s: results[s]["expected"])

        # Build actual counts
        largest_patch_deg = _angular_width_deg(OBJECT_SIZE_M, DISTANCES_M[0])
        actual_counts = {}
        for species in SPECIES:
            mosaic, cfg = _build_mosaic(species, largest_patch_deg)
            scene = _scene_at_distance(species, distance_m)
            actual_counts[species] = _count_receptors_within_extent(
                mosaic, scene.retinal_width_mm, scene.retinal_height_mm,
            )

        sorted_actual = [actual_counts[s] for s in sorted_species]
        for i in range(len(sorted_actual) - 1):
            assert sorted_actual[i] <= sorted_actual[i + 1], (
                f"d={distance_m}: expected ordering {sorted_species} by density×area, "
                f"but counts {sorted_actual} not monotonic"
            )


class TestMosaicCoverageAtDistance:
    """Validate that mosaics adequately cover the retinal image area."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_mosaic_covers_nearest_distance_fully(self, species):
        """At 1m, the mosaic should fully cover the retinal image (patch >= image)."""
        largest_patch_deg = _angular_width_deg(OBJECT_SIZE_M, DISTANCES_M[0])
        mosaic, cfg = _build_mosaic(species, largest_patch_deg)
        scene = _scene_at_distance(species, 1.0)

        # The mosaic patch should be at least as wide as the retinal image
        pos = mosaic.positions
        mosaic_extent_x = pos[:, 0].max() - pos[:, 0].min()
        mosaic_extent_y = pos[:, 1].max() - pos[:, 1].min()

        assert mosaic_extent_x >= scene.retinal_width_mm * 0.95, (
            f"{species}: mosaic x-extent {mosaic_extent_x:.4f}mm < "
            f"retinal width {scene.retinal_width_mm:.4f}mm"
        )
        assert mosaic_extent_y >= scene.retinal_height_mm * 0.95, (
            f"{species}: mosaic y-extent {mosaic_extent_y:.4f}mm < "
            f"retinal height {scene.retinal_height_mm:.4f}mm"
        )

    @pytest.mark.parametrize("species", SPECIES)
    def test_actual_density_near_expected(self, species):
        """Receptor density inside the retinal image approximates the target density."""
        largest_patch_deg = _angular_width_deg(OBJECT_SIZE_M, DISTANCES_M[0])
        mosaic, cfg = _build_mosaic(species, largest_patch_deg)

        # Use 10m distance — large enough count for stable measurement,
        # small enough to be well within the mosaic
        scene = _scene_at_distance(species, 10.0)
        count = _count_receptors_within_extent(
            mosaic, scene.retinal_width_mm, scene.retinal_height_mm,
        )
        area_mm2 = scene.retinal_width_mm * scene.retinal_height_mm
        measured_density = count / area_mm2

        center_cone = cfg.retinal.cone_density_fn(0.0, 0.0)
        center_rod = cfg.retinal.rod_density_fn(0.0, 0.0)
        expected_density = sum(center_cone.values()) + center_rod

        rel_err = abs(measured_density - expected_density) / expected_density
        assert rel_err < 0.10, (
            f"{species}: measured density {measured_density:.0f}/mm², "
            f"expected {expected_density:.0f}/mm², rel err {rel_err:.2%}"
        )


class TestSeedStability:
    """Verify that results are reproducible and stable across seeds."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_same_seed_same_counts(self, species):
        """Same seed should give identical counts."""
        c1 = _distance_counts(species)
        c2 = _distance_counts(species)
        assert c1 == c2, f"{species}: different counts with same seed"

    @pytest.mark.parametrize("species", SPECIES)
    def test_different_seeds_similar_counts(self, species):
        """Different seeds should give similar counts (within stochastic noise)."""
        largest_patch_deg = _angular_width_deg(OBJECT_SIZE_M, DISTANCES_M[0])
        counts_by_seed = {}
        for seed in [0, 1, 2]:
            mosaic, cfg = _build_mosaic(species, largest_patch_deg, seed=seed)
            scene = _scene_at_distance(species, 10.0)
            counts_by_seed[seed] = _count_receptors_within_extent(
                mosaic, scene.retinal_width_mm, scene.retinal_height_mm,
            )

        vals = list(counts_by_seed.values())
        mean_c = np.mean(vals)
        max_dev = max(abs(v - mean_c) / mean_c for v in vals)
        assert max_dev < 0.05, (
            f"{species}: seed variance too high — counts {counts_by_seed}, max_dev {max_dev:.2%}"
        )


class TestEdgeDistances:
    """Test behaviour at extreme viewing distances."""

    @pytest.mark.parametrize("species", SPECIES)
    def test_very_near_distance(self, species):
        """At very close range (0.25m), the retinal image is large."""
        scene = _scene_at_distance(species, 0.25)
        cfg = SpeciesConfig.load(species)
        # At 0.25m, angular subtense of 20cm object ≈ 43.6°
        expected_ang = 2.0 * np.degrees(np.arctan(OBJECT_SIZE_M / (2.0 * 0.25)))
        np.testing.assert_allclose(
            scene.angular_width_deg, expected_ang, rtol=1e-10,
        )
        # Retinal image should be large
        assert scene.retinal_width_mm > 5.0, (
            f"{species}: at 0.25m, retinal width {scene.retinal_width_mm:.3f}mm too small"
        )

    @pytest.mark.parametrize("species", SPECIES)
    def test_very_far_distance(self, species):
        """At very far range (1000m), the retinal image is tiny but nonzero."""
        scene = _scene_at_distance(species, 1000.0)
        assert scene.retinal_width_mm > 0
        # At 1000m, angular subtense of 20cm ≈ 0.0115°
        expected_ang = 2.0 * np.degrees(np.arctan(OBJECT_SIZE_M / (2.0 * 1000.0)))
        np.testing.assert_allclose(
            scene.angular_width_deg, expected_ang, rtol=1e-6,
        )

    @pytest.mark.parametrize("species", SPECIES)
    def test_far_to_very_far_ratio(self, species):
        """1/d² should still hold for 100m vs 1000m."""
        s100 = _scene_at_distance(species, 100.0)
        s1000 = _scene_at_distance(species, 1000.0)
        area_100 = s100.retinal_width_mm * s100.retinal_height_mm
        area_1000 = s1000.retinal_width_mm * s1000.retinal_height_mm
        ratio = area_100 / area_1000
        np.testing.assert_allclose(ratio, 100.0, rtol=AREA_RATIO_TOL)
