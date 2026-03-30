"""Phase 3 tests: scene geometry module."""
import math
import pytest
from retinal_sim.scene.geometry import SceneGeometry, SceneDescription
from retinal_sim.optical.stage import OpticalParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optical(focal_length_mm: float, species: str = "human") -> OpticalParams:
    op = OpticalParams(
        pupil_shape="circular",
        pupil_diameter_mm=3.0,
        axial_length_mm=24.0,
        focal_length_mm=focal_length_mm,
        corneal_radius_mm=7.8,
        lca_diopters=2.0,
    )
    op._species_name = species
    return op


HUMAN = _make_optical(22.3, "human")
DOG   = _make_optical(17.0, "dog")
CAT   = _make_optical(18.5, "cat")

IMAGE_SQUARE = (100, 100)


# ---------------------------------------------------------------------------
# TestAngularSubtense
# ---------------------------------------------------------------------------

class TestAngularSubtense:
    def test_1m_at_57_3m_is_1_deg(self):
        """1 m object at 57.3 m → ~1° (within 0.01%)."""
        geom = SceneGeometry(scene_width_m=1.0, viewing_distance_m=57.3)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        expected = 2 * math.degrees(math.atan(1.0 / (2 * 57.3)))
        assert abs(desc.angular_width_deg - expected) / expected < 1e-4

    def test_collimated_is_zero(self):
        geom = SceneGeometry(scene_width_m=1.0, viewing_distance_m=float("inf"))
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.angular_width_deg == 0.0
        assert desc.angular_height_deg == 0.0

    def test_small_angle_approximation_within_10_percent_at_large_distance(self):
        """arctan formula ≈ small-angle for distant objects."""
        geom = SceneGeometry(scene_width_m=0.1, viewing_distance_m=100.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        small_angle_deg = math.degrees(0.1 / 100.0)  # w/d in radians
        assert abs(desc.angular_width_deg - small_angle_deg) / small_angle_deg < 0.01

    def test_square_scene_square_image_equal_angular_extents(self):
        geom = SceneGeometry(scene_width_m=0.2, scene_height_m=0.2,
                             viewing_distance_m=1.0)
        desc = geom.compute((200, 200), HUMAN)
        assert abs(desc.angular_width_deg - desc.angular_height_deg) < 1e-9

    def test_aspect_ratio_inferred_from_image(self):
        """scene_height_m inferred from (H/W) × scene_width_m when not provided."""
        geom = SceneGeometry(scene_width_m=0.4, viewing_distance_m=1.0)
        desc = geom.compute((100, 200), HUMAN)  # H=100, W=200 → height=0.2 m
        assert abs(desc.scene_height_m - 0.2) < 1e-9
        assert desc.angular_height_deg < desc.angular_width_deg


# ---------------------------------------------------------------------------
# TestRetinalScaling
# ---------------------------------------------------------------------------

class TestRetinalScaling:
    def test_retinal_width_proportional_to_focal_length(self):
        """Same scene → retinal width scales with focal length across species."""
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=1.0)
        human_w = geom.compute(IMAGE_SQUARE, HUMAN).retinal_width_mm
        dog_w   = geom.compute(IMAGE_SQUARE, DOG).retinal_width_mm
        cat_w   = geom.compute(IMAGE_SQUARE, CAT).retinal_width_mm
        # human > cat > dog (focal lengths: 22.3 > 18.5 > 17.0)
        assert human_w > cat_w > dog_w

    def test_retinal_width_matches_focal_length_ratio(self):
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=2.0)
        human_w = geom.compute(IMAGE_SQUARE, HUMAN).retinal_width_mm
        dog_w   = geom.compute(IMAGE_SQUARE, DOG).retinal_width_mm
        ratio = human_w / dog_w
        expected_ratio = 22.3 / 17.0
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-6

    def test_collimated_retinal_size_is_zero(self):
        geom = SceneGeometry(scene_width_m=1.0, viewing_distance_m=float("inf"))
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.retinal_width_mm == 0.0
        assert desc.retinal_height_mm == 0.0

    def test_mm_per_pixel_positive(self):
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=1.0)
        desc = geom.compute((480, 640), HUMAN)
        assert desc.mm_per_pixel[0] > 0
        assert desc.mm_per_pixel[1] > 0

    def test_mm_per_pixel_consistent_with_retinal_size(self):
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=1.0)
        desc = geom.compute((480, 640), HUMAN)
        assert abs(desc.mm_per_pixel[0] * 640 - desc.retinal_width_mm) < 1e-9
        assert abs(desc.mm_per_pixel[1] * 480 - desc.retinal_height_mm) < 1e-9


# ---------------------------------------------------------------------------
# TestAccommodation
# ---------------------------------------------------------------------------

class TestAccommodation:
    def test_collimated_zero_demand(self):
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=float("inf"))
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.accommodation_demand_diopters == 0.0
        assert desc.defocus_residual_diopters == 0.0

    def test_1m_is_1_diopter(self):
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=1.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert abs(desc.accommodation_demand_diopters - 1.0) < 1e-9

    def test_human_within_range_no_defocus(self):
        """Human (max 10D) at 0.5 m = 2D demand → no defocus."""
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=0.5)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.defocus_residual_diopters == 0.0

    def test_dog_beyond_range_has_defocus(self):
        """Dog (max 2.5D) at 0.2 m = 5D demand → 2.5D residual."""
        geom = SceneGeometry(scene_width_m=0.1, viewing_distance_m=0.2)
        desc = geom.compute(IMAGE_SQUARE, DOG)
        assert abs(desc.defocus_residual_diopters - 2.5) < 1e-9

    def test_defocus_never_negative(self):
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=100.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.defocus_residual_diopters >= 0.0


# ---------------------------------------------------------------------------
# TestPatchClipping
# ---------------------------------------------------------------------------

class TestPatchClipping:
    def test_small_scene_not_clipped(self):
        """0.01 m at 1 m ≈ 0.57° < 2° PoC patch → not clipped."""
        geom = SceneGeometry(scene_width_m=0.01, viewing_distance_m=1.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert not desc.clipped

    def test_large_scene_clipped(self):
        """0.5 m at 1 m ≈ 28° > 2° PoC patch → clipped."""
        geom = SceneGeometry(scene_width_m=0.5, viewing_distance_m=1.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.clipped

    def test_scene_covers_fraction_leq_1(self):
        geom = SceneGeometry(scene_width_m=0.5, viewing_distance_m=1.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert desc.scene_covers_patch_fraction <= 1.0

    def test_small_scene_fraction_lt_1(self):
        geom = SceneGeometry(scene_width_m=0.001, viewing_distance_m=1.0)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        assert 0.0 <= desc.scene_covers_patch_fraction < 1.0


# ---------------------------------------------------------------------------
# TestValidation  (architecture §11e criteria)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_angular_subtense_accuracy(self):
        """Architecture §11e: 1m at 57.3m → 1° within 0.01%."""
        geom = SceneGeometry(scene_width_m=1.0, viewing_distance_m=57.3)
        desc = geom.compute(IMAGE_SQUARE, HUMAN)
        expected = 2 * math.degrees(math.atan(0.5 / 57.3))
        rel_err = abs(desc.angular_width_deg - expected) / expected
        assert rel_err < 1e-4, f"Relative error {rel_err:.2e} exceeds 0.01%"

    def test_retinal_scaling_human_gt_cat_gt_dog(self):
        """Architecture §11e: retinal size human > cat > dog."""
        geom = SceneGeometry(scene_width_m=0.2, viewing_distance_m=1.0)
        h = geom.compute(IMAGE_SQUARE, HUMAN).retinal_width_mm
        c = geom.compute(IMAGE_SQUARE, CAT).retinal_width_mm
        d = geom.compute(IMAGE_SQUARE, DOG).retinal_width_mm
        assert h > c > d

    def test_receptor_count_scales_inverse_square_distance(self):
        """Architecture §11e: mm_per_pixel ∝ 1/d, so receptor count ∝ 1/d²."""
        geom1 = SceneGeometry(scene_width_m=1.0, viewing_distance_m=1.0)
        geom2 = SceneGeometry(scene_width_m=1.0, viewing_distance_m=2.0)
        d1 = geom1.compute(IMAGE_SQUARE, HUMAN)
        d2 = geom2.compute(IMAGE_SQUARE, HUMAN)
        # retinal area ∝ (mm_per_pixel)² × N_pixels → at fixed pixels, area ∝ retinal size²
        area_ratio = (d1.retinal_width_mm * d1.retinal_height_mm) / \
                     (d2.retinal_width_mm * d2.retinal_height_mm)
        # For large distances (small angle approx), area ratio ≈ 4.0 = (2/1)²
        # For exact arctan: slightly less; tolerance 5%
        assert abs(area_ratio - 4.0) / 4.0 < 0.05


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            SceneGeometry(scene_width_m=-1.0, viewing_distance_m=1.0)

    def test_zero_distance_raises(self):
        with pytest.raises(ValueError):
            SceneGeometry(scene_width_m=1.0, viewing_distance_m=0.0)
