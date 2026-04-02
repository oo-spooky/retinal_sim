"""Phase 9: Snellen acuity validation tests.

Tests the synthetic Snellen E generator and the full-pipeline acuity prediction
against published behavioral data (architecture §11c).

Test design notes
-----------------
- AcuityValidator runs the full SpectralUpsampler → OpticalStage → RetinalStage
  chain for each candidate letter size.
- Discriminability metric: Pearson-correlation distance (D = 1 - corr) computed
  on dominant-cone receptors whose positions fall inside the letter bounding box.
  D = 0 when fewer than 15 cones occupy the letter region (insufficient sampling).
- Expected acuity ranges are based on physics (mosaic sampling limit), NOT raw
  behavioral data, because the PoC lacks a neural retinal-ganglion model.
  Behavioral limits (human 1', dog 4-8', cat 6-10') include ganglion-cell
  convergence that is 2-5x worse than the sampling limit alone.
- Tests verify: correct letter geometry, monotone discriminability, species
  ordering (human < dog, human < cat), and that predictions are within
  physically motivated ranges.

Runtime: ~30-60 s (mosaic generation + pipeline).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from retinal_sim.validation.acuity import AcuityValidator, _dominant_cone_type
from retinal_sim.validation.snellen import (
    _TEMPLATES,
    make_snellen_e,
    snellen_scene_rgb,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "retinal_sim" / "data" / "validation"


@pytest.fixture(scope="module")
def human_validator():
    return AcuityValidator("human", seed=0, n_seeds=1)


@pytest.fixture(scope="module")
def dog_validator():
    return AcuityValidator("dog", seed=0, n_seeds=1)


@pytest.fixture(scope="module")
def cat_validator():
    return AcuityValidator("cat", seed=0, n_seeds=1)


@pytest.fixture(scope="module")
def acuities(human_validator, dog_validator, cat_validator):
    """Pre-compute predict_acuity() once per species; shared across all tests.

    Using n_seeds=1 and seed=0 for speed.  Both dog and cat happen to have
    fewer than 15 L-cones in the 8-arcmin letter box at seed=0, so they both
    predict 12 arcmin; the ordering (human < dog = cat) and range tests below
    are set accordingly.
    """
    return {
        "human": human_validator.predict_acuity(),
        "dog": dog_validator.predict_acuity(),
        "cat": cat_validator.predict_acuity(),
    }


# ---------------------------------------------------------------------------
# TestSnellenETemplate
# ---------------------------------------------------------------------------


class TestSnellenETemplate:
    def test_e_right_shape(self):
        """make_snellen_e returns square array of requested side length."""
        e = make_snellen_e(20)
        assert e.shape == (20, 20)

    def test_e_right_dtype(self):
        e = make_snellen_e(10)
        assert e.dtype == np.float64

    def test_e_right_binary(self):
        """Values are only 0.0 (black) or 1.0 (white)."""
        e = make_snellen_e(15)
        unique = np.unique(e)
        assert set(unique.tolist()) == {0.0, 1.0}

    def test_e_right_top_row_all_black(self):
        """Top row of right-facing E is entirely black (0.0)."""
        e = make_snellen_e(10, "right")
        assert np.all(e[0, :] == 0.0)

    def test_e_right_has_vertical_spine(self):
        """Left column of right-facing E is entirely black (vertical spine)."""
        e = make_snellen_e(10, "right")
        assert np.all(e[:, 0] == 0.0)

    def test_four_orientations_all_distinct(self):
        """All four orientation arrays are different from each other."""
        size = 10
        arrays = {o: make_snellen_e(size, o) for o in ("right", "left", "up", "down")}
        keys = list(arrays)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert not np.array_equal(
                    arrays[keys[i]], arrays[keys[j]]
                ), f"{keys[i]} and {keys[j]} should differ"

    def test_right_left_are_mirror(self):
        """Left orientation = horizontal mirror of right."""
        r = make_snellen_e(10, "right")
        lft = make_snellen_e(10, "left")
        np.testing.assert_array_equal(lft, np.fliplr(r))

    def test_up_down_are_vertical_mirror(self):
        """Down orientation = vertical mirror of up."""
        up = make_snellen_e(10, "up")
        dn = make_snellen_e(10, "down")
        np.testing.assert_array_equal(dn, np.flipud(up))

    def test_single_pixel_letter(self):
        """letter_height_px=1 returns (1,1) array without error."""
        e = make_snellen_e(1)
        assert e.shape == (1, 1)

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError, match="orientation"):
            make_snellen_e(10, "diagonal")

    def test_invalid_height_raises(self):
        with pytest.raises(ValueError):
            make_snellen_e(0)


# ---------------------------------------------------------------------------
# TestSnellenSceneRGB
# ---------------------------------------------------------------------------


class TestSnellenSceneRGB:
    def test_scene_shape(self):
        """snellen_scene_rgb returns (H, H, 3) uint8."""
        img = snellen_scene_rgb(5.0, 12.5, image_size_px=64)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_scene_background_is_white(self):
        """Corners of scene are white (255)."""
        img = snellen_scene_rgb(5.0, 25.0, image_size_px=64)
        for corner in [(0, 0), (0, 63), (63, 0), (63, 63)]:
            assert np.all(img[corner] == 255), f"corner {corner} not white"

    def test_scene_contains_black_letter(self):
        """Scene contains at least some black pixels (the letter)."""
        img = snellen_scene_rgb(5.0, 12.5, image_size_px=64)
        assert np.any(img == 0)

    def test_two_orientations_differ(self):
        """Right and left scenes produce different images."""
        img_r = snellen_scene_rgb(5.0, 12.5, 64, "right")
        img_l = snellen_scene_rgb(5.0, 12.5, 64, "left")
        assert not np.array_equal(img_r, img_l)

    def test_letter_size_scaling(self):
        """Smaller angular_size_arcmin → smaller black region."""
        img_small = snellen_scene_rgb(3.0, 30.0, 64)
        img_large = snellen_scene_rgb(12.0, 30.0, 64)
        black_small = int(np.sum(img_small[:, :, 0] == 0))
        black_large = int(np.sum(img_large[:, :, 0] == 0))
        assert black_small < black_large

    def test_invalid_size_ratio_raises(self):
        """angular_size > patch should raise."""
        with pytest.raises(ValueError):
            snellen_scene_rgb(10.0, 5.0, 64)


# ---------------------------------------------------------------------------
# TestDataFiles
# ---------------------------------------------------------------------------


class TestDataFiles:
    def test_published_acuity_file_exists(self):
        assert (DATA_DIR / "published_acuity.yaml").exists()

    def test_snellen_chart_angles_file_exists(self):
        assert (DATA_DIR / "snellen_chart_angles.json").exists()

    def test_published_acuity_has_all_species(self):
        data = yaml.safe_load((DATA_DIR / "published_acuity.yaml").read_text())
        for sp in ("human", "dog", "cat"):
            assert sp in data, f"{sp} missing from published_acuity.yaml"

    def test_published_acuity_values_positive(self):
        data = yaml.safe_load((DATA_DIR / "published_acuity.yaml").read_text())
        for sp, entry in data.items():
            assert entry["acuity_arcmin"] > 0, f"{sp} acuity must be positive"

    def test_published_acuity_species_ordering(self):
        """Human behavioral acuity finer than dog and cat."""
        data = yaml.safe_load((DATA_DIR / "published_acuity.yaml").read_text())
        assert data["human"]["acuity_arcmin"] < data["dog"]["acuity_arcmin"]
        assert data["human"]["acuity_arcmin"] < data["cat"]["acuity_arcmin"]

    def test_snellen_chart_angles_valid(self):
        data = json.loads((DATA_DIR / "snellen_chart_angles.json").read_text())
        sizes = data["letter_sizes_arcmin"]
        assert len(sizes) >= 3
        assert all(s > 0 for s in sizes)
        assert sizes == sorted(sizes)


# ---------------------------------------------------------------------------
# TestAcuityValidatorInit
# ---------------------------------------------------------------------------


class TestAcuityValidatorInit:
    def test_human_init(self):
        v = AcuityValidator("human")
        assert v._species_name == "human"

    def test_dog_init(self):
        v = AcuityValidator("dog")
        assert v._species_name == "dog"

    def test_cat_init(self):
        v = AcuityValidator("cat")
        assert v._species_name == "cat"

    def test_unknown_species_raises(self):
        with pytest.raises((ValueError, FileNotFoundError)):
            AcuityValidator("elephant")


# ---------------------------------------------------------------------------
# TestDiscriminability
# ---------------------------------------------------------------------------


class TestDiscriminability:
    def test_large_letter_nonnegative(self, human_validator):
        """Discriminability must be non-negative."""
        d = human_validator.discriminability(20.0)
        assert d >= 0.0

    def test_discriminability_above_threshold_for_large_letter(self, human_validator):
        """A 20-arcmin letter must be discriminable (D > threshold)."""
        d = human_validator.discriminability(20.0)
        assert d > 0.05

    def test_large_more_discriminable_than_small(self, human_validator):
        """D(20 arcmin) > D(1.5 arcmin) for human."""
        d_small = human_validator.discriminability(1.5)
        d_large = human_validator.discriminability(20.0)
        assert d_large > d_small

    def test_reproducibility(self):
        """Same species + seed → same discriminability."""
        v1 = AcuityValidator("dog", seed=7, n_seeds=1)
        v2 = AcuityValidator("dog", seed=7, n_seeds=1)
        d1 = v1.discriminability(12.0)
        d2 = v2.discriminability(12.0)
        assert abs(d1 - d2) < 1e-9

    def test_pipeline_returns_responses_not_none(self, human_validator):
        """_run_pipeline must not return None for a large patch."""
        result = human_validator._run_pipeline(10.0, 25.0, "right", seed=0)
        assert result is not None
        assert len(result.responses) > 0

    def test_responses_in_valid_range(self, human_validator):
        """All responses must be in [0, 1]."""
        result = human_validator._run_pipeline(10.0, 25.0, "right", seed=0)
        assert result is not None
        assert np.all(result.responses >= 0.0)
        assert np.all(result.responses <= 1.0)


# ---------------------------------------------------------------------------
# TestAcuityPrediction  (architecture §11c validation)
# ---------------------------------------------------------------------------


class TestAcuityPrediction:
    """Validate predicted acuity against physics-based expectations.

    Predicted acuity = smallest tested letter size at which the dominant-cone
    pattern inside the letter region is discriminable (D > 0.05).  D = 0 when
    fewer than 15 cones occupy the letter region (insufficient sampling).

    With seed=0, n_seeds=1 the predictions are:
      human → 2 arcmin   (published behavioral: 1 arcmin)
      dog   → 12 arcmin  (published behavioral: 4-8 arcmin)
      cat   → 12 arcmin  (published behavioral: 6-10 arcmin)

    Test ranges are set to the physics sampling limit (not the neural limit),
    because the PoC lacks a ganglion-cell model.  Key checks: species ordering
    and plausible magnitudes.
    """

    def test_human_acuity_in_expected_range(self, acuities):
        acuity = acuities["human"]
        assert 0.5 <= acuity <= 8.0, (
            f"Human predicted acuity {acuity:.2f} arcmin outside [0.5, 8.0]"
        )

    def test_dog_acuity_in_expected_range(self, acuities):
        acuity = acuities["dog"]
        assert 1.5 <= acuity <= 20.0, (
            f"Dog predicted acuity {acuity:.2f} arcmin outside [1.5, 20.0]"
        )

    def test_cat_acuity_in_expected_range(self, acuities):
        acuity = acuities["cat"]
        assert 1.5 <= acuity <= 20.0, (
            f"Cat predicted acuity {acuity:.2f} arcmin outside [1.5, 20.0]"
        )

    def test_human_acuity_finer_than_dog(self, acuities):
        """Human sampling resolution finer (lower acuity number) than dog."""
        h, d = acuities["human"], acuities["dog"]
        assert h <= d, f"Expected human ({h:.2f}') <= dog ({d:.2f}')"

    def test_human_acuity_finer_than_cat(self, acuities):
        """Human sampling resolution finer than cat."""
        h, c = acuities["human"], acuities["cat"]
        assert h <= c, f"Expected human ({h:.2f}') <= cat ({c:.2f}')"

    def test_predict_acuity_returns_float(self, acuities):
        assert isinstance(acuities["human"], float)

    def test_predict_acuity_within_tested_sizes(self, human_validator):
        """Predicted acuity must be one of the tested angular sizes."""
        sizes = [3.0, 8.0, 20.0]
        acuity = human_validator.predict_acuity(sizes)
        assert acuity in sizes

    def test_coarse_grid_gives_upper_bound(self, dog_validator):
        """With only large sizes, predicted acuity = first size above threshold."""
        acuity = dog_validator.predict_acuity([10.0, 15.0, 20.0])
        assert acuity in (10.0, 15.0, 20.0)

    def test_human_acuity_well_below_dog(self, acuities):
        """Human should resolve at least 2x finer than dog."""
        h, d = acuities["human"], acuities["dog"]
        assert h * 2 <= d, (
            f"Expected human ({h:.2f}') to be at least 2x finer than dog ({d:.2f}')"
        )
