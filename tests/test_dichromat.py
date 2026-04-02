"""Phase 10 — Dichromat confusion validation tests.

Tests are grouped into three classes:

TestIshiharaPattern
    Unit tests for ``make_dot_pattern``: shape, dtype, mask geometry,
    colour placement, and seed reproducibility.

TestFindConfusionPair
    Unit tests for ``find_confusion_pair``: return format, spectral
    response similarity for the species, and M-cone contrast for human.

TestDichromatDiscriminability
    Integration tests running the full pipeline.  Key assertions:

    * For a confusion pair (dog confusion axis): human D >> dog D.
    * For an easy chromatic pair (blue vs yellow, detectable by both):
      both human and dog show non-trivial D.
    * Convenience method ``is_confused`` tracks ``discriminability``.
    * Parameter variations (n_seeds, patch size, image size) do not
      crash and return sensible values.
"""
from __future__ import annotations

import numpy as np
import pytest

from retinal_sim.constants import WAVELENGTHS
from retinal_sim.retina.opsin import LAMBDA_MAX, govardovskii_a1
from retinal_sim.spectral.upsampler import SpectralUpsampler
from retinal_sim.validation.dichromat import DichromatValidator
from retinal_sim.validation.ishihara import find_confusion_pair, make_dot_pattern

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def confusion_pair():
    """Dog confusion pair — computed once for all tests that need it."""
    return find_confusion_pair(species="dog", n_candidates=400, seed=7)


@pytest.fixture(scope="module")
def blue_yellow():
    """An easily discriminable pair: blue vs yellow.

    Both human and dog can distinguish these by S (blue) vs L (yellow) cone
    contrast — blue strongly activates S_dog (429 nm), yellow does not.
    """
    blue = np.array([30, 80, 220], dtype=np.uint8)
    yellow = np.array([220, 210, 20], dtype=np.uint8)
    return blue, yellow


# ===========================================================================
# TestIshiharaPattern
# ===========================================================================

class TestIshiharaPattern:
    """Unit tests for make_dot_pattern."""

    def test_output_shape(self):
        img, mask = make_dot_pattern(
            np.array([255, 100, 0], dtype=np.uint8),
            np.array([0, 150, 80], dtype=np.uint8),
            image_size_px=32,
        )
        assert img.shape == (32, 32, 3)
        assert mask.shape == (32, 32)

    def test_output_dtype(self):
        img, mask = make_dot_pattern(
            np.array([200, 80, 10], dtype=np.uint8),
            np.array([60, 160, 10], dtype=np.uint8),
            image_size_px=32,
        )
        assert img.dtype == np.uint8
        assert mask.dtype == bool

    def test_figure_mask_is_central_disc(self):
        H = W = 64
        _, mask = make_dot_pattern(
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 255, 0], dtype=np.uint8),
            image_size_px=H,
        )
        # Centre pixel must be in figure.
        assert mask[H // 2, W // 2]
        # Corners must be outside figure.
        for corner in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
            assert not mask[corner]

    def test_figure_mask_fraction(self):
        """Figure disc ~= π * 0.30² ≈ 28 % of image area (allow 5 pp margin)."""
        _, mask = make_dot_pattern(
            np.array([255, 0, 0], dtype=np.uint8),
            np.array([0, 255, 0], dtype=np.uint8),
            image_size_px=128,
        )
        frac = mask.sum() / mask.size
        assert 0.23 <= frac <= 0.33

    def test_fg_color_dominant_in_figure(self):
        """Mean red channel inside figure should be close to fg_rgb[0]."""
        fg = np.array([200, 50, 10], dtype=np.uint8)
        bg = np.array([10, 50, 200], dtype=np.uint8)
        img, mask = make_dot_pattern(fg, bg, image_size_px=64, n_dots=20, seed=0)
        mean_r_inside = img[mask, 0].mean()
        # With 15 % noise, expect mean to be > midpoint between fg and bg reds.
        assert mean_r_inside > (int(fg[0]) + int(bg[0])) / 2

    def test_bg_color_dominant_in_background(self):
        """Mean blue channel outside figure should be close to bg_rgb[2]."""
        fg = np.array([200, 50, 10], dtype=np.uint8)
        bg = np.array([10, 50, 200], dtype=np.uint8)
        img, mask = make_dot_pattern(fg, bg, image_size_px=64, n_dots=20, seed=0)
        mean_b_outside = img[~mask, 2].mean()
        assert mean_b_outside > (int(fg[2]) + int(bg[2])) / 2

    def test_seed_reproducibility(self):
        fg = np.array([200, 80, 10], dtype=np.uint8)
        bg = np.array([60, 160, 10], dtype=np.uint8)
        img1, _ = make_dot_pattern(fg, bg, image_size_px=32, seed=42)
        img2, _ = make_dot_pattern(fg, bg, image_size_px=32, seed=42)
        np.testing.assert_array_equal(img1, img2)

    def test_different_seeds_differ(self):
        fg = np.array([200, 80, 10], dtype=np.uint8)
        bg = np.array([60, 160, 10], dtype=np.uint8)
        img1, _ = make_dot_pattern(fg, bg, image_size_px=32, seed=0)
        img2, _ = make_dot_pattern(fg, bg, image_size_px=32, seed=1)
        assert not np.array_equal(img1, img2)

    def test_no_dots_produces_solid_regions(self):
        """With n_dots=0 the image is exactly two solid colour regions."""
        fg = np.array([200, 80, 10], dtype=np.uint8)
        bg = np.array([60, 160, 10], dtype=np.uint8)
        img, mask = make_dot_pattern(fg, bg, image_size_px=32, n_dots=0)
        # img[mask] has shape (N_pix, 3); compare each pixel row to fg.
        assert (img[mask] == fg).all()
        assert (img[~mask] == bg).all()

    def test_pixel_values_in_uint8_range(self):
        fg = np.array([250, 250, 250], dtype=np.uint8)
        bg = np.array([5, 5, 5], dtype=np.uint8)
        img, _ = make_dot_pattern(fg, bg, image_size_px=32, n_dots=50)
        assert img.min() >= 0
        assert img.max() <= 255


# ===========================================================================
# TestFindConfusionPair
# ===========================================================================

class TestFindConfusionPair:
    """Unit tests for find_confusion_pair."""

    def test_returns_two_uint8_arrays(self, confusion_pair):
        fg, bg = confusion_pair
        assert fg.dtype == np.uint8
        assert bg.dtype == np.uint8

    def test_output_shape(self, confusion_pair):
        fg, bg = confusion_pair
        assert fg.shape == (3,)
        assert bg.shape == (3,)

    def test_values_in_valid_rgb_range(self, confusion_pair):
        for arr in confusion_pair:
            assert arr.min() >= 0
            assert arr.max() <= 255

    def test_low_blue_components(self, confusion_pair):
        """Search constrains B ≤ 50 to target the red-green axis."""
        fg, bg = confusion_pair
        assert int(fg[2]) <= 50
        assert int(bg[2]) <= 50

    def test_fg_is_warmer_than_bg(self, confusion_pair):
        """Convention: fg has higher R channel."""
        fg, bg = confusion_pair
        assert int(fg[0]) >= int(bg[0])

    def test_deterministic(self):
        fg1, bg1 = find_confusion_pair(seed=99)
        fg2, bg2 = find_confusion_pair(seed=99)
        np.testing.assert_array_equal(fg1, fg2)
        np.testing.assert_array_equal(bg1, bg2)

    def test_human_raises(self):
        with pytest.raises(ValueError, match="trichromat"):
            find_confusion_pair(species="human")

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            find_confusion_pair(species="fish")

    def test_dog_l_cone_responses_similar(self, confusion_pair):
        """After upsampling, dog-L responses for fg and bg should be close."""
        fg, bg = confusion_pair
        upsampler = SpectralUpsampler()
        wl = WAVELENGTHS.astype(float)
        dlam = float(np.mean(np.diff(wl)))
        L_dog = govardovskii_a1(LAMBDA_MAX["dog"]["L_cone"], wl)

        def l_response(rgb):
            spectral = upsampler.upsample(rgb[np.newaxis, np.newaxis, :])
            spec = spectral.data[0, 0, :].astype(float)
            return float(spec @ L_dog) * dlam

        l_fg = l_response(fg)
        l_bg = l_response(bg)
        # Normalised difference in dog-L should be small.
        norm_diff = abs(l_fg - l_bg) / max(l_fg, l_bg, 1e-12)
        assert norm_diff < 0.20, (
            f"Dog L-cone responses too different: fg={l_fg:.4f}, bg={l_bg:.4f}"
        )

    def test_human_m_cone_responses_differ(self, confusion_pair):
        """Human M-cone responses should differ noticeably for the pair."""
        fg, bg = confusion_pair
        upsampler = SpectralUpsampler()
        wl = WAVELENGTHS.astype(float)
        dlam = float(np.mean(np.diff(wl)))
        M_human = govardovskii_a1(LAMBDA_MAX["human"]["M_cone"], wl)

        def m_response(rgb):
            spectral = upsampler.upsample(rgb[np.newaxis, np.newaxis, :])
            spec = spectral.data[0, 0, :].astype(float)
            return float(spec @ M_human) * dlam

        m_fg = m_response(fg)
        m_bg = m_response(bg)
        norm_diff = abs(m_fg - m_bg) / max(m_fg, m_bg, 1e-12)
        assert norm_diff > 0.10, (
            f"Human M-cone responses not different enough: fg={m_fg:.4f}, bg={m_bg:.4f}"
        )

    def test_cat_confusion_pair(self):
        """find_confusion_pair works for cat as well."""
        fg, bg = find_confusion_pair(species="cat", seed=5)
        assert fg.dtype == np.uint8
        assert bg.shape == (3,)


# ===========================================================================
# TestDichromatDiscriminability
# ===========================================================================

class TestDichromatDiscriminability:
    """Integration tests: full pipeline discriminability for human and dog."""

    # --- Constructor / parameter tests ------------------------------------

    def test_constructor_human(self):
        v = DichromatValidator("human", n_seeds=1)
        assert v._species_name == "human"

    def test_constructor_dog(self):
        v = DichromatValidator("dog", n_seeds=1)
        assert v._species_name == "dog"

    def test_discriminability_returns_float(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("dog", seed=0, n_seeds=1)
        d = v.discriminability(fg, bg, patch_size_deg=2.0, image_size_px=32)
        assert isinstance(d, float)

    def test_discriminability_in_unit_interval(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("human", seed=0, n_seeds=1)
        d = v.discriminability(fg, bg, patch_size_deg=2.0, image_size_px=32)
        assert 0.0 <= d <= 1.0

    def test_identical_colors_give_zero(self):
        """Same fg and bg → no figure signal → D ≈ 0."""
        gray = np.array([128, 128, 128], dtype=np.uint8)
        v = DichromatValidator("human", seed=0, n_seeds=1)
        d = v.discriminability(gray, gray, patch_size_deg=2.0, image_size_px=32)
        assert d < 0.05

    def test_n_seeds_1_compatible(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("dog", seed=0, n_seeds=1)
        d = v.discriminability(fg, bg, patch_size_deg=2.0, image_size_px=32)
        assert 0.0 <= d <= 1.0

    def test_n_seeds_2_compatible(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("dog", seed=0, n_seeds=2)
        d = v.discriminability(fg, bg, patch_size_deg=2.0, image_size_px=32)
        assert 0.0 <= d <= 1.0

    def test_larger_image_compatible(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("human", seed=0, n_seeds=1)
        d = v.discriminability(fg, bg, patch_size_deg=2.0, image_size_px=64)
        assert 0.0 <= d <= 1.0

    def test_small_patch_compatible(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("dog", seed=0, n_seeds=1)
        d = v.discriminability(fg, bg, patch_size_deg=1.0, image_size_px=32)
        assert 0.0 <= d <= 1.0

    # --- Key validation: confusion pair -----------------------------------

    def test_human_discriminability_exceeds_dog(self, confusion_pair):
        """Human D should exceed dog D for the confusion pair.

        This is the primary Phase 10 validation: the confusion pair lies on
        the dog's dichromatic confusion axis, so the dog is poor at detecting
        the figure while the human sees it clearly.
        """
        fg, bg = confusion_pair
        d_human = DichromatValidator("human", seed=0, n_seeds=2).discriminability(
            fg, bg, patch_size_deg=2.0, image_size_px=48
        )
        d_dog = DichromatValidator("dog", seed=0, n_seeds=2).discriminability(
            fg, bg, patch_size_deg=2.0, image_size_px=48
        )
        assert d_human > d_dog, (
            f"Expected human D ({d_human:.3f}) > dog D ({d_dog:.3f}) "
            "for confusion pair."
        )

    def test_human_discriminability_is_non_trivial(self, confusion_pair):
        """Human should reliably detect the figure (D > 0.01).

        The absolute value is small because the confusion pair is designed to
        be near-isochromatic for many cone types; only human M-cone provides
        the chromatic signal.  The key biological result is the ratio
        D_human / D_dog >> 1 (tested separately).
        """
        fg, bg = confusion_pair
        d = DichromatValidator("human", seed=0, n_seeds=2).discriminability(
            fg, bg, patch_size_deg=2.0, image_size_px=48
        )
        assert d > 0.01, (
            f"Human discriminability too low: D={d:.4f} (expected > 0.01)"
        )

    # --- Easy pair: blue vs yellow, detectable by both --------------------

    def test_blue_yellow_human_discriminates(self, blue_yellow):
        """Human should easily detect blue vs yellow figure (D > 0.05)."""
        blue, yellow = blue_yellow
        d = DichromatValidator("human", seed=0, n_seeds=2).discriminability(
            blue, yellow, patch_size_deg=2.0, image_size_px=48
        )
        assert d > 0.05, f"Human D={d:.3f} too low for blue vs yellow"

    def test_blue_yellow_dog_discriminates(self, blue_yellow):
        """Dog should also detect blue vs yellow (different S_dog responses)."""
        blue, yellow = blue_yellow
        d = DichromatValidator("dog", seed=0, n_seeds=2).discriminability(
            blue, yellow, patch_size_deg=2.0, image_size_px=48
        )
        assert d > 0.05, f"Dog D={d:.3f} too low for blue vs yellow"

    # --- is_confused convenience method -----------------------------------

    def test_is_confused_returns_bool(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("dog", seed=0, n_seeds=1)
        result = v.is_confused(fg, bg, patch_size_deg=2.0, image_size_px=32)
        assert isinstance(result, bool)

    def test_is_confused_consistent_with_discriminability(self, confusion_pair):
        fg, bg = confusion_pair
        v = DichromatValidator("dog", seed=0, n_seeds=1)
        d = v.discriminability(fg, bg, patch_size_deg=2.0, image_size_px=32)
        confused = v.is_confused(fg, bg, patch_size_deg=2.0, image_size_px=32)
        # is_confused uses threshold 0.10; check consistency.
        if d < 0.10:
            assert confused
        else:
            assert not confused
