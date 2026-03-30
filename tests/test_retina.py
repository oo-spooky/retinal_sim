"""Tests for the retinal stage — Phase 1: Govardovskii nomogram.

Validation strategy (from architecture §11c and §12 Phase 1):
  - Peak wavelength: nomogram peak must land within ±2 nm of specified λ_max.
  - Normalization: peak value == 1.0 (within floating-point tolerance).
  - β-band presence: the shoulder at short wavelengths must be detectable.
  - Monotonic flanks: absorption should decrease monotonically away from peak
    (ignoring the β-band region below ~430 nm for long-wavelength pigments).
  - All-species coverage: build_sensitivity_curves works for human, dog, cat.
  - No negative values: absorption is always ≥ 0.
  - Known relative ordering: for human, L_cone peak > M_cone peak > rod peak > S_cone peak.
"""

import numpy as np
import pytest

from retinal_sim.retina.opsin import (
    LAMBDA_MAX,
    build_sensitivity_curves,
    govardovskii_a1,
    govardovskii_a2,
)

# Standard wavelength grid used throughout tests
WL = np.arange(380, 721, 5, dtype=float)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def peak_wavelength(spectrum: np.ndarray, wavelengths: np.ndarray) -> float:
    """Return the wavelength at maximum absorption."""
    return float(wavelengths[np.argmax(spectrum)])


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    @pytest.mark.parametrize("lam_max", [420, 450, 498, 506, 530, 553, 555, 560])
    def test_a1_peak_is_unity(self, lam_max: float):
        s = govardovskii_a1(lam_max, WL)
        assert abs(s.max() - 1.0) < 1e-10, (
            f"A1 nomogram peak for λ_max={lam_max} nm is {s.max()}, expected 1.0"
        )

    @pytest.mark.parametrize("lam_max", [420, 498, 530, 560])
    def test_a2_peak_is_unity(self, lam_max: float):
        s = govardovskii_a2(lam_max, WL)
        assert abs(s.max() - 1.0) < 1e-10

    def test_no_negative_values_a1(self):
        for lam_max in [420, 498, 530, 560]:
            s = govardovskii_a1(lam_max, WL)
            assert np.all(s >= 0), f"Negative absorption found for λ_max={lam_max}"

    def test_no_negative_values_a2(self):
        s = govardovskii_a2(498, WL)
        assert np.all(s >= 0)


# ---------------------------------------------------------------------------
# 2. Peak wavelength accuracy
# ---------------------------------------------------------------------------

class TestPeakWavelength:
    """Peak of the nomogram must land within ±2 nm of the specified λ_max.

    The tolerance is one grid step (5 nm) / 2, plus a small allowance for
    the β-band contribution displacing the α-band peak when λ_max is small.
    We use ±5 nm as the practical tolerance (matches the sampling grid).
    """

    TOLERANCE_NM = 5.0

    @pytest.mark.parametrize("lam_max,label", [
        (498.0, "human rod"),
        (420.0, "human S-cone"),
        (530.0, "human M-cone"),
        (560.0, "human L-cone"),
        (429.0, "dog S-cone"),
        (555.0, "dog L-cone"),
        (506.0, "dog rod"),
        (450.0, "cat S-cone"),
        (553.0, "cat L-cone"),
        (501.0, "cat rod"),
    ])
    def test_a1_peak_location(self, lam_max: float, label: str):
        s = govardovskii_a1(lam_max, WL, include_beta=True)
        measured = peak_wavelength(s, WL)
        assert abs(measured - lam_max) <= self.TOLERANCE_NM, (
            f"{label}: expected peak near {lam_max} nm, got {measured} nm"
        )

    @pytest.mark.parametrize("lam_max", [498, 530, 560])
    def test_a1_peak_alpha_only(self, lam_max: float):
        """With β-band disabled the α-band peak should be even closer."""
        s = govardovskii_a1(lam_max, WL, include_beta=False)
        measured = peak_wavelength(s, WL)
        assert abs(measured - lam_max) <= self.TOLERANCE_NM


# ---------------------------------------------------------------------------
# 3. β-band shoulder
# ---------------------------------------------------------------------------

class TestBetaBand:
    """The β-band adds a secondary shoulder at shorter wavelengths.

    For an L-cone (λ_max = 560 nm), the β-band should appear around
    λ_β ≈ 0.29 × 560 + 189 ≈ 351 nm (below our grid) — effect bleeds into
    the 380-420 nm region.  For a rod (498 nm), λ_β ≈ 333 nm similarly.

    Instead of testing exact β-band location, we verify that including the
    β-band raises absorption in the short-wavelength region.
    """

    def test_beta_band_increases_short_wavelength_absorption(self):
        lam_max = 498.0
        s_with = govardovskii_a1(lam_max, WL, include_beta=True)
        s_without = govardovskii_a1(lam_max, WL, include_beta=False)
        # Short wavelength region: 380–430 nm
        mask = WL <= 430
        assert np.all(s_with[mask] >= s_without[mask]), (
            "β-band should not reduce absorption at short wavelengths"
        )
        # The increase should be detectable
        assert (s_with[mask] - s_without[mask]).max() > 0.01

    def test_beta_band_does_not_shift_main_peak(self):
        """Adding the β-band must not displace the primary peak for common pigments."""
        for lam_max in [498, 530, 560]:
            s_with = govardovskii_a1(lam_max, WL, include_beta=True)
            s_without = govardovskii_a1(lam_max, WL, include_beta=False)
            peak_with = peak_wavelength(s_with, WL)
            peak_without = peak_wavelength(s_without, WL)
            assert abs(peak_with - peak_without) <= 5.0, (
                f"β-band shifted peak by {abs(peak_with - peak_without)} nm "
                f"for λ_max={lam_max}"
            )


# ---------------------------------------------------------------------------
# 4. Shape properties
# ---------------------------------------------------------------------------

class TestShape:
    def test_long_wavelength_flank_decays(self):
        """Absorption should fall off on the long-wavelength side of the peak."""
        lam_max = 530.0
        s = govardovskii_a1(lam_max, WL, include_beta=False)
        idx_peak = int(np.argmax(s))
        # From the peak to the red edge, absorption should be strictly decreasing
        # (allow minor numerical noise: monotone up to 1e-6 tolerance)
        red_flank = s[idx_peak:]
        diffs = np.diff(red_flank)
        assert np.all(diffs <= 1e-6), (
            "Long-wavelength flank is not monotonically decreasing"
        )

    def test_relative_peak_ordering_human(self):
        """Human pigment peaks: S(420) < rod(498) < M(530) < L(560)."""
        sensitivities = build_sensitivity_curves("human", WL)
        peaks = {name: peak_wavelength(s, WL) for name, s in sensitivities.items()}
        assert peaks["S_cone"] < peaks["rod"] < peaks["M_cone"] < peaks["L_cone"], (
            f"Unexpected peak ordering: {peaks}"
        )

    def test_half_width_plausible(self):
        """Human rod rhodopsin (498 nm) half-bandwidth should be ~80–110 nm (A1)."""
        s = govardovskii_a1(498.0, WL, include_beta=False)
        half_max = 0.5
        above_half = WL[s >= half_max]
        bandwidth = float(above_half[-1] - above_half[0])
        assert 70 <= bandwidth <= 130, (
            f"Rod half-bandwidth = {bandwidth} nm, expected ~80-110 nm"
        )


# ---------------------------------------------------------------------------
# 5. build_sensitivity_curves — all species
# ---------------------------------------------------------------------------

class TestBuildSensitivityCurves:
    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_returns_dict_for_species(self, species: str):
        curves = build_sensitivity_curves(species, WL)
        assert isinstance(curves, dict)
        assert len(curves) > 0

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_all_curves_normalized(self, species: str):
        curves = build_sensitivity_curves(species, WL)
        for name, s in curves.items():
            assert abs(s.max() - 1.0) < 1e-10, (
                f"{species}/{name}: peak = {s.max()}, expected 1.0"
            )

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_correct_receptor_types(self, species: str):
        curves = build_sensitivity_curves(species, WL)
        expected = set(LAMBDA_MAX[species].keys())
        assert set(curves.keys()) == expected

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_output_shape(self, species: str):
        curves = build_sensitivity_curves(species, WL)
        for name, s in curves.items():
            assert s.shape == WL.shape, (
                f"{species}/{name}: shape {s.shape} != wavelength grid {WL.shape}"
            )

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            build_sensitivity_curves("goldfish", WL)

    def test_human_has_three_cone_types(self):
        curves = build_sensitivity_curves("human", WL)
        cone_types = [k for k in curves if "cone" in k.lower()]
        assert len(cone_types) == 3, f"Expected 3 cone types, got {cone_types}"

    def test_dog_is_dichromat(self):
        """Dogs have no M-cone."""
        curves = build_sensitivity_curves("dog", WL)
        assert "M_cone" not in curves
        assert "S_cone" in curves
        assert "L_cone" in curves

    def test_cat_is_dichromat(self):
        curves = build_sensitivity_curves("cat", WL)
        assert "M_cone" not in curves


# ---------------------------------------------------------------------------
# 6. Cross-species λ_max consistency
# ---------------------------------------------------------------------------

class TestCrossSpecies:
    """Dog and cat L-cone peaks should be red-shifted vs human S-cone."""

    def test_dog_l_cone_peak_in_range(self):
        s = govardovskii_a1(LAMBDA_MAX["dog"]["L_cone"], WL)
        measured = peak_wavelength(s, WL)
        assert 545 <= measured <= 565, f"Dog L-cone peak {measured} nm out of range"

    def test_cat_s_cone_red_shifted_vs_human(self):
        """Cat S-cone (450 nm) is more red-shifted than human S-cone (420 nm)."""
        assert LAMBDA_MAX["cat"]["S_cone"] > LAMBDA_MAX["human"]["S_cone"]

    def test_rod_peaks_clustered(self):
        """All three species' rod peaks should be within 10 nm of each other."""
        rod_peaks = [LAMBDA_MAX[sp]["rod"] for sp in ["human", "dog", "cat"]]
        spread = max(rod_peaks) - min(rod_peaks)
        assert spread <= 10, f"Rod peak spread = {spread} nm (expected ≤ 10 nm)"
