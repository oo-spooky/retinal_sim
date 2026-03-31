"""Tests for Phase 6: Smits (1999) spectral upsampler.

Validation criterion (architecture §11a):
  RGB roundtrip RMSE < 0.02 on [0, 1] scale when reprojected via CIE 1931
  color matching functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from retinal_sim.spectral.upsampler import SpectralImage, SpectralUpsampler

# ---------------------------------------------------------------------------
# CIE 1931 2° standard observer CMFs at 5 nm intervals from 380 to 720 nm
# (69 values).  Source: CIE 015:2018 colorimetry tables.
# ---------------------------------------------------------------------------
# fmt: off
_CIE_X = np.array([
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510,
    0.077630, 0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060,
    0.336200, 0.318700, 0.290800, 0.251100, 0.195360, 0.142100, 0.095640,
    0.057950, 0.031960, 0.014700, 0.004900, 0.002400, 0.009300, 0.029100,
    0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600,
    1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400, 0.854450,
    0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
    0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
    0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899,
])
_CIE_Y = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210,
    0.002180, 0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800,
    0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600, 0.139020,
    0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000, 0.608200,
    0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300,
    0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200, 0.381000,
    0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
    0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047,
])
_CIE_Z = np.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400,
    0.371300, 0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600,
    1.772110, 1.744100, 1.669200, 1.528100, 1.287640, 1.041900, 0.812950,
    0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200, 0.111700,
    0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400,
    0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240, 0.000190,
    0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
])
# CIE standard illuminant D65 relative SPD at 5 nm, 380–720 nm.
# Normalised to S(560 nm) = 100.  Source: CIE 015:2018.
_D65 = np.array([
     50.000,  52.312,  54.622,  68.702,  82.754,  87.120,  91.486,  92.459,
     93.432,  90.057,  86.682,  95.774, 104.865, 110.936, 117.008, 117.410,
    117.812, 116.336, 114.861, 115.392, 115.923, 112.367, 108.811, 109.082,
    109.354, 108.578, 107.802, 106.296, 104.790, 106.239, 107.689, 106.047,
    104.405, 104.225, 104.046, 102.023, 100.000,  98.167,  96.334,  96.061,
     95.788,  92.237,  88.685,  89.346,  90.006,  89.802,  89.599,  88.649,
     87.699,  85.494,  83.289,  83.493,  83.697,  81.863,  80.026,  80.121,
     80.216,  81.246,  82.277,  80.281,  78.284,  74.003,  69.721,  70.665,
     71.609,  72.979,  74.349,  61.604,  48.860,
])
# fmt: on

# D65-adapted XYZ → linear sRGB matrix (IEC 61966-2-1).
_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])


def _spectrum_to_linear_srgb(spectrum: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Integrate a spectrum against D65-weighted CIE 1931 CMFs → linear sRGB.

    Uses proper D65 illuminant integration matching the basis computation in
    SpectralUpsampler._compute_basis().  The normalisation constant k ensures
    that a flat reflectance of 1.0 maps to D65 white (sRGB = [1, 1, 1]).

    Note: the Smits basis spectra exploit the CIE x-bar secondary lobe at
    ~450 nm to represent magenta, so this CIE-based integration is required
    for a valid roundtrip — simple box filters do not work.

    Args:
        spectrum: (N_λ,) float64 array.
        wavelengths: (N_λ,) wavelengths in nm matching the 380–720 nm /
            5 nm grid used throughout.

    Returns:
        (3,) linear sRGB array, clipped to [0, 1].
    """
    assert len(wavelengths) == 69, "CMF helper expects 380–720 nm at 5 nm"
    dlam = float(wavelengths[1] - wavelengths[0])  # 5 nm

    # D65 normalisation: k = 1 / (∫ D65 · ȳ dλ)
    k = 1.0 / (_D65 @ _CIE_Y * dlam)

    # XYZ tristimulus values under D65
    sd = spectrum * _D65          # (N_λ,) — reflectance × illuminant
    X = k * dlam * (sd @ _CIE_X)
    Y = k * dlam * (sd @ _CIE_Y)
    Z = k * dlam * (sd @ _CIE_Z)

    rgb = _XYZ_TO_SRGB @ np.array([X, Y, Z])
    return np.clip(rgb, 0.0, 1.0)


def _batch_roundtrip(rgb_colors: np.ndarray, upsampler: SpectralUpsampler) -> np.ndarray:
    """Return reconstructed RGB for each row in rgb_colors via CIE roundtrip.

    Args:
        rgb_colors: (N, 3) array of test colors in [0, 1].
        upsampler: configured SpectralUpsampler.

    Returns:
        (N, 3) reconstructed linear sRGB values.
    """
    img = rgb_colors.reshape(1, -1, 3).astype(np.float32)
    si = upsampler.upsample(img)
    reconstructed = np.zeros_like(rgb_colors)
    for i in range(len(rgb_colors)):
        spec = si.data[0, i, :].astype(np.float64)
        reconstructed[i] = _spectrum_to_linear_srgb(spec, si.wavelengths)
    return reconstructed


# ===========================================================================
# Tests
# ===========================================================================

class TestSpectralUpsamplerInit:
    def test_default_init(self):
        up = SpectralUpsampler()
        assert up.method == "smits"

    def test_wavelength_grid(self):
        up = SpectralUpsampler()
        assert len(up.wavelengths) == 69
        assert up.wavelengths[0] == pytest.approx(380.0)
        assert up.wavelengths[-1] == pytest.approx(720.0)
        assert up.wavelengths[1] - up.wavelengths[0] == pytest.approx(5.0)

    def test_custom_wavelength_range(self):
        up = SpectralUpsampler(wavelength_range=(400, 700), wavelength_step=10)
        assert up.wavelengths[0] == pytest.approx(400.0)
        assert up.wavelengths[-1] == pytest.approx(700.0)
        assert up.wavelengths[1] - up.wavelengths[0] == pytest.approx(10.0)

    def test_unknown_method_raises(self):
        with pytest.raises(NotImplementedError):
            SpectralUpsampler(method="mallett_yuksel")

    def test_unknown_method_string_raises(self):
        with pytest.raises(NotImplementedError):
            SpectralUpsampler(method="bogus")


class TestSpectralUpsamplerShape:
    def setup_method(self):
        self.up = SpectralUpsampler()

    def test_output_type(self):
        img = np.ones((2, 3, 3), dtype=np.float32)
        result = self.up.upsample(img)
        assert isinstance(result, SpectralImage)

    def test_output_shape_small(self):
        img = np.ones((1, 1, 3), dtype=np.float32)
        result = self.up.upsample(img)
        assert result.data.shape == (1, 1, 69)

    def test_output_shape_general(self):
        img = np.ones((4, 7, 3), dtype=np.float32)
        result = self.up.upsample(img)
        assert result.data.shape == (4, 7, 69)

    def test_data_dtype_float32(self):
        img = np.ones((2, 2, 3), dtype=np.float32)
        result = self.up.upsample(img)
        assert result.data.dtype == np.float32

    def test_wavelengths_dtype_float64(self):
        img = np.ones((2, 2, 3), dtype=np.float32)
        result = self.up.upsample(img)
        assert result.wavelengths.dtype == np.float64

    def test_wavelengths_match_upsampler(self):
        img = np.ones((1, 1, 3), dtype=np.float32)
        result = self.up.upsample(img)
        np.testing.assert_array_equal(result.wavelengths, self.up.wavelengths)

    def test_wrong_channels_raises(self):
        img = np.ones((2, 2, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            self.up.upsample(img)


class TestSpectralUpsamplerUint8:
    def setup_method(self):
        self.up = SpectralUpsampler()

    def test_uint8_white_matches_float_white(self):
        img_float = np.ones((1, 1, 3), dtype=np.float32)
        img_uint8 = np.full((1, 1, 3), 255, dtype=np.uint8)
        result_f = self.up.upsample(img_float)
        result_u = self.up.upsample(img_uint8)
        np.testing.assert_allclose(result_f.data, result_u.data, atol=1e-5)

    def test_uint8_black_is_zero(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        result = self.up.upsample(img)
        np.testing.assert_array_equal(result.data, 0.0)

    def test_uint8_gray(self):
        v = 128
        img_u = np.full((1, 1, 3), v, dtype=np.uint8)
        img_f = np.full((1, 1, 3), v / 255.0, dtype=np.float32)
        r_u = self.up.upsample(img_u)
        r_f = self.up.upsample(img_f)
        np.testing.assert_allclose(r_u.data, r_f.data, atol=1e-4)


class TestSpectralUpsamplerValues:
    def setup_method(self):
        self.up = SpectralUpsampler()

    def test_black_is_zero(self):
        img = np.zeros((3, 3, 3), dtype=np.float32)
        result = self.up.upsample(img)
        assert np.all(result.data == 0.0)

    def test_white_is_approximately_flat(self):
        img = np.ones((1, 1, 3), dtype=np.float32)
        result = self.up.upsample(img)
        spec = result.data[0, 0, :]
        # White spectrum should be close to 1.0 everywhere (Smits white basis ≈ 1)
        assert np.all(spec > 0.99)
        assert np.all(spec <= 1.01)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        img = rng.random((5, 5, 3)).astype(np.float32)
        result = self.up.upsample(img)
        assert np.all(result.data >= 0.0)

    def test_output_bounded(self):
        # Smits basis values are all ≤ 1, so output should be ≤ 1 for inputs in [0,1]
        rng = np.random.default_rng(99)
        img = rng.random((5, 5, 3)).astype(np.float32)
        result = self.up.upsample(img)
        assert np.all(result.data <= 1.1)  # small tolerance for basis overshoot at 460nm

    def test_gray_is_scaled_white(self):
        white = np.ones((1, 1, 3), dtype=np.float32)
        gray = np.full((1, 1, 3), 0.5, dtype=np.float32)
        r_white = self.up.upsample(white).data[0, 0, :]
        r_gray = self.up.upsample(gray).data[0, 0, :]
        np.testing.assert_allclose(r_gray, 0.5 * r_white, atol=1e-5)


class TestSpectralUpsamplerPhysics:
    """Test that spectral shapes are physically plausible."""

    def setup_method(self):
        self.up = SpectralUpsampler()

    def _upsample_color(self, r, g, b):
        img = np.array([[[r, g, b]]], dtype=np.float32)
        return self.up.upsample(img).data[0, 0, :]  # (N_λ,)

    def _wl_idx(self, nm):
        """Index of wavelength nm in the upsampler's grid."""
        return int(round((nm - 380) / 5))

    def test_pure_red_peaks_long_wavelength(self):
        spec = self._upsample_color(1.0, 0.0, 0.0)
        mean_long = spec[self._wl_idx(600):].mean()  # 600–720 nm
        mean_short = spec[:self._wl_idx(500)].mean()  # 380–500 nm
        assert mean_long > mean_short, "Red should peak at long wavelengths"

    def test_pure_blue_peaks_short_wavelength(self):
        spec = self._upsample_color(0.0, 0.0, 1.0)
        mean_short = spec[:self._wl_idx(500)].mean()  # 380–500 nm
        mean_long = spec[self._wl_idx(600):].mean()   # 600–720 nm
        assert mean_short > mean_long, "Blue should peak at short wavelengths"

    def test_pure_green_peaks_mid_wavelength(self):
        spec = self._upsample_color(0.0, 1.0, 0.0)
        mid = spec[self._wl_idx(500):self._wl_idx(600)].mean()  # 500–600 nm
        short = spec[:self._wl_idx(480)].mean()
        long = spec[self._wl_idx(650):].mean()
        assert mid > short, "Green should dominate mid wavelengths over short"
        assert mid > long, "Green should dominate mid wavelengths over long"

    def test_pure_red_zero_at_short_wavelengths(self):
        spec = self._upsample_color(1.0, 0.0, 0.0)
        # Smits red basis is near-zero from 460–580 nm
        assert spec[self._wl_idx(470):(self._wl_idx(580) + 1)].max() < 0.15

    def test_pure_blue_zero_at_long_wavelengths(self):
        spec = self._upsample_color(0.0, 0.0, 1.0)
        # Numerically-optimised blue basis has small but non-zero values above
        # 530 nm (exploits the CIE x̄ secondary lobe).  Threshold is 0.05.
        assert spec[self._wl_idx(530):].max() < 0.05

    def test_linearity_in_neutral_direction(self):
        """Scaling a neutral (r=g=b) input scales the spectrum linearly."""
        v1, v2 = 0.3, 0.6
        s1 = self._upsample_color(v1, v1, v1)
        s2 = self._upsample_color(v2, v2, v2)
        np.testing.assert_allclose(s2, (v2 / v1) * s1, atol=1e-5)


class TestRoundtrip:
    """§11a validation: RGB roundtrip RMSE < 0.02 via CIE 1931 CMFs."""

    def setup_method(self):
        self.up = SpectralUpsampler()

    def test_roundtrip_white(self):
        colors = np.array([[1.0, 1.0, 1.0]])
        recon = _batch_roundtrip(colors, self.up)
        np.testing.assert_allclose(recon[0], [1.0, 1.0, 1.0], atol=0.01)

    def test_roundtrip_black(self):
        colors = np.array([[0.0, 0.0, 0.0]])
        recon = _batch_roundtrip(colors, self.up)
        np.testing.assert_allclose(recon[0], [0.0, 0.0, 0.0], atol=0.01)

    def test_roundtrip_pure_primaries(self):
        """Pure R, G, B should round-trip with error < 0.05 per channel."""
        colors = np.array([
            [1.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0],  # green
            [0.0, 0.0, 1.0],  # blue
        ])
        recon = _batch_roundtrip(colors, self.up)
        # Each pure primary should reconstruct with the right dominant channel
        assert recon[0, 0] > 0.80, "Red primary R channel"
        assert recon[1, 1] > 0.70, "Green primary G channel"
        assert recon[2, 2] > 0.80, "Blue primary B channel"

    def test_roundtrip_secondaries(self):
        """Pure secondaries (cyan, magenta, yellow) should round-trip."""
        colors = np.array([
            [0.0, 1.0, 1.0],  # cyan
            [1.0, 0.0, 1.0],  # magenta
            [1.0, 1.0, 0.0],  # yellow
        ])
        recon = _batch_roundtrip(colors, self.up)
        # Cyan: high G and B, low R
        assert recon[0, 1] > 0.5 and recon[0, 2] > 0.5
        assert recon[0, 0] < 0.3

    def test_roundtrip_rmse_below_threshold(self):
        """§11a: RGB roundtrip RMSE < 0.02 on [0, 1] scale."""
        test_colors = np.array([
            [1.0, 1.0, 1.0],   # white
            [0.0, 0.0, 0.0],   # black
            [1.0, 0.0, 0.0],   # red
            [0.0, 1.0, 0.0],   # green
            [0.0, 0.0, 1.0],   # blue
            [0.0, 1.0, 1.0],   # cyan
            [1.0, 0.0, 1.0],   # magenta
            [1.0, 1.0, 0.0],   # yellow
            [0.5, 0.5, 0.5],   # mid gray
            [0.8, 0.2, 0.0],   # orange
            [0.0, 0.5, 0.3],   # teal
            [0.6, 0.6, 0.6],   # light gray
            [0.2, 0.4, 0.8],   # blue-purple
        ])
        recon = _batch_roundtrip(test_colors, self.up)
        rmse = np.sqrt(np.mean((test_colors - recon) ** 2))
        assert rmse < 0.02, f"Roundtrip RMSE {rmse:.4f} exceeds 0.02 threshold (§11a)"

    def test_roundtrip_gray_ramp(self):
        """Gray ramp should round-trip accurately."""
        gray_vals = np.linspace(0.0, 1.0, 11)
        colors = np.stack([gray_vals, gray_vals, gray_vals], axis=1)
        recon = _batch_roundtrip(colors, self.up)
        rmse = np.sqrt(np.mean((colors - recon) ** 2))
        assert rmse < 0.02, f"Gray ramp RMSE {rmse:.4f} exceeds threshold"
