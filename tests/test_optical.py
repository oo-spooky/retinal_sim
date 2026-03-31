"""Tests for Phase 5: Gaussian PSF and OpticalStage."""
from __future__ import annotations

import numpy as np
import pytest

from retinal_sim.optical.psf import PSFGenerator
from retinal_sim.optical.stage import OpticalParams, OpticalStage, RetinalIrradiance
from retinal_sim.spectral.upsampler import SpectralImage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _human_params(**kwargs) -> OpticalParams:
    """Human-like OpticalParams (architecture §2b)."""
    defaults = dict(
        pupil_shape="circular",
        pupil_diameter_mm=3.0,
        axial_length_mm=24.0,
        focal_length_mm=22.3,
        corneal_radius_mm=7.8,
        lca_diopters=2.0,
    )
    defaults.update(kwargs)
    return OpticalParams(**defaults)


def _dog_params(**kwargs) -> OpticalParams:
    """Dog-like OpticalParams (architecture §2b)."""
    defaults = dict(
        pupil_shape="circular",
        pupil_diameter_mm=7.0,
        axial_length_mm=21.0,
        focal_length_mm=17.0,
        corneal_radius_mm=8.5,
        lca_diopters=1.5,
    )
    defaults.update(kwargs)
    return OpticalParams(**defaults)


WAVELENGTHS_5 = np.array([400.0, 450.0, 550.0, 650.0, 700.0])
WAVELENGTHS_1 = np.array([550.0])


# ---------------------------------------------------------------------------
# PSFGenerator — shape and type
# ---------------------------------------------------------------------------

class TestPSFGeneratorShape:
    def test_returns_3d_array(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5)
        assert kernels.ndim == 3

    def test_shape_n_lam_k_k(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5, kernel_size=31)
        assert kernels.shape == (5, 31, 31)

    def test_shape_single_wavelength(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_1, kernel_size=15)
        assert kernels.shape == (1, 15, 15)

    def test_custom_kernel_size(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5, kernel_size=11)
        assert kernels.shape == (5, 11, 11)

    def test_non_negative(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5)
        assert np.all(kernels >= 0.0)


# ---------------------------------------------------------------------------
# PSF energy conservation (§11b validation criterion)
# ---------------------------------------------------------------------------

class TestPSFEnergyConservation:
    """§11b: |sum(PSF) - 1.0| < 1e-6 per wavelength band."""

    def test_energy_human_default(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5)
        for i, lam in enumerate(WAVELENGTHS_5):
            total = float(kernels[i].sum())
            assert abs(total - 1.0) < 1e-6, (
                f"Energy not conserved at λ={lam}nm: sum={total}"
            )

    def test_energy_dog_large_pupil(self):
        gen = PSFGenerator(_dog_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5, kernel_size=31)
        for i in range(len(WAVELENGTHS_5)):
            assert abs(float(kernels[i].sum()) - 1.0) < 1e-6

    def test_energy_with_defocus(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5, defocus_diopters=2.0)
        for i in range(len(WAVELENGTHS_5)):
            assert abs(float(kernels[i].sum()) - 1.0) < 1e-6

    def test_energy_single_wavelength(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_1)
        assert abs(float(kernels[0].sum()) - 1.0) < 1e-6

    def test_energy_fine_pixel_scale(self):
        # Smaller pixel scale → larger sigma_px → check normalisation still holds.
        gen = PSFGenerator(_human_params(), pixel_scale_mm_per_px=0.0001)
        kernels = gen.gaussian_psf(WAVELENGTHS_1, kernel_size=63)
        assert abs(float(kernels[0].sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# PSF physical properties
# ---------------------------------------------------------------------------

class TestPSFPhysics:
    def test_peak_at_centre_odd_kernel(self):
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_5, kernel_size=31)
        center = 15
        for i in range(len(WAVELENGTHS_5)):
            peak_idx = np.unravel_index(np.argmax(kernels[i]), kernels[i].shape)
            assert peak_idx == (center, center), (
                f"Peak not at centre for band {i}: got {peak_idx}"
            )

    def test_peak_at_centre_even_kernel(self):
        # Even kernel: peak should still be at one of the two central pixels.
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_1, kernel_size=32)
        # Centre region: rows/cols 15–16 should contain the maximum.
        peak_idx = np.unravel_index(np.argmax(kernels[0]), kernels[0].shape)
        assert peak_idx[0] in (15, 16) and peak_idx[1] in (15, 16)

    def test_wavelength_dependent_size(self):
        """Longer wavelengths produce wider PSFs (diffraction limit scales with λ)."""
        gen = PSFGenerator(_human_params())
        lam_short = np.array([400.0])
        lam_long = np.array([700.0])
        k_short = gen.gaussian_psf(lam_short)[0]
        k_long = gen.gaussian_psf(lam_long)[0]

        # Measure variance of each kernel as proxy for width.
        center = 15
        y, x = np.mgrid[:31, :31].astype(float) - center
        r2 = x**2 + y**2
        var_short = float(np.sum(k_short * r2))
        var_long = float(np.sum(k_long * r2))
        assert var_long > var_short, (
            f"Expected wider PSF at 700nm than 400nm; var_short={var_short:.4f}, var_long={var_long:.4f}"
        )

    def test_defocus_widens_psf(self):
        """Adding defocus should produce a wider PSF than in-focus."""
        gen = PSFGenerator(_human_params())
        k_focus = gen.gaussian_psf(WAVELENGTHS_1)[0]
        k_blur = gen.gaussian_psf(WAVELENGTHS_1, defocus_diopters=3.0)[0]

        center = 15
        y, x = np.mgrid[:31, :31].astype(float) - center
        r2 = x**2 + y**2
        var_focus = float(np.sum(k_focus * r2))
        var_blur = float(np.sum(k_blur * r2))
        assert var_blur > var_focus, (
            f"Defocus PSF not wider: var_focus={var_focus:.4f}, var_blur={var_blur:.4f}"
        )

    def test_large_defocus_still_normalised(self):
        gen = PSFGenerator(_human_params(), pixel_scale_mm_per_px=0.001)
        kernels = gen.gaussian_psf(WAVELENGTHS_1, kernel_size=63, defocus_diopters=10.0)
        assert abs(float(kernels[0].sum()) - 1.0) < 1e-6

    def test_symmetric_kernel(self):
        """Gaussian kernel must be symmetric (equal values at ±x, ±y)."""
        gen = PSFGenerator(_human_params())
        kernels = gen.gaussian_psf(WAVELENGTHS_1, kernel_size=31)
        k = kernels[0]
        assert np.allclose(k, k[::-1, :], atol=1e-10)
        assert np.allclose(k, k[:, ::-1], atol=1e-10)


# ---------------------------------------------------------------------------
# PSFGenerator pixel scale parameter
# ---------------------------------------------------------------------------

class TestPSFPixelScale:
    def test_coarser_scale_produces_narrower_kernel_in_pixels(self):
        """Coarser pixel scale → same physical sigma → fewer pixels → tighter kernel."""
        fine = PSFGenerator(_human_params(), pixel_scale_mm_per_px=0.001)
        coarse = PSFGenerator(_human_params(), pixel_scale_mm_per_px=0.005)

        k_fine = fine.gaussian_psf(WAVELENGTHS_1, kernel_size=31)[0]
        k_coarse = coarse.gaussian_psf(WAVELENGTHS_1, kernel_size=31)[0]

        center = 15
        y, x = np.mgrid[:31, :31].astype(float) - center
        r2 = x**2 + y**2
        var_fine = float(np.sum(k_fine * r2))
        var_coarse = float(np.sum(k_coarse * r2))
        assert var_fine > var_coarse


# ---------------------------------------------------------------------------
# OpticalStage — initialisation and compute_psf
# ---------------------------------------------------------------------------

class TestOpticalStageInit:
    def test_instantiate(self):
        stage = OpticalStage(_human_params())
        assert stage is not None

    def test_compute_psf_shape(self):
        stage = OpticalStage(_human_params())
        kernels = stage.compute_psf(WAVELENGTHS_5)
        assert kernels.shape == (5, 31, 31)

    def test_compute_psf_energy(self):
        stage = OpticalStage(_human_params())
        kernels = stage.compute_psf(WAVELENGTHS_5)
        for i in range(len(WAVELENGTHS_5)):
            assert abs(float(kernels[i].sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# OpticalStage.apply — output type and shape
# ---------------------------------------------------------------------------

class TestOpticalStageApply:
    def _make_spectral_image(self, h=32, w=32, n_lam=5) -> SpectralImage:
        data = np.random.default_rng(0).random((h, w, n_lam)).astype(np.float32)
        return SpectralImage(data=data, wavelengths=WAVELENGTHS_5[:n_lam])

    def test_returns_retinal_irradiance(self):
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image()
        result = stage.apply(img)
        assert isinstance(result, RetinalIrradiance)

    def test_output_shape_matches_input(self):
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image(h=16, w=24, n_lam=5)
        result = stage.apply(img)
        assert result.data.shape == (16, 24, 5)

    def test_output_wavelengths_preserved(self):
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image()
        result = stage.apply(img)
        np.testing.assert_array_equal(result.wavelengths, WAVELENGTHS_5)

    def test_output_non_negative_for_non_negative_input(self):
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image()
        result = stage.apply(img)
        assert np.all(result.data >= 0.0)

    def test_apply_with_scene(self):
        """apply() should accept a SceneDescription-like object."""
        from types import SimpleNamespace
        scene = SimpleNamespace(
            mm_per_pixel=(0.005, 0.005),
            defocus_residual_diopters=0.0,
        )
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image()
        result = stage.apply(img, scene=scene)
        assert result.data.shape == img.data.shape

    def test_apply_with_defocus(self):
        """Defocus from scene should produce a different (blurrier) result."""
        from types import SimpleNamespace
        scene_focus = SimpleNamespace(
            mm_per_pixel=(0.005, 0.005),
            defocus_residual_diopters=0.0,
        )
        scene_blur = SimpleNamespace(
            mm_per_pixel=(0.005, 0.005),
            defocus_residual_diopters=3.0,
        )
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image()
        r_focus = stage.apply(img, scene=scene_focus)
        r_blur = stage.apply(img, scene=scene_blur)
        assert not np.allclose(r_focus.data, r_blur.data)

    def test_metadata_stored(self):
        stage = OpticalStage(_human_params())
        img = self._make_spectral_image()
        result = stage.apply(img)
        assert "pixel_scale_mm" in result.metadata
        assert "defocus_diopters" in result.metadata

    def test_media_transmission_applied(self):
        """media_transmission halving all wavelengths should roughly halve output energy."""
        half = OpticalParams(
            pupil_shape="circular",
            pupil_diameter_mm=3.0,
            axial_length_mm=24.0,
            focal_length_mm=22.3,
            corneal_radius_mm=7.8,
            lca_diopters=2.0,
            media_transmission=lambda lam: np.full_like(lam, 0.5, dtype=float),
        )
        no_trans = OpticalStage(_human_params())
        with_trans = OpticalStage(half)
        img = self._make_spectral_image()

        r_no = no_trans.apply(img)
        r_half = with_trans.apply(img)

        ratio = float(r_half.data.sum()) / float(r_no.data.sum())
        assert abs(ratio - 0.5) < 0.01, f"Expected ~0.5 ratio, got {ratio:.4f}"
