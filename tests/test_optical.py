"""Tests for the optical stage and PSF generation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from retinal_sim.optical.psf import PSFGenerator
from retinal_sim.optical.stage import OpticalParams, OpticalStage, RetinalIrradiance
from retinal_sim.species.config import SpeciesConfig
from retinal_sim.spectral.upsampler import SpectralImage


WAVELENGTHS_5 = np.array([400.0, 450.0, 550.0, 650.0, 700.0], dtype=float)
WAVELENGTHS_1 = np.array([550.0], dtype=float)


def _human_params(**kwargs) -> OpticalParams:
    defaults = dict(
        pupil_shape="circular",
        pupil_diameter_mm=3.0,
        pupil_height_mm=None,
        axial_length_mm=24.0,
        focal_length_mm=22.3,
        corneal_radius_mm=7.8,
        lca_diopters=2.0,
    )
    defaults.update(kwargs)
    return OpticalParams(**defaults)


def _dog_params(**kwargs) -> OpticalParams:
    defaults = dict(
        pupil_shape="circular",
        pupil_diameter_mm=4.0,
        pupil_height_mm=None,
        axial_length_mm=21.0,
        focal_length_mm=17.0,
        corneal_radius_mm=8.5,
        lca_diopters=1.5,
    )
    defaults.update(kwargs)
    return OpticalParams(**defaults)


def _cat_params(**kwargs) -> OpticalParams:
    defaults = dict(
        pupil_shape="slit",
        pupil_diameter_mm=2.0,
        pupil_height_mm=7.0,
        axial_length_mm=22.5,
        focal_length_mm=18.5,
        corneal_radius_mm=8.5,
        lca_diopters=1.5,
    )
    defaults.update(kwargs)
    return OpticalParams(**defaults)


def _kernel_second_moments(kernel: np.ndarray) -> tuple[float, float]:
    center = kernel.shape[0] // 2
    y, x = np.mgrid[: kernel.shape[0], : kernel.shape[1]].astype(float) - center
    var_x = float(np.sum(kernel * (x**2)))
    var_y = float(np.sum(kernel * (y**2)))
    return var_x, var_y


def _point_spectral_image(size: int = 33, wavelengths: np.ndarray = WAVELENGTHS_1) -> SpectralImage:
    data = np.zeros((size, size, len(wavelengths)), dtype=np.float32)
    data[size // 2, size // 2, :] = 1.0
    return SpectralImage(data=data, wavelengths=wavelengths)


class TestOpticalParams:
    def test_circular_pupil_area_matches_analytic_area(self):
        params = _human_params(pupil_diameter_mm=3.0)
        expected = np.pi * (3.0 / 2.0) ** 2
        assert params.pupil_area_mm2() == pytest.approx(expected)

    def test_slit_pupil_area_uses_elliptical_approximation(self):
        params = _cat_params(pupil_diameter_mm=2.0, pupil_height_mm=7.0)
        expected = np.pi * (2.0 / 2.0) * (7.0 / 2.0)
        assert params.pupil_area_mm2() == pytest.approx(expected)

    def test_slit_anisotropy_flag_requires_taller_height_than_width(self):
        assert _cat_params().anisotropy_active() is True
        assert _cat_params(pupil_height_mm=2.0).anisotropy_active() is False


class TestPSFGenerator:
    def test_returns_3d_array(self):
        kernels = PSFGenerator(_human_params()).gaussian_psf(WAVELENGTHS_5)
        assert kernels.shape == (5, 31, 31)

    def test_compute_psf_can_return_metadata(self):
        stage = OpticalStage(_cat_params())
        kernels, metadata = stage.compute_psf(WAVELENGTHS_1, return_metadata=True)
        assert kernels.shape == (1, 31, 31)
        assert metadata["anisotropy_active"] is True
        assert metadata["effective_f_number_x"] > metadata["effective_f_number_y"]

    @pytest.mark.parametrize("params", [_human_params(), _dog_params(), _cat_params()])
    def test_energy_conserved_per_wavelength(self, params: OpticalParams):
        kernels = PSFGenerator(params).gaussian_psf(WAVELENGTHS_5, defocus_diopters=2.0)
        for i in range(len(WAVELENGTHS_5)):
            assert abs(float(kernels[i].sum()) - 1.0) < 1e-6

    def test_longer_wavelength_produces_wider_circular_psf(self):
        gen = PSFGenerator(_human_params())
        short = gen.gaussian_psf(np.array([400.0]))[0]
        long = gen.gaussian_psf(np.array([700.0]))[0]
        var_x_short, _ = _kernel_second_moments(short)
        var_x_long, _ = _kernel_second_moments(long)
        assert var_x_long > var_x_short

    def test_defocus_widens_psf(self):
        gen = PSFGenerator(_human_params())
        focused = gen.gaussian_psf(WAVELENGTHS_1)[0]
        blurred = gen.gaussian_psf(WAVELENGTHS_1, defocus_diopters=3.0)[0]
        var_x_focus, _ = _kernel_second_moments(focused)
        var_x_blur, _ = _kernel_second_moments(blurred)
        assert var_x_blur > var_x_focus

    def test_circular_psf_remains_symmetric(self):
        kernel = PSFGenerator(_human_params()).gaussian_psf(WAVELENGTHS_1)[0]
        assert np.allclose(kernel, kernel[::-1, :], atol=1e-10)
        assert np.allclose(kernel, kernel[:, ::-1], atol=1e-10)

    def test_cat_slit_psf_is_anisotropic(self):
        kernel = PSFGenerator(_cat_params(), pixel_scale_mm_per_px=0.001).gaussian_psf(WAVELENGTHS_1)[0]
        var_x, var_y = _kernel_second_moments(kernel)
        assert var_x > var_y

    def test_cat_slit_reports_axis_sigmas(self):
        _, metadata = PSFGenerator(_cat_params(), pixel_scale_mm_per_px=0.001).gaussian_psf(
            WAVELENGTHS_1,
            return_metadata=True,
        )
        assert metadata["sigma_mm_x"][0] > metadata["sigma_mm_y"][0]
        assert metadata["sigma_px_x"][0] > metadata["sigma_px_y"][0]

    def test_coarser_pixel_scale_makes_psf_narrower_in_pixels(self):
        fine = PSFGenerator(_human_params(), pixel_scale_mm_per_px=0.001).gaussian_psf(WAVELENGTHS_1)[0]
        coarse = PSFGenerator(_human_params(), pixel_scale_mm_per_px=0.005).gaussian_psf(WAVELENGTHS_1)[0]
        var_x_fine, _ = _kernel_second_moments(fine)
        var_x_coarse, _ = _kernel_second_moments(coarse)
        assert var_x_fine > var_x_coarse


class TestOpticalStage:
    def _make_spectral_image(self, h: int = 32, w: int = 32, n_lam: int = 5) -> SpectralImage:
        data = np.random.default_rng(0).random((h, w, n_lam)).astype(np.float32)
        return SpectralImage(data=data, wavelengths=WAVELENGTHS_5[:n_lam])

    def test_apply_returns_retinal_irradiance(self):
        result = OpticalStage(_human_params()).apply(self._make_spectral_image())
        assert isinstance(result, RetinalIrradiance)

    def test_output_shape_matches_input(self):
        img = self._make_spectral_image(h=16, w=24)
        result = OpticalStage(_human_params()).apply(img)
        assert result.data.shape == img.data.shape

    def test_output_wavelengths_preserved(self):
        img = self._make_spectral_image()
        result = OpticalStage(_human_params()).apply(img)
        np.testing.assert_array_equal(result.wavelengths, img.wavelengths)

    def test_apply_uses_scene_pixel_scale_and_defocus(self):
        scene = SimpleNamespace(mm_per_pixel=(0.005, 0.005), defocus_residual_diopters=1.5)
        result = OpticalStage(_human_params()).apply(self._make_spectral_image(), scene=scene)
        assert result.metadata["pixel_scale_mm"] == pytest.approx(0.005)
        assert result.metadata["defocus_residual_diopters"] == pytest.approx(1.5)

    def test_metadata_contains_r1_optical_diagnostics(self):
        result = OpticalStage(_cat_params()).apply(_point_spectral_image())
        expected_keys = {
            "pixel_scale_mm",
            "defocus_diopters",
            "defocus_residual_diopters",
            "pupil_shape",
            "pupil_area_mm2",
            "reference_pupil_area_mm2",
            "pupil_throughput_scale",
            "effective_f_number",
            "effective_f_number_x",
            "effective_f_number_y",
            "psf_sigma_mm_x",
            "psf_sigma_mm_y",
            "psf_sigma_px_x",
            "psf_sigma_px_y",
            "anisotropy_active",
        }
        assert expected_keys <= set(result.metadata)

    def test_anisotropy_flag_false_for_circular_pupil(self):
        result = OpticalStage(_human_params()).apply(_point_spectral_image())
        assert result.metadata["anisotropy_active"] is False

    def test_anisotropy_flag_true_for_cat_slit(self):
        result = OpticalStage(_cat_params()).apply(_point_spectral_image())
        assert result.metadata["anisotropy_active"] is True

    def test_throughput_scaling_matches_area_ratio_for_same_circular_psf(self):
        img = self._make_spectral_image()
        base = OpticalStage(_human_params(pupil_diameter_mm=3.0)).apply(img)
        larger = OpticalStage(_human_params(pupil_diameter_mm=6.0)).apply(img)
        expected_ratio = (6.0 / 3.0) ** 2
        ratio = float(larger.data.sum()) / float(base.data.sum())
        assert ratio == pytest.approx(expected_ratio, rel=0.02)

    def test_media_transmission_scales_total_energy(self):
        img = self._make_spectral_image()
        no_trans = OpticalStage(_human_params()).apply(img)
        half_trans = OpticalStage(
            _human_params(media_transmission=lambda lam: np.full_like(lam, 0.5, dtype=float))
        ).apply(img)
        ratio = float(half_trans.data.sum()) / float(no_trans.data.sum())
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_species_energy_differs_with_pupil_geometry(self):
        img = self._make_spectral_image()
        species_results = {
            "human": OpticalStage(SpeciesConfig.load("human").optical).apply(img),
            "dog": OpticalStage(SpeciesConfig.load("dog").optical).apply(img),
            "cat": OpticalStage(SpeciesConfig.load("cat").optical).apply(img),
        }
        energies = {name: float(result.data.sum()) for name, result in species_results.items()}
        assert energies["dog"] > energies["human"]
        assert energies["cat"] > energies["human"]

    def test_cat_point_spread_is_horizontally_broader(self):
        img = _point_spectral_image(size=41)
        result = OpticalStage(_cat_params()).apply(img)
        kernel_like = result.data[:, :, 0] / result.data[:, :, 0].sum()
        var_x, var_y = _kernel_second_moments(kernel_like)
        assert var_x > var_y
