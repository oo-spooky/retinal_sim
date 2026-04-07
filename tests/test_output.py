"""Phase 8 tests: Voronoi visualization and mosaic reconstruction."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from retinal_sim.optical.stage import RetinalIrradiance
from retinal_sim.retina.mosaic import PhotoreceptorMosaic
from retinal_sim.retina.stage import MosaicActivation
from retinal_sim.spectral.upsampler import SpectralImage
from retinal_sim.output.diagnostics import (
    build_comparative_renderings,
    build_photoreceptor_activation_diagnostics,
    build_retinal_irradiance_diagnostics,
)
from retinal_sim.output.voronoi import render_voronoi, _TYPE_BASE_COLOR
from retinal_sim.output.reconstruction import render_reconstructed
from retinal_sim.output.comparison import render_comparison, render_mosaic_map


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_WL = 69  # 380–720 nm at 5 nm


def _make_activation(
    n: int = 30,
    seed: int = 0,
    types: list | None = None,
    response_value: float | None = None,
    positions: np.ndarray | None = None,
) -> MosaicActivation:
    """Synthetic MosaicActivation — no need to run the full pipeline."""
    rng = np.random.default_rng(seed)

    if positions is None:
        pos = rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32)
    else:
        pos = np.asarray(positions, dtype=np.float32)
        n = len(pos)

    all_types = ["L_cone", "M_cone", "S_cone", "rod"]
    if types is None:
        idx = rng.integers(0, len(all_types), n)
        t = np.array([all_types[i] for i in idx], dtype="U10")
    else:
        t = np.array(types, dtype="U10")

    apertures = np.full(n, 5.0, dtype=np.float32)
    sensitivities = np.ones((n, N_WL), dtype=np.float32)
    mosaic = PhotoreceptorMosaic(
        positions=pos,
        types=t,
        apertures=apertures,
        sensitivities=sensitivities,
    )

    if response_value is not None:
        resp = np.full(n, float(response_value), dtype=np.float32)
    else:
        resp = rng.uniform(0.0, 1.0, n).astype(np.float32)

    return MosaicActivation(mosaic=mosaic, responses=resp)


# ---------------------------------------------------------------------------
# TestRenderVoronoi
# ---------------------------------------------------------------------------

class TestRenderVoronoi:
    def test_returns_ndarray(self):
        act = _make_activation()
        result = render_voronoi(act)
        assert isinstance(result, np.ndarray)

    def test_output_shape_default(self):
        act = _make_activation()
        img = render_voronoi(act)
        assert img.shape == (256, 256, 3)

    def test_output_shape_custom(self):
        act = _make_activation()
        img = render_voronoi(act, output_size=(64, 128))
        assert img.shape == (64, 128, 3)

    def test_dtype_float32(self):
        act = _make_activation()
        img = render_voronoi(act)
        assert img.dtype == np.float32

    def test_values_in_unit_range(self):
        act = _make_activation()
        img = render_voronoi(act)
        assert img.min() >= 0.0
        assert img.max() <= 1.0 + 1e-6

    def test_zero_response_gives_black_image(self):
        act = _make_activation(response_value=0.0)
        img = render_voronoi(act)
        assert img.max() < 1e-6

    def test_full_response_gives_bright_image(self):
        act = _make_activation(response_value=1.0)
        img = render_voronoi(act)
        # At least some pixels must be bright
        assert img.max() > 0.9

    def test_rods_produce_grayscale(self):
        """Rod-only mosaic: R == G == B for every pixel."""
        act = _make_activation(n=20, types=["rod"] * 20, response_value=0.7)
        img = render_voronoi(act)
        assert np.allclose(img[..., 0], img[..., 1], atol=1e-6)
        assert np.allclose(img[..., 1], img[..., 2], atol=1e-6)

    def test_l_cone_only_is_red_dominant(self):
        """L-cone-only mosaic: red channel > green and blue channels."""
        act = _make_activation(n=20, types=["L_cone"] * 20, response_value=0.8)
        img = render_voronoi(act)
        assert img[..., 0].mean() > img[..., 1].mean()
        assert img[..., 0].mean() > img[..., 2].mean()

    def test_s_cone_only_is_blue_dominant(self):
        """S-cone-only mosaic: blue channel > red and green channels."""
        act = _make_activation(n=20, types=["S_cone"] * 20, response_value=0.8)
        img = render_voronoi(act)
        assert img[..., 2].mean() > img[..., 0].mean()
        assert img[..., 2].mean() > img[..., 1].mean()

    def test_m_cone_only_is_green_dominant(self):
        """M-cone-only mosaic: green channel > red and blue channels."""
        act = _make_activation(n=20, types=["M_cone"] * 20, response_value=0.8)
        img = render_voronoi(act)
        assert img[..., 1].mean() > img[..., 0].mean()
        assert img[..., 1].mean() > img[..., 2].mean()

    def test_single_receptor_fills_entire_image(self):
        """With one receptor, every pixel must map to that receptor's colour."""
        act = _make_activation(
            n=1,
            types=["L_cone"],
            response_value=1.0,
            positions=np.array([[0.0, 0.0]]),
        )
        img = render_voronoi(act, output_size=(32, 32))
        # Entire image should be red (1, 0, 0)
        assert np.allclose(img[..., 0], 1.0, atol=1e-6)
        assert np.allclose(img[..., 1], 0.0, atol=1e-6)
        assert np.allclose(img[..., 2], 0.0, atol=1e-6)

    def test_mm_range_kwarg(self):
        """Explicit mm_range is accepted and output shape is unchanged."""
        act = _make_activation()
        img = render_voronoi(act, output_size=(64, 64), mm_range=(-1.0, 1.0, -1.0, 1.0))
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.float32

    def test_response_scales_brightness(self):
        """Higher response → brighter pixels (same type, same positions)."""
        pos = np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32)
        act_dim = _make_activation(types=["L_cone", "L_cone"], response_value=0.2, positions=pos)
        act_bright = _make_activation(types=["L_cone", "L_cone"], response_value=0.8, positions=pos)
        img_dim = render_voronoi(act_dim, output_size=(32, 32))
        img_bright = render_voronoi(act_bright, output_size=(32, 32))
        assert img_bright.mean() > img_dim.mean()


# ---------------------------------------------------------------------------
# TestPerceptualRendering — end-to-end color round-trip
# ---------------------------------------------------------------------------


class TestPerceptualHumanColor:
    """Whole-pipeline checks that the human perceptual rendering doesn't
    cast a global hue. These are end-to-end and a bit slow, but they catch
    regressions in the LMS→sRGB transform that synthetic-mosaic tests can't.
    """

    def _render_uniform(self, rgb_tuple, size=32):
        from retinal_sim.output.perceptual import render_perceptual_image
        from retinal_sim.pipeline import RetinalSimulator

        rgb = np.full((size, size, 3), rgb_tuple, dtype=np.float32)
        sim = RetinalSimulator("human", patch_extent_deg=2.0)
        result = sim.simulate(rgb, input_mode="reflectance_under_d65")
        return render_perceptual_image(result, grid_shape=(size, size))

    def test_white_in_is_approximately_neutral(self):
        out = self._render_uniform((1.0, 1.0, 1.0))
        means = out.reshape(-1, 3).mean(axis=0)
        # Channels should be within 0.1 of each other (no global cast).
        assert abs(means[0] - means[1]) < 0.18
        assert abs(means[1] - means[2]) < 0.18
        assert abs(means[0] - means[2]) < 0.18

    def test_red_in_is_red_dominant(self):
        out = self._render_uniform((1.0, 0.0, 0.0))
        means = out.reshape(-1, 3).mean(axis=0)
        assert means[0] > means[1]
        assert means[0] > means[2]


# ---------------------------------------------------------------------------
# TestRenderReconstructed
# ---------------------------------------------------------------------------

class TestRenderReconstructed:
    def test_returns_ndarray(self):
        act = _make_activation()
        result = render_reconstructed(act, (64, 64))
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_grid_shape(self):
        act = _make_activation()
        for shape in [(32, 32), (64, 128), (100, 50)]:
            result = render_reconstructed(act, shape)
            assert result.shape == shape

    def test_dtype_float32(self):
        act = _make_activation()
        result = render_reconstructed(act, (64, 64))
        assert result.dtype == np.float32

    def test_values_in_unit_range(self):
        act = _make_activation()
        result = render_reconstructed(act, (64, 64))
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_zero_response_gives_zero_output(self):
        act = _make_activation(response_value=0.0)
        result = render_reconstructed(act, (64, 64))
        assert result.max() < 1e-6

    def test_full_response_gives_ones(self):
        act = _make_activation(response_value=1.0)
        result = render_reconstructed(act, (64, 64))
        assert result.min() > 0.9

    def test_spatial_locality(self):
        """Left receptor (bright) → left half of grid brighter than right half (dark)."""
        # Two receptors separated horizontally; use explicit mm_range so the
        # dividing boundary falls exactly at the midpoint column.
        positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        types = ["L_cone", "L_cone"]
        mosaic = PhotoreceptorMosaic(
            positions=positions,
            types=np.array(types, dtype="U10"),
            apertures=np.array([5.0, 5.0], dtype=np.float32),
            sensitivities=np.ones((2, N_WL), dtype=np.float32),
        )
        responses = np.array([1.0, 0.0], dtype=np.float32)
        act = MosaicActivation(mosaic=mosaic, responses=responses)

        # mm_range centres the two receptors in left/right halves
        result = render_reconstructed(
            act, (64, 64), mm_range=(-0.1, 1.1, -0.5, 0.5)
        )
        left_mean = float(result[:, :32].mean())
        right_mean = float(result[:, 32:].mean())
        assert left_mean > right_mean

    def test_mm_range_kwarg(self):
        act = _make_activation()
        result = render_reconstructed(act, (64, 64), mm_range=(-1.0, 1.0, -1.0, 1.0))
        assert result.shape == (64, 64)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestRenderComparison
# ---------------------------------------------------------------------------

class TestRenderComparison:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        acts = {"human": _make_activation(seed=0), "dog": _make_activation(seed=1)}
        fig = render_comparison(acts)
        assert hasattr(fig, "axes"), "Expected a matplotlib Figure"
        plt.close(fig)

    def test_figure_has_one_axis_per_species(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for n in [1, 2, 3]:
            acts = {f"sp{i}": _make_activation(seed=i) for i in range(n)}
            fig = render_comparison(acts)
            assert len(fig.axes) >= n
            plt.close(fig)

    def test_single_species(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        acts = {"cat": _make_activation()}
        fig = render_comparison(acts)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# TestRenderMosaicMap
# ---------------------------------------------------------------------------

class TestRenderMosaicMap:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        act = _make_activation()
        fig = render_mosaic_map(act.mosaic)
        assert hasattr(fig, "axes")
        plt.close(fig)

    def test_accepts_all_receptor_types(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = 40
        types = (["rod", "S_cone", "M_cone", "L_cone"] * (n // 4))[:n]
        act = _make_activation(n=n, types=types)
        fig = render_mosaic_map(act.mosaic)
        assert fig is not None
        plt.close(fig)

    def test_custom_output_size(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        act = _make_activation()
        fig = render_mosaic_map(act.mosaic, output_size=(200, 200))
        assert fig is not None
        plt.close(fig)


class TestDiagnosticBuilders:
    def test_build_retinal_irradiance_diagnostics_has_structured_groups(self):
        wavelengths = np.linspace(380.0, 720.0, N_WL, dtype=np.float64)
        y, x = np.mgrid[0:6, 0:6]
        base = (x + y)[..., np.newaxis].astype(np.float32)
        spectral = SpectralImage(
            data=np.repeat(base + 0.5, N_WL, axis=2),
            wavelengths=wavelengths,
        )
        irradiance = RetinalIrradiance(
            data=np.repeat(base + 1.0, N_WL, axis=2),
            wavelengths=wavelengths,
            metadata={
                "pupil_area_mm2": 7.0,
                "pupil_throughput_scale": 1.0,
                "effective_f_number": 7.4,
                "anisotropy_active": False,
                "media_transmission_source": "unit-test",
                "psf_sigma_px_x": [1.0] * N_WL,
                "psf_sigma_px_y": [1.0] * N_WL,
            },
        )
        diagnostics = build_retinal_irradiance_diagnostics(spectral, irradiance)

        assert diagnostics["family_label"] == "retinal irradiance diagnostics"
        assert "delivered_spectrum_summary" in diagnostics
        assert "selected_wavelength_slices" in diagnostics
        assert len(diagnostics["selected_wavelength_slices"]) == 3
        assert diagnostics["band_composite"]["image_data"].shape == (6, 6, 3)
        assert diagnostics["optical_stage_summary"]["media_transmission_source"] == "unit-test"

    def test_build_photoreceptor_activation_diagnostics_has_response_summary(self):
        activation = _make_activation(
            n=12,
            types=["L_cone", "M_cone", "S_cone", "rod"] * 3,
            response_value=0.5,
        )
        diagnostics = build_photoreceptor_activation_diagnostics(
            scene=SimpleNamespace(
                scene_covers_patch_fraction=0.75,
                retinal_width_mm=1.2,
                retinal_height_mm=1.0,
            ),
            mosaic=activation.mosaic,
            activation=activation,
            receptor_rows=np.arange(12, dtype=int) % 6,
            receptor_cols=np.arange(12, dtype=int) % 6,
            stimulated_mask=np.array([True] * 8 + [False] * 4),
            input_shape=(6, 6),
        )

        assert diagnostics["family_label"] == "photoreceptor activation diagnostics"
        assert diagnostics["overall_summary"]["n_receptors"] == 12
        assert len(diagnostics["response_summary_by_type"]) == 4
        assert diagnostics["sampling_footprint_summary"]["stimulated_receptor_count"] == 8
        assert diagnostics["mosaic_footprint_overlay"]["image_data"].shape == (6, 6, 3)

    def test_build_comparative_renderings_are_claim_calibrated(self):
        activation = _make_activation(n=20, seed=3)
        diagnostics = build_comparative_renderings(activation, input_shape=(24, 24))

        assert diagnostics["family_label"] == "comparative renderings"
        assert "not direct perceptual" in diagnostics["scope_note"]
        labels = [item["label"] for item in diagnostics["items"]]
        assert "Retinal-information rendering" in labels
        assert any("Comparative rendering" in label for label in labels)
