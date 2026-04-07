"""Phase 7 tests: RetinalStage spectral integration + Naka-Rushton transduction."""
from __future__ import annotations

import numpy as np
import pytest

from retinal_sim.retina.stage import MosaicActivation, RetinalStage
from retinal_sim.retina.mosaic import PhotoreceptorMosaic
from retinal_sim.optical.stage import RetinalIrradiance
from retinal_sim.species.config import SpeciesConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WL = np.arange(380, 721, 5, dtype=np.float32)  # 69 bands
N_WL = len(WL)  # 69


def _make_irradiance(value: float = 0.01, H: int = 64, W: int = 64) -> RetinalIrradiance:
    """Uniform spectral irradiance across the patch."""
    data = np.full((H, W, N_WL), value, dtype=np.float32)
    return RetinalIrradiance(
        data=data,
        wavelengths=WL.astype(float),
        metadata={"pixel_scale_mm": 0.005},  # 5 µm/px
    )


def _make_spectral_irradiance(
    spectrum: np.ndarray, H: int = 64, W: int = 64
) -> RetinalIrradiance:
    """Irradiance with a uniform spatial field and given spectral shape (N_WL,)."""
    data = np.outer(np.ones(H * W), spectrum).reshape(H, W, N_WL).astype(np.float32)
    return RetinalIrradiance(
        data=data,
        wavelengths=WL.astype(float),
        metadata={"pixel_scale_mm": 0.005},
    )


@pytest.fixture(scope="module")
def human_stage() -> RetinalStage:
    cfg = SpeciesConfig.load("human")
    return RetinalStage(cfg.retinal, cfg.optical)


@pytest.fixture(scope="module")
def human_mosaic(human_stage: RetinalStage) -> PhotoreceptorMosaic:
    return human_stage.generate_mosaic(seed=42)


@pytest.fixture(scope="module")
def dog_stage() -> RetinalStage:
    cfg = SpeciesConfig.load("dog")
    return RetinalStage(cfg.retinal, cfg.optical)


@pytest.fixture(scope="module")
def cat_stage() -> RetinalStage:
    cfg = SpeciesConfig.load("cat")
    return RetinalStage(cfg.retinal, cfg.optical)


# ---------------------------------------------------------------------------
# TestRetinalStageInit
# ---------------------------------------------------------------------------

class TestRetinalStageInit:
    def test_human_init(self) -> None:
        cfg = SpeciesConfig.load("human")
        stage = RetinalStage(cfg.retinal, cfg.optical)
        assert stage is not None

    def test_dog_init(self) -> None:
        cfg = SpeciesConfig.load("dog")
        RetinalStage(cfg.retinal, cfg.optical)

    def test_cat_init(self) -> None:
        cfg = SpeciesConfig.load("cat")
        RetinalStage(cfg.retinal, cfg.optical)


# ---------------------------------------------------------------------------
# TestGenerateMosaic
# ---------------------------------------------------------------------------

class TestGenerateMosaic:
    def test_returns_photoreceptor_mosaic(self, human_stage: RetinalStage) -> None:
        mosaic = human_stage.generate_mosaic(seed=0)
        assert isinstance(mosaic, PhotoreceptorMosaic)

    def test_mosaic_has_receptors(self, human_mosaic: PhotoreceptorMosaic) -> None:
        assert human_mosaic.n_receptors > 0

    def test_sensitivities_shape(self, human_mosaic: PhotoreceptorMosaic) -> None:
        assert human_mosaic.sensitivities.shape == (human_mosaic.n_receptors, N_WL)

    def test_reproducible(self, human_stage: RetinalStage) -> None:
        m1 = human_stage.generate_mosaic(seed=7)
        m2 = human_stage.generate_mosaic(seed=7)
        np.testing.assert_array_equal(m1.positions, m2.positions)

    def test_different_seeds_differ(self, human_stage: RetinalStage) -> None:
        m1 = human_stage.generate_mosaic(seed=1)
        m2 = human_stage.generate_mosaic(seed=2)
        assert not np.array_equal(m1.positions, m2.positions)


# ---------------------------------------------------------------------------
# TestComputeResponseOutput
# ---------------------------------------------------------------------------

class TestComputeResponseOutput:
    def test_returns_mosaic_activation(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert isinstance(result, MosaicActivation)

    def test_responses_shape(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.responses.shape == (human_mosaic.n_receptors,)

    def test_mosaic_reference_preserved(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.mosaic is human_mosaic

    def test_metadata_keys(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        for key in ("mm_per_pixel", "mean_excitation", "mean_response", "n_receptors"):
            assert key in result.metadata

    def test_n_receptors_metadata(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.metadata["n_receptors"] == human_mosaic.n_receptors


# ---------------------------------------------------------------------------
# TestResponseBounds
# ---------------------------------------------------------------------------

class TestResponseBounds:
    def test_responses_in_unit_interval(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert np.all(result.responses >= 0.0)
        assert np.all(result.responses <= 1.0)

    def test_no_nan_or_inf(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert np.all(np.isfinite(result.responses))

    def test_float32_dtype(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.01)
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.responses.dtype == np.float32


# ---------------------------------------------------------------------------
# TestZeroInput
# ---------------------------------------------------------------------------

class TestZeroInput:
    def test_zero_irradiance_gives_zero_response(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.0)
        result = human_stage.compute_response(human_mosaic, irr)
        np.testing.assert_array_equal(result.responses, 0.0)

    def test_zero_response_metadata(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = _make_irradiance(0.0)
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.metadata["mean_excitation"] == pytest.approx(0.0, abs=1e-9)
        assert result.metadata["mean_response"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestMonotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_brighter_gives_higher_mean_response(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr_low = _make_irradiance(0.001)
        irr_high = _make_irradiance(0.1)
        r_low = human_stage.compute_response(human_mosaic, irr_low)
        r_high = human_stage.compute_response(human_mosaic, irr_high)
        assert r_high.metadata["mean_response"] > r_low.metadata["mean_response"]

    def test_response_below_r_max_at_moderate_input(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        # At moderate light levels responses should not saturate to 1.0
        irr = _make_irradiance(0.005)
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.metadata["mean_response"] < 0.99


# ---------------------------------------------------------------------------
# TestSpectralSelectivity
# ---------------------------------------------------------------------------

class TestSpectralSelectivity:
    """Red light should drive L/M cones more than S cones; blue light the reverse."""

    def _cone_responses_by_type(
        self, mosaic: PhotoreceptorMosaic, activation: MosaicActivation
    ) -> dict:
        """Return mean response per cone type."""
        out: dict = {}
        for ctype in np.unique(mosaic.types):
            mask = mosaic.types == ctype
            out[ctype] = float(np.mean(activation.responses[mask]))
        return out

    def test_red_stimulus_drives_l_more_than_s_human(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        # Red band: high energy only above 600 nm
        spectrum = np.zeros(N_WL, dtype=np.float32)
        red_idx = WL >= 600
        spectrum[red_idx] = 0.05
        irr = _make_spectral_irradiance(spectrum)
        result = human_stage.compute_response(human_mosaic, irr)
        means = self._cone_responses_by_type(human_mosaic, result)
        assert means.get("L_cone", 0.0) > means.get("S_cone", 0.0)

    def test_blue_stimulus_drives_s_more_than_l_human(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        # Blue band: high energy only below 460 nm
        spectrum = np.zeros(N_WL, dtype=np.float32)
        blue_idx = WL <= 460
        spectrum[blue_idx] = 0.05
        irr = _make_spectral_irradiance(spectrum)
        result = human_stage.compute_response(human_mosaic, irr)
        means = self._cone_responses_by_type(human_mosaic, result)
        assert means.get("S_cone", 0.0) > means.get("L_cone", 0.0)

    def test_red_stimulus_drives_l_more_than_s_dog(
        self, dog_stage: RetinalStage
    ) -> None:
        mosaic = dog_stage.generate_mosaic(seed=42)
        spectrum = np.zeros(N_WL, dtype=np.float32)
        spectrum[WL >= 600] = 0.05
        irr = _make_spectral_irradiance(spectrum)
        result = dog_stage.compute_response(mosaic, irr)
        means = self._cone_responses_by_type(mosaic, result)
        # Dog is dichromat: L_cone and S_cone
        assert means.get("L_cone", 0.0) > means.get("S_cone", 0.0)

    def test_direct_unfiltered_blue_stimulus_is_filtered_once_by_retinal_stage(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        spectrum = np.zeros(N_WL, dtype=np.float32)
        spectrum[WL <= 460] = 0.05
        raw = _make_spectral_irradiance(spectrum)
        raw.metadata["media_transmission_applied"] = False
        filtered = _make_spectral_irradiance(spectrum * human_stage._op.media_transmission(WL))
        filtered.metadata["media_transmission_applied"] = True

        result_raw = human_stage.compute_response(human_mosaic, raw)
        result_filtered = human_stage.compute_response(human_mosaic, filtered)
        np.testing.assert_allclose(
            result_raw.responses,
            result_filtered.responses,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_media_filtering_reduces_short_wavelength_excitation(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        spectrum = np.zeros(N_WL, dtype=np.float32)
        spectrum[WL <= 460] = 0.05
        unfiltered = _make_spectral_irradiance(spectrum)
        unfiltered.metadata["media_transmission_applied"] = True
        fallback = _make_spectral_irradiance(spectrum)
        fallback.metadata["media_transmission_applied"] = False

        no_filter = human_stage.compute_response(human_mosaic, unfiltered)
        with_filter = human_stage.compute_response(human_mosaic, fallback)
        assert with_filter.metadata["mean_excitation"] < no_filter.metadata["mean_excitation"]


# ---------------------------------------------------------------------------
# TestRodVsCone
# ---------------------------------------------------------------------------

class TestRodVsCone:
    """Rods have lower half-saturation (sigma=0.1) so they saturate earlier."""

    def test_rods_saturate_faster_than_cones(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        # At medium-high irradiance rods should be more saturated than cones.
        # The human 2° patch extends ±0.39 mm; use 200×200 at 5 µm/px (±0.50 mm)
        # so peripheral rods are not zeroed by the out-of-bounds mask (CR-16).
        irr = _make_irradiance(0.05, H=200, W=200)
        result = human_stage.compute_response(human_mosaic, irr)
        rod_mask = human_mosaic.types == "rod"
        cone_mask = np.char.endswith(human_mosaic.types, "_cone")
        if rod_mask.any() and cone_mask.any():
            mean_rod = float(np.mean(result.responses[rod_mask]))
            mean_cone = float(np.mean(result.responses[cone_mask]))
            assert mean_rod > mean_cone, (
                f"Rods ({mean_rod:.3f}) should saturate faster than cones ({mean_cone:.3f})"
            )


# ---------------------------------------------------------------------------
# TestPixelScaleFallbacks
# ---------------------------------------------------------------------------

class TestPixelScaleFallbacks:
    def test_scene_mm_per_pixel_used(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        """Scene mm_per_pixel overrides irradiance metadata."""
        class FakeScene:
            mm_per_pixel = (0.005, 0.005)

        irr = RetinalIrradiance(
            data=np.full((64, 64, N_WL), 0.01, dtype=np.float32),
            wavelengths=WL.astype(float),
            metadata={},  # no pixel_scale_mm
        )
        result = human_stage.compute_response(human_mosaic, irr, scene=FakeScene())
        assert result.metadata["mm_per_pixel"] == pytest.approx(0.005)

    def test_irradiance_metadata_fallback(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        irr = RetinalIrradiance(
            data=np.full((64, 64, N_WL), 0.01, dtype=np.float32),
            wavelengths=WL.astype(float),
            metadata={"pixel_scale_mm": 0.003},
        )
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.metadata["mm_per_pixel"] == pytest.approx(0.003)

    def test_patch_extent_fallback(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        """When no pixel scale is available, fallback uses patch-extent estimate."""
        irr = RetinalIrradiance(
            data=np.full((64, 64, N_WL), 0.01, dtype=np.float32),
            wavelengths=WL.astype(float),
            metadata={},
        )
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.metadata["mm_per_pixel"] > 0.0
        assert np.all(np.isfinite(result.responses))


# ---------------------------------------------------------------------------
# TestWavelengthMismatch
# ---------------------------------------------------------------------------

class TestWavelengthMismatch:
    def test_coarser_wavelength_grid_handled(
        self, human_stage: RetinalStage, human_mosaic: PhotoreceptorMosaic
    ) -> None:
        """Irradiance on a 10 nm grid should still produce finite responses."""
        wl_coarse = np.arange(380, 721, 10, dtype=float)  # 35 bands
        data = np.full((32, 32, len(wl_coarse)), 0.01, dtype=np.float32)
        irr = RetinalIrradiance(
            data=data,
            wavelengths=wl_coarse,
            metadata={"pixel_scale_mm": 0.005},
        )
        result = human_stage.compute_response(human_mosaic, irr)
        assert result.responses.shape == (human_mosaic.n_receptors,)
        assert np.all(np.isfinite(result.responses))
        assert np.all(result.responses >= 0.0)


# ---------------------------------------------------------------------------
# TestAllSpecies
# ---------------------------------------------------------------------------

class TestAllSpecies:
    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_species_pipeline(self, species: str) -> None:
        cfg = SpeciesConfig.load(species)
        stage = RetinalStage(cfg.retinal, cfg.optical)
        mosaic = stage.generate_mosaic(seed=0)
        irr = _make_irradiance(0.01)
        result = stage.compute_response(mosaic, irr)
        assert isinstance(result, MosaicActivation)
        assert result.responses.shape == (mosaic.n_receptors,)
        assert np.all(result.responses >= 0.0)
        assert np.all(result.responses <= 1.0)
        assert np.all(np.isfinite(result.responses))

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_species_zero_input(self, species: str) -> None:
        cfg = SpeciesConfig.load(species)
        stage = RetinalStage(cfg.retinal, cfg.optical)
        mosaic = stage.generate_mosaic(seed=0)
        irr = _make_irradiance(0.0)
        result = stage.compute_response(mosaic, irr)
        np.testing.assert_array_equal(result.responses, 0.0)
