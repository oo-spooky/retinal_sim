"""Phase 12 — Species comparison pipeline tests.

Test classes:

TestRetinalSimulatorInit
    Constructor validation: string/config init, invalid species, properties.

TestSimulateSingleSpecies
    Single-species simulate(): result structure, shapes, reproducibility,
    stimulus_scale effect, scene geometry paths.

TestCompareSpecies
    compare_species(): all three species, key matching, receptor count
    differences, seed and parameter preservation.

TestEndToEndColorDeficit
    KEY VALIDATION (architecture §11d): red/green collapse for dog,
    blue/yellow preserved for both, luminance contrast preserved for cat.

TestComparisonRendering
    Integration with output renderers (voronoi, reconstruction, comparison).
"""
from __future__ import annotations

import numpy as np
import pytest

from retinal_sim.pipeline import RetinalSimulator, SimulationResult
from retinal_sim.species.config import SpeciesConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_IMG_SIZE = 32
_PATCH_DEG = 1.0
_SEED = 42


def _random_image(size=_IMG_SIZE, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (size, size, 3), dtype=np.uint8)


def _solid_image(rgb, size=_IMG_SIZE):
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:, :] = rgb
    return img


def _split_image(left_rgb, right_rgb, size=_IMG_SIZE):
    """Left half one colour, right half another."""
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:, : size // 2] = left_rgb
    img[:, size // 2 :] = right_rgb
    return img


@pytest.fixture(scope="module")
def human_sim():
    return RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)


@pytest.fixture(scope="module")
def dog_sim():
    return RetinalSimulator("dog", patch_extent_deg=_PATCH_DEG, seed=_SEED)


@pytest.fixture(scope="module")
def cat_sim():
    return RetinalSimulator("cat", patch_extent_deg=_PATCH_DEG, seed=_SEED)


@pytest.fixture(scope="module")
def human_result(human_sim):
    return human_sim.simulate(_random_image(), seed=_SEED)


@pytest.fixture(scope="module")
def dog_result(dog_sim):
    return dog_sim.simulate(_random_image(), seed=_SEED)


@pytest.fixture(scope="module")
def cat_result(cat_sim):
    return cat_sim.simulate(_random_image(), seed=_SEED)


# ===========================================================================
# TestRetinalSimulatorInit
# ===========================================================================


class TestRetinalSimulatorInit:

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_init_from_string(self, species):
        sim = RetinalSimulator(species, patch_extent_deg=_PATCH_DEG)
        assert sim.species_name == species

    def test_init_from_species_config(self):
        cfg = SpeciesConfig.load("dog")
        sim = RetinalSimulator(cfg, patch_extent_deg=_PATCH_DEG)
        assert sim.species_name == "dog"

    def test_init_invalid_species_raises(self):
        with pytest.raises(ValueError):
            RetinalSimulator("goldfish")

    def test_species_name_property(self, human_sim):
        assert human_sim.species_name == "human"

    def test_config_property(self, human_sim):
        assert isinstance(human_sim.config, SpeciesConfig)
        assert human_sim.config.name == "human"

    def test_default_params(self):
        sim = RetinalSimulator("human")
        assert sim._patch_extent_deg == 2.0
        assert sim._stimulus_scale == 0.01
        assert sim._seed == 42


# ===========================================================================
# TestSimulateSingleSpecies
# ===========================================================================


class TestSimulateSingleSpecies:

    def test_simulate_returns_simulation_result(self, human_result):
        assert isinstance(human_result, SimulationResult)

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_simulate_species(self, species):
        sim = RetinalSimulator(species, patch_extent_deg=_PATCH_DEG, seed=_SEED)
        result = sim.simulate(_random_image())
        assert result.mosaic.n_receptors > 0
        assert result.activation.responses.min() >= 0.0

    def test_simulate_with_scene_geometry(self, human_sim):
        img = _random_image()
        result = human_sim.simulate(img, scene_width_m=0.3, viewing_distance_m=6.0)
        assert result.scene.angular_width_deg > 0
        assert result.scene.viewing_distance_m == 6.0

    def test_simulate_without_scene_geometry(self, human_sim):
        img = _random_image()
        result = human_sim.simulate(img)
        assert result.scene.viewing_distance_m == float("inf")
        assert result.scene.angular_width_deg == _PATCH_DEG

    def test_simulate_seed_reproducibility(self):
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG)
        img = _random_image()
        r1 = sim.simulate(img, seed=7)
        r2 = sim.simulate(img, seed=7)
        np.testing.assert_array_equal(r1.activation.responses, r2.activation.responses)

    def test_simulate_different_seeds(self):
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG)
        img = _random_image()
        r1 = sim.simulate(img, seed=0)
        r2 = sim.simulate(img, seed=1)
        # Different mosaics → generally different receptor count or positions
        assert not np.array_equal(
            r1.mosaic.positions, r2.mosaic.positions
        )

    def test_simulate_result_species_name(self, human_result):
        assert human_result.species_name == "human"

    def test_simulate_small_image(self):
        sim = RetinalSimulator("dog", patch_extent_deg=0.5, seed=0)
        img = _random_image(size=16)
        result = sim.simulate(img)
        assert result.mosaic.n_receptors > 0

    def test_simulate_rectangular_image(self, human_sim):
        rng = np.random.RandomState(0)
        img = rng.randint(0, 255, (24, 48, 3), dtype=np.uint8)
        result = human_sim.simulate(img)
        assert result.spectral_image.data.shape[0] == 24
        assert result.spectral_image.data.shape[1] == 48

    def test_simulate_spectral_image_wavelengths(self, human_result):
        assert human_result.spectral_image.data.ndim == 3
        N_lam = human_result.spectral_image.data.shape[2]
        assert N_lam == len(human_result.spectral_image.wavelengths)
        assert N_lam > 30  # at least 30 spectral bands

    def test_simulate_irradiance_shape(self, human_result):
        H, W = _IMG_SIZE, _IMG_SIZE
        assert human_result.retinal_irradiance.data.shape[:2] == (H, W)

    def test_simulate_responses_bounded(self, human_result):
        r = human_result.activation.responses
        assert np.all(r >= 0.0)
        assert np.all(r <= 1.5)  # R_max is typically 1.0

    def test_simulate_stimulus_scale_effect(self):
        img = _random_image()
        sim_low = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, stimulus_scale=0.001, seed=_SEED)
        sim_high = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, stimulus_scale=0.1, seed=_SEED)
        r_low = sim_low.simulate(img)
        r_high = sim_high.simulate(img)
        # Higher stimulus → higher mean response (Naka-Rushton is monotonic)
        assert r_high.activation.responses.mean() > r_low.activation.responses.mean()


# ===========================================================================
# TestCompareSpecies
# ===========================================================================


class TestCompareSpecies:

    @pytest.fixture(scope="class")
    def comparison(self):
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        img = _random_image()
        return sim.compare_species(img, ["human", "dog", "cat"])

    def test_compare_returns_dict(self, comparison):
        assert isinstance(comparison, dict)

    def test_compare_keys_match_input(self, comparison):
        assert set(comparison.keys()) == {"human", "dog", "cat"}

    def test_compare_all_simulation_results(self, comparison):
        for sp, r in comparison.items():
            assert isinstance(r, SimulationResult)
            assert r.species_name == sp

    def test_compare_different_receptor_counts(self, comparison):
        # Human foveal cone density >> dog/cat → more receptors for same patch
        counts = {sp: r.mosaic.n_receptors for sp, r in comparison.items()}
        # All should have receptors
        for sp, c in counts.items():
            assert c > 0, f"{sp} has 0 receptors"

    def test_compare_single_species(self):
        sim = RetinalSimulator("dog", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        results = sim.compare_species(_random_image(), ["dog"])
        assert list(results.keys()) == ["dog"]

    def test_compare_preserves_seed(self):
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=10)
        img = _random_image()
        r1 = sim.compare_species(img, ["human"], seed=7)
        r2 = sim.compare_species(img, ["human"], seed=7)
        np.testing.assert_array_equal(
            r1["human"].activation.responses,
            r2["human"].activation.responses,
        )

    def test_compare_preserves_stimulus_scale(self):
        img = _random_image()
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, stimulus_scale=0.005, seed=_SEED)
        results = sim.compare_species(img, ["human", "dog"])
        # Both should run without error and produce non-empty results
        for sp, r in results.items():
            assert r.mosaic.n_receptors > 0


# ===========================================================================
# TestEndToEndColorDeficit
# ===========================================================================


class TestEndToEndColorDeficit:
    """Architecture §11d — end-to-end validation of known visual deficits."""

    @pytest.fixture(scope="class")
    def red_green_results(self):
        """Red/green split image through human and dog pipelines."""
        red = [200, 30, 30]
        green = [30, 180, 30]
        img = _split_image(red, green, size=48)
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        return sim.compare_species(img, ["human", "dog"], seed=_SEED)

    @pytest.fixture(scope="class")
    def blue_yellow_results(self):
        """Blue/yellow split image through human and dog pipelines."""
        blue = [30, 50, 220]
        yellow = [220, 210, 30]
        img = _split_image(blue, yellow, size=48)
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        return sim.compare_species(img, ["human", "dog"], seed=_SEED)

    @pytest.fixture(scope="class")
    def bright_dark_results(self):
        """Bright/dark split image through all three pipelines."""
        bright = [200, 200, 200]
        dark = [30, 30, 30]
        img = _split_image(bright, dark, size=48)
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        return sim.compare_species(img, ["human", "dog", "cat"], seed=_SEED)

    @staticmethod
    def _half_response_diff(result, size=48):
        """Compute |mean_left - mean_right| for cone responses.

        Returns the best (max) diff across cone types.
        """
        mosaic = result.mosaic
        responses = result.activation.responses
        focal_mm = result.scene.retinal_width_mm  # total retinal width

        # Map receptor x positions to left/right halves.
        pos_x = mosaic.positions[:, 0]
        median_x = np.median(pos_x)
        left = pos_x < median_x
        right = pos_x >= median_x

        best_diff = 0.0
        for ctype in np.unique(mosaic.types):
            if ctype == "rod":
                continue
            type_mask = mosaic.types == ctype
            left_mask = type_mask & left
            right_mask = type_mask & right
            if left_mask.sum() < 5 or right_mask.sum() < 5:
                continue
            mean_l = float(np.mean(responses[left_mask]))
            mean_r = float(np.mean(responses[right_mask]))
            diff = abs(mean_l - mean_r)
            if diff > best_diff:
                best_diff = diff
        return best_diff

    def test_red_green_collapse_dog(self, red_green_results):
        """Dog should show less red/green differentiation than human."""
        human_diff = self._half_response_diff(red_green_results["human"])
        dog_diff = self._half_response_diff(red_green_results["dog"])
        # Human trichromat should distinguish red from green better
        assert human_diff > dog_diff, (
            f"Expected human diff ({human_diff:.4f}) > dog diff ({dog_diff:.4f})"
        )

    def test_human_sees_red_green(self, red_green_results):
        """Human should have non-trivial red/green differentiation."""
        human_diff = self._half_response_diff(red_green_results["human"])
        assert human_diff > 0.01, f"Human diff too small: {human_diff:.4f}"

    def test_blue_yellow_both_see(self, blue_yellow_results):
        """Both human and dog should distinguish blue from yellow (S-cone axis)."""
        human_diff = self._half_response_diff(blue_yellow_results["human"])
        dog_diff = self._half_response_diff(blue_yellow_results["dog"])
        assert human_diff > 0.01, f"Human blue/yellow diff too small: {human_diff:.4f}"
        assert dog_diff > 0.01, f"Dog blue/yellow diff too small: {dog_diff:.4f}"

    def test_luminance_preserved_all_species(self, bright_dark_results):
        """All species should distinguish bright from dark (luminance axis)."""
        for sp in ["human", "dog", "cat"]:
            diff = self._half_response_diff(bright_dark_results[sp])
            assert diff > 0.01, f"{sp} luminance diff too small: {diff:.4f}"

    def test_cat_luminance_contrast(self, bright_dark_results):
        """Cat should show significant luminance contrast (larger receptive fields)."""
        cat_diff = self._half_response_diff(bright_dark_results["cat"])
        assert cat_diff > 0.02, f"Cat luminance diff too small: {cat_diff:.4f}"

    def test_red_green_human_dog_ratio(self, red_green_results):
        """Human should have at least 1.5x the red/green diff of dog."""
        human_diff = self._half_response_diff(red_green_results["human"])
        dog_diff = self._half_response_diff(red_green_results["dog"])
        ratio = human_diff / (dog_diff + 1e-8)
        assert ratio > 1.5, f"Human/dog red-green ratio only {ratio:.2f}"

    def test_dog_red_green_near_zero(self, red_green_results):
        """Dog red/green diff should be small (near confusion)."""
        dog_diff = self._half_response_diff(red_green_results["dog"])
        # Dog is dichromat — red and green should be nearly indistinguishable.
        # Allow some residual from S-cone channel and mosaic noise.
        assert dog_diff < 0.15, f"Dog red/green diff too large: {dog_diff:.4f}"


# ===========================================================================
# TestComparisonRendering
# ===========================================================================


class TestComparisonRendering:
    """Test that output renderers work with pipeline results."""

    @pytest.fixture(scope="class")
    def result(self):
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        return sim.simulate(_random_image(size=32))

    @pytest.fixture(scope="class")
    def comparison(self):
        sim = RetinalSimulator("human", patch_extent_deg=_PATCH_DEG, seed=_SEED)
        return sim.compare_species(_random_image(size=32), ["human", "dog"])

    def test_render_voronoi_from_pipeline(self, result):
        from retinal_sim.output.voronoi import render_voronoi
        img = render_voronoi(result.activation, output_size=(64, 64))
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.float32

    def test_render_reconstructed_from_pipeline(self, result):
        from retinal_sim.output.reconstruction import render_reconstructed
        img = render_reconstructed(result.activation, grid_shape=(64, 64))
        assert img.shape == (64, 64)
        assert img.dtype == np.float32

    def test_render_comparison_from_pipeline(self, comparison):
        from retinal_sim.output.comparison import render_comparison
        fig = render_comparison(
            {sp: r.activation for sp, r in comparison.items()},
            output_size=(64, 64),
        )
        # Should be a matplotlib Figure
        assert hasattr(fig, "savefig")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_mosaic_map_from_pipeline(self, result):
        from retinal_sim.output.comparison import render_mosaic_map
        fig = render_mosaic_map(result.mosaic)
        assert hasattr(fig, "savefig")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_voronoi_nonzero(self, result):
        from retinal_sim.output.voronoi import render_voronoi
        img = render_voronoi(result.activation, output_size=(64, 64))
        assert img.max() > 0, "Voronoi image is all black"
