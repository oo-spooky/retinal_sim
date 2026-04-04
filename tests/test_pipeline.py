"""Phase 12: species comparison pipeline tests.

Tests are grouped into five classes:

TestRetinalSimulatorInit
    Constructor validation: string/config initialisation, invalid species,
    exposed properties, and default parameters.

TestSimulateSingleSpecies
    Single-species ``simulate()`` integration tests: result type, per-species
    execution, geometry paths, reproducibility, shapes, bounded responses, and
    stimulus-scale sensitivity.

TestCompareSpecies
    Multi-species ``compare_species()`` tests: dict structure, key preservation,
    species-specific mosaics, and propagation of seed / stimulus-scale.

TestEndToEndColorDeficit
    Key Phase 12 validation (architecture section 11d): red-green collapse in
    dog relative to human, blue-yellow preserved, human confusion-pair
    discriminability above dog, and cat luminance contrast preserved.

TestComparisonRendering
    Output-renderer integration using pipeline-generated activations and mosaics.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from retinal_sim.output.comparison import render_comparison, render_mosaic_map
from retinal_sim.output.reconstruction import render_reconstructed
from retinal_sim.output.voronoi import render_voronoi
from retinal_sim.pipeline import RetinalSimulator, SimulationResult
from retinal_sim.species.config import SpeciesConfig
from retinal_sim.validation.datasets import control_pairs
from retinal_sim.validation.metrics import split_half_discriminability
from retinal_sim.validation.ishihara import find_confusion_pair, make_dot_pattern


PATCH_EXTENT_DEG = 1.0
IMAGE_SIZE = 32
VALIDATION_IMAGE_SIZE = 48
STIMULUS_SCALE = 0.01
SEED = 42


def _random_image(height: int = IMAGE_SIZE, width: int = IMAGE_SIZE, seed: int = 0) -> np.ndarray:
    """Return a deterministic uint8 RGB image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _split_image(
    left_rgb: tuple[int, int, int] | list[int],
    right_rgb: tuple[int, int, int] | list[int],
    size: int = VALIDATION_IMAGE_SIZE,
) -> np.ndarray:
    """Return an image with one colour on each half."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, : size // 2] = np.asarray(left_rgb, dtype=np.uint8)
    img[:, size // 2 :] = np.asarray(right_rgb, dtype=np.uint8)
    return img


def _pixel_coordinates(result: SimulationResult, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map receptor positions back to nearest image pixel indices."""
    height, width = image_shape
    mm_per_pixel = float(result.scene.mm_per_pixel[0])
    positions = result.mosaic.positions
    cols = np.clip(
        np.round(positions[:, 0] / mm_per_pixel + (width - 1) / 2.0).astype(int),
        0,
        width - 1,
    )
    rows = np.clip(
        np.round(positions[:, 1] / mm_per_pixel + (height - 1) / 2.0).astype(int),
        0,
        height - 1,
    )
    return rows, cols


def _mean_response_in_half(result: SimulationResult, receptor_type: str, left_half: bool, width: int) -> float:
    """Return mean response for one receptor type in one image half."""
    _, cols = _pixel_coordinates(result, (width, width))
    type_mask = result.mosaic.types == receptor_type
    half_mask = cols < (width // 2) if left_half else cols >= (width // 2)
    mask = type_mask & half_mask
    if not np.any(mask):
        return float("nan")
    return float(np.mean(result.activation.responses[mask]))


def _cone_vector_difference(result: SimulationResult, width: int) -> float:
    """Return Euclidean difference between left/right mean cone-response vectors."""
    _, cols = _pixel_coordinates(result, (width, width))
    left_means: list[float] = []
    right_means: list[float] = []

    for receptor_type in ("L_cone", "M_cone", "S_cone"):
        type_mask = result.mosaic.types == receptor_type
        left_mask = type_mask & (cols < (width // 2))
        right_mask = type_mask & (cols >= (width // 2))
        if not np.any(left_mask) or not np.any(right_mask):
            continue
        left_means.append(float(np.mean(result.activation.responses[left_mask])))
        right_means.append(float(np.mean(result.activation.responses[right_mask])))

    if not left_means:
        return 0.0

    return float(np.linalg.norm(np.asarray(left_means) - np.asarray(right_means)))


def _best_figure_background_discriminability(
    result: SimulationResult,
    figure_mask: np.ndarray,
) -> float:
    """Return best cone-type figure/background discriminability for one result."""
    rows, cols = _pixel_coordinates(result, figure_mask.shape)
    in_figure = figure_mask[rows, cols]
    best_d = 0.0

    for receptor_type in np.unique(result.mosaic.types):
        if receptor_type == "rod":
            continue
        type_mask = result.mosaic.types == receptor_type
        figure_type_mask = type_mask & in_figure
        background_type_mask = type_mask & ~in_figure
        if figure_type_mask.sum() < 5 or background_type_mask.sum() < 5:
            continue
        mean_figure = float(np.mean(result.activation.responses[figure_type_mask]))
        mean_background = float(np.mean(result.activation.responses[background_type_mask]))
        d_value = abs(mean_figure - mean_background) / (mean_figure + mean_background + 1e-8)
        best_d = max(best_d, d_value)

    return best_d


@pytest.fixture(scope="module")
def human_config() -> SpeciesConfig:
    return SpeciesConfig.load("human")


@pytest.fixture(scope="module")
def human_sim() -> RetinalSimulator:
    return RetinalSimulator(
        "human",
        patch_extent_deg=PATCH_EXTENT_DEG,
        stimulus_scale=STIMULUS_SCALE,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def dog_sim() -> RetinalSimulator:
    return RetinalSimulator(
        "dog",
        patch_extent_deg=PATCH_EXTENT_DEG,
        stimulus_scale=STIMULUS_SCALE,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def cat_sim() -> RetinalSimulator:
    return RetinalSimulator(
        "cat",
        patch_extent_deg=PATCH_EXTENT_DEG,
        stimulus_scale=STIMULUS_SCALE,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def base_image() -> np.ndarray:
    return _random_image(seed=11)


@pytest.fixture(scope="module")
def human_result(human_sim: RetinalSimulator, base_image: np.ndarray) -> SimulationResult:
    return human_sim.simulate(base_image, seed=SEED)


@pytest.fixture(scope="module")
def dog_result(dog_sim: RetinalSimulator, base_image: np.ndarray) -> SimulationResult:
    return dog_sim.simulate(base_image, seed=SEED)


@pytest.fixture(scope="module")
def cat_result(cat_sim: RetinalSimulator, base_image: np.ndarray) -> SimulationResult:
    return cat_sim.simulate(base_image, seed=SEED)


@pytest.fixture(scope="module")
def comparison_results(human_sim: RetinalSimulator, base_image: np.ndarray) -> dict[str, SimulationResult]:
    return human_sim.compare_species(base_image, ["human", "dog", "cat"], seed=SEED)


@pytest.fixture(scope="module")
def red_green_image() -> np.ndarray:
    return _split_image((200, 30, 30), (30, 180, 30))


@pytest.fixture(scope="module")
def red_green_results(human_sim: RetinalSimulator, red_green_image: np.ndarray) -> dict[str, SimulationResult]:
    return human_sim.compare_species(red_green_image, ["human", "dog"], seed=SEED)


@pytest.fixture(scope="module")
def blue_yellow_image() -> np.ndarray:
    return _split_image((30, 50, 220), (220, 210, 30))


@pytest.fixture(scope="module")
def blue_yellow_results(
    human_sim: RetinalSimulator,
    blue_yellow_image: np.ndarray,
) -> dict[str, SimulationResult]:
    return human_sim.compare_species(blue_yellow_image, ["human", "dog"], seed=SEED)


@pytest.fixture(scope="module")
def confusion_pair() -> tuple[np.ndarray, np.ndarray]:
    return find_confusion_pair(species="dog", n_candidates=400, seed=7)


@pytest.fixture(scope="module")
def confusion_pair_pattern(
    confusion_pair: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    fg_rgb, bg_rgb = confusion_pair
    return make_dot_pattern(
        fg_rgb,
        bg_rgb,
        image_size_px=VALIDATION_IMAGE_SIZE,
        n_dots=120,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def confusion_pair_results(
    human_sim: RetinalSimulator,
    confusion_pair_pattern: tuple[np.ndarray, np.ndarray],
) -> dict[str, SimulationResult]:
    image, _ = confusion_pair_pattern
    return human_sim.compare_species(image, ["human", "dog"], seed=SEED)


@pytest.fixture(scope="module")
def bright_dark_image() -> np.ndarray:
    return _split_image((220, 220, 220), (20, 20, 20))


@pytest.fixture(scope="module")
def bright_dark_results(
    human_sim: RetinalSimulator,
    bright_dark_image: np.ndarray,
) -> dict[str, SimulationResult]:
    return human_sim.compare_species(bright_dark_image, ["cat"], seed=SEED)


class TestRetinalSimulatorInit:
    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_init_from_string(self, species: str):
        sim = RetinalSimulator(species, patch_extent_deg=PATCH_EXTENT_DEG)
        assert sim.species_name == species

    def test_init_from_species_config(self, human_config: SpeciesConfig):
        sim = RetinalSimulator(human_config, patch_extent_deg=PATCH_EXTENT_DEG)
        assert sim.species_name == human_config.name
        assert sim.config is human_config

    def test_init_invalid_species_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            RetinalSimulator("goldfish")

    def test_species_name_property(self, dog_sim: RetinalSimulator):
        assert dog_sim.species_name == "dog"

    def test_config_property(self, cat_sim: RetinalSimulator):
        assert isinstance(cat_sim.config, SpeciesConfig)
        assert cat_sim.config.name == "cat"

    def test_default_params(self):
        sim = RetinalSimulator("human")
        assert sim._patch_extent_deg == 2.0
        assert sim._light_level == "photopic"
        assert sim._stimulus_scale == 0.01
        assert sim._seed == 42


class TestSimulateSingleSpecies:
    def test_simulate_returns_simulation_result(self, human_result: SimulationResult):
        assert isinstance(human_result, SimulationResult)

    def test_simulate_human(self, human_result: SimulationResult):
        assert human_result.mosaic.n_receptors > 0
        assert np.all(human_result.activation.responses >= 0.0)
        assert np.all(human_result.activation.responses <= 1.0)

    def test_simulate_dog(self, dog_result: SimulationResult):
        assert dog_result.mosaic.n_receptors > 0
        assert np.all(dog_result.activation.responses >= 0.0)
        assert np.all(dog_result.activation.responses <= 1.0)

    def test_simulate_cat(self, cat_result: SimulationResult):
        assert cat_result.mosaic.n_receptors > 0
        assert np.all(cat_result.activation.responses >= 0.0)
        assert np.all(cat_result.activation.responses <= 1.0)

    def test_simulate_with_scene_geometry(self, human_sim: RetinalSimulator):
        image = _random_image(seed=2)
        result = human_sim.simulate(image, scene_width_m=0.3, viewing_distance_m=6.0, seed=SEED)
        assert result.scene.scene_width_m == 0.3
        assert result.scene.viewing_distance_m == 6.0
        assert result.scene.angular_width_deg > 0.0
        assert result.scene.retinal_width_mm > 0.0

    def test_simulate_without_scene_geometry(self, human_sim: RetinalSimulator):
        image = _random_image(seed=3)
        result = human_sim.simulate(image, scene_width_m=None, seed=SEED)
        assert math.isinf(result.scene.viewing_distance_m)
        assert result.scene.angular_width_deg == PATCH_EXTENT_DEG
        assert result.scene.scene_covers_patch_fraction == 1.0

    def test_simulate_seed_reproducibility(self, human_sim: RetinalSimulator, base_image: np.ndarray):
        result_a = human_sim.simulate(base_image, seed=7)
        result_b = human_sim.simulate(base_image, seed=7)
        np.testing.assert_array_equal(result_a.mosaic.positions, result_b.mosaic.positions)
        np.testing.assert_array_equal(result_a.mosaic.types, result_b.mosaic.types)
        np.testing.assert_allclose(result_a.activation.responses, result_b.activation.responses)

    def test_simulate_different_seeds(self, human_sim: RetinalSimulator, base_image: np.ndarray):
        result_a = human_sim.simulate(base_image, seed=0)
        result_b = human_sim.simulate(base_image, seed=1)
        assert not np.array_equal(result_a.mosaic.positions, result_b.mosaic.positions)

    def test_simulate_result_species_name(self, dog_result: SimulationResult):
        assert dog_result.species_name == "dog"

    def test_simulate_small_image(self, human_sim: RetinalSimulator):
        image = _random_image(height=16, width=16, seed=4)
        result = human_sim.simulate(image, seed=SEED)
        assert result.spectral_image.data.shape[:2] == (16, 16)
        assert result.mosaic.n_receptors > 0

    def test_simulate_rectangular_image(self, human_sim: RetinalSimulator):
        image = _random_image(height=16, width=48, seed=5)
        result = human_sim.simulate(image, seed=SEED)
        assert result.spectral_image.data.shape[:2] == (16, 48)
        assert result.retinal_irradiance.data.shape[:2] == (16, 48)

    def test_simulate_spectral_image_shape(self, human_result: SimulationResult):
        height, width, n_wavelengths = human_result.spectral_image.data.shape
        assert (height, width) == (IMAGE_SIZE, IMAGE_SIZE)
        assert n_wavelengths == len(human_result.spectral_image.wavelengths)
        assert n_wavelengths > 30

    def test_simulate_irradiance_shape(self, human_result: SimulationResult):
        assert human_result.retinal_irradiance.data.shape == human_result.spectral_image.data.shape

    def test_simulate_responses_bounded(self, cat_result: SimulationResult):
        responses = cat_result.activation.responses
        assert np.all(responses >= 0.0)
        assert np.all(responses <= 1.0)

    def test_simulate_stimulus_scale_effect(self, base_image: np.ndarray):
        sim_low = RetinalSimulator(
            "human",
            patch_extent_deg=PATCH_EXTENT_DEG,
            stimulus_scale=0.001,
            seed=SEED,
        )
        sim_high = RetinalSimulator(
            "human",
            patch_extent_deg=PATCH_EXTENT_DEG,
            stimulus_scale=0.1,
            seed=SEED,
        )
        result_low = sim_low.simulate(base_image, seed=SEED)
        result_high = sim_high.simulate(base_image, seed=SEED)
        assert result_high.activation.responses.mean() > result_low.activation.responses.mean()

    def test_simulate_activation_count_matches_mosaic(self, human_result: SimulationResult):
        assert human_result.activation.responses.shape == (human_result.mosaic.n_receptors,)

    def test_simulate_activation_metadata_present(self, human_result: SimulationResult):
        assert "n_receptors" in human_result.activation.metadata
        assert human_result.activation.metadata["n_receptors"] == human_result.mosaic.n_receptors

    def test_simulate_populates_artifacts_and_summary_metrics(self, human_result: SimulationResult):
        assert "stimulated_receptor_mask" in human_result.artifacts
        assert human_result.artifacts["stimulated_receptor_mask"].shape == (human_result.mosaic.n_receptors,)
        assert "stimulated_receptor_count" in human_result.summary_metrics
        assert human_result.summary_metrics["stimulated_receptor_count"] > 0

    def test_simulate_projected_scene_reduces_stimulated_receptors(self, human_sim: RetinalSimulator):
        image = _random_image(seed=12)
        near = human_sim.simulate(image, scene_width_m=0.02, viewing_distance_m=1.0, seed=SEED)
        far = human_sim.simulate(image, scene_width_m=0.02, viewing_distance_m=8.0, seed=SEED)
        assert near.summary_metrics["stimulated_receptor_count"] > far.summary_metrics["stimulated_receptor_count"]


class TestCompareSpecies:
    def test_compare_returns_dict_with_all_species(self, comparison_results: dict[str, SimulationResult]):
        assert isinstance(comparison_results, dict)
        assert set(comparison_results) == {"human", "dog", "cat"}

    def test_compare_human_dog_cat(self, comparison_results: dict[str, SimulationResult]):
        for species in ("human", "dog", "cat"):
            assert isinstance(comparison_results[species], SimulationResult)
            assert comparison_results[species].species_name == species

    def test_compare_result_keys_match_input(self, human_sim: RetinalSimulator, base_image: np.ndarray):
        species_list = ["cat", "human"]
        results = human_sim.compare_species(base_image, species_list, seed=SEED)
        assert list(results.keys()) == species_list

    def test_compare_same_image_different_mosaics(self, comparison_results: dict[str, SimulationResult]):
        human_count = comparison_results["human"].mosaic.n_receptors
        dog_count = comparison_results["dog"].mosaic.n_receptors
        assert human_count != dog_count

    def test_compare_single_species(self, dog_sim: RetinalSimulator, base_image: np.ndarray):
        results = dog_sim.compare_species(base_image, ["dog"], seed=SEED)
        assert list(results.keys()) == ["dog"]
        assert results["dog"].species_name == "dog"

    def test_compare_preserves_seed(self, human_sim: RetinalSimulator, base_image: np.ndarray):
        results_a = human_sim.compare_species(base_image, ["human", "dog"], seed=9)
        results_b = human_sim.compare_species(base_image, ["human", "dog"], seed=9)
        for species in ("human", "dog"):
            np.testing.assert_array_equal(
                results_a[species].mosaic.positions,
                results_b[species].mosaic.positions,
            )
            np.testing.assert_allclose(
                results_a[species].activation.responses,
                results_b[species].activation.responses,
            )

    def test_compare_preserves_stimulus_scale(self, base_image: np.ndarray):
        sim_low = RetinalSimulator(
            "human",
            patch_extent_deg=PATCH_EXTENT_DEG,
            stimulus_scale=0.001,
            seed=SEED,
        )
        sim_high = RetinalSimulator(
            "human",
            patch_extent_deg=PATCH_EXTENT_DEG,
            stimulus_scale=0.1,
            seed=SEED,
        )
        low_results = sim_low.compare_species(base_image, ["human", "dog"], seed=SEED)
        high_results = sim_high.compare_species(base_image, ["human", "dog"], seed=SEED)
        assert high_results["human"].activation.responses.mean() > low_results["human"].activation.responses.mean()
        assert high_results["dog"].activation.responses.mean() > low_results["dog"].activation.responses.mean()

    def test_compare_values_are_simulation_results(self, comparison_results: dict[str, SimulationResult]):
        assert all(isinstance(result, SimulationResult) for result in comparison_results.values())


class TestEndToEndColorDeficit:
    def test_red_green_collapse_dog(self, red_green_results: dict[str, SimulationResult]):
        human_diff = _cone_vector_difference(red_green_results["human"], VALIDATION_IMAGE_SIZE)
        dog_diff = _cone_vector_difference(red_green_results["dog"], VALIDATION_IMAGE_SIZE)
        assert human_diff > dog_diff * 1.5

    def test_blue_yellow_preserved_both(self, blue_yellow_results: dict[str, SimulationResult]):
        human_diff = _cone_vector_difference(blue_yellow_results["human"], VALIDATION_IMAGE_SIZE)
        dog_diff = _cone_vector_difference(blue_yellow_results["dog"], VALIDATION_IMAGE_SIZE)
        assert human_diff > 0.1
        assert dog_diff > 0.1

    def test_human_trichromat_better_than_dog_dichromat(
        self,
        confusion_pair_pattern: tuple[np.ndarray, np.ndarray],
        confusion_pair_results: dict[str, SimulationResult],
    ):
        _, figure_mask = confusion_pair_pattern
        human_d = _best_figure_background_discriminability(
            confusion_pair_results["human"],
            figure_mask,
        )
        dog_d = _best_figure_background_discriminability(
            confusion_pair_results["dog"],
            figure_mask,
        )
        assert human_d > dog_d

    def test_cat_retains_luminance_contrast(self, bright_dark_results: dict[str, SimulationResult]):
        cat_result = bright_dark_results["cat"]
        left_mean = _mean_response_in_half(cat_result, "rod", True, VALIDATION_IMAGE_SIZE)
        right_mean = _mean_response_in_half(cat_result, "rod", False, VALIDATION_IMAGE_SIZE)
        assert abs(left_mean - right_mean) > 0.5

    def test_control_pairs_remain_discriminable(self, human_sim: RetinalSimulator):
        for item in control_pairs()[:3]:
            img = _split_image(item["rgb_a"], item["rgb_b"])
            results = human_sim.compare_species(img, ["human", "dog", "cat"], seed=SEED)
            for species in ("human", "dog", "cat"):
                diff = split_half_discriminability(results[species], (VALIDATION_IMAGE_SIZE, VALIDATION_IMAGE_SIZE))
                assert diff > 0.05


class TestComparisonRendering:
    def test_render_comparison_from_pipeline(self, comparison_results: dict[str, SimulationResult]):
        fig = render_comparison(
            {species: result.activation for species, result in comparison_results.items()},
            output_size=(64, 64),
        )
        assert hasattr(fig, "axes")
        assert len(fig.axes) >= 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_voronoi_from_pipeline(self, human_result: SimulationResult):
        image = render_voronoi(human_result.activation, output_size=(64, 64))
        assert image.shape == (64, 64, 3)
        assert image.dtype == np.float32
        assert image.max() > 0.0

    def test_render_reconstructed_from_pipeline(self, human_result: SimulationResult):
        image = render_reconstructed(human_result.activation, grid_shape=(64, 64))
        assert image.shape == (64, 64)
        assert image.dtype == np.float32
        assert image.max() > 0.0

    def test_render_mosaic_map_from_pipeline(self, human_result: SimulationResult):
        fig = render_mosaic_map(human_result.mosaic, output_size=(200, 200))
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 1
        import matplotlib.pyplot as plt
        plt.close(fig)
