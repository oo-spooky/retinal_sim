"""Tests for the species config loader — Phase 2.

Validation strategy (architecture §12 Phase 2):
  - All three species load without error.
  - Unknown species raises ValueError.
  - Optical params match the architecture doc constants.
  - Retinal cone types match LAMBDA_MAX keys for each species.
  - Cone peak wavelengths match LAMBDA_MAX values.
  - Cone ratios sum to 1.0.
  - Density functions are callable and return physically plausible values.
  - Naka-Rushton params contain required keys for every receptor type.
"""

import pytest

from retinal_sim.retina.opsin import LAMBDA_MAX
from retinal_sim.species.config import SpeciesConfig

SPECIES = ["human", "dog", "cat"]

# Architecture doc focal lengths (§2, optical stage table)
FOCAL_LENGTHS = {"human": 22.3, "dog": 17.0, "cat": 18.5}
AXIAL_LENGTHS = {"human": 24.0, "dog": 21.0, "cat": 22.5}
PUPIL_SHAPES = {"human": "circular", "dog": "circular", "cat": "slit"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    @pytest.mark.parametrize("species", SPECIES)
    def test_loads_without_error(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.name == species

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            SpeciesConfig.load("elephant")

    @pytest.mark.parametrize("species", SPECIES)
    def test_returns_species_config_instance(self, species):
        assert isinstance(SpeciesConfig.load(species), SpeciesConfig)


# ---------------------------------------------------------------------------
# Optical params
# ---------------------------------------------------------------------------

class TestOpticalParams:
    @pytest.mark.parametrize("species", SPECIES)
    def test_focal_length_matches_architecture(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.optical.focal_length_mm == pytest.approx(FOCAL_LENGTHS[species])

    @pytest.mark.parametrize("species", SPECIES)
    def test_axial_length_matches_architecture(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.optical.axial_length_mm == pytest.approx(AXIAL_LENGTHS[species])

    @pytest.mark.parametrize("species", SPECIES)
    def test_pupil_shape_matches_architecture(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.optical.pupil_shape == PUPIL_SHAPES[species]

    @pytest.mark.parametrize("species", SPECIES)
    def test_pupil_diameter_positive(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.optical.pupil_diameter_mm > 0

    @pytest.mark.parametrize("species", SPECIES)
    def test_lca_diopters_positive(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.optical.lca_diopters > 0

    def test_cat_slit_height_present(self):
        cfg = SpeciesConfig.load("cat")
        assert cfg.optical.pupil_height_mm is not None
        assert cfg.optical.pupil_height_mm > cfg.optical.pupil_diameter_mm

    @pytest.mark.parametrize("species", ["human", "dog"])
    def test_circular_species_have_no_slit_height(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.optical.pupil_height_mm is None


# ---------------------------------------------------------------------------
# Retinal params — structure
# ---------------------------------------------------------------------------

class TestRetinalStructure:
    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_types_match_lambda_max(self, species):
        cfg = SpeciesConfig.load(species)
        expected = set(LAMBDA_MAX[species]) - {"rod"}
        assert set(cfg.retinal.cone_types) == expected

    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_peak_wavelengths_match_lambda_max(self, species):
        cfg = SpeciesConfig.load(species)
        for receptor, lam in LAMBDA_MAX[species].items():
            if receptor == "rod":
                assert cfg.retinal.rod_peak_wavelength == pytest.approx(lam)
            else:
                assert cfg.retinal.cone_peak_wavelengths[receptor] == pytest.approx(lam)

    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_ratio_sums_to_one(self, species):
        cfg = SpeciesConfig.load(species)
        ratios = cfg.retinal.cone_ratio_fn(0.0)
        assert sum(ratios.values()) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_ratio_keys_match_cone_types(self, species):
        cfg = SpeciesConfig.load(species)
        ratios = cfg.retinal.cone_ratio_fn(0.0)
        assert set(ratios.keys()) == set(cfg.retinal.cone_types)

    @pytest.mark.parametrize("species", SPECIES)
    def test_naka_rushton_has_required_keys(self, species):
        cfg = SpeciesConfig.load(species)
        required = {"n", "sigma", "R_max"}
        for rtype, params in cfg.retinal.naka_rushton_params.items():
            assert required <= params.keys(), (
                f"{species} {rtype} missing keys: {required - params.keys()}"
            )

    @pytest.mark.parametrize("species", SPECIES)
    def test_naka_rushton_covers_all_receptor_types(self, species):
        cfg = SpeciesConfig.load(species)
        expected = set(cfg.retinal.cone_types) | {"rod"}
        assert set(cfg.retinal.naka_rushton_params.keys()) == expected


# ---------------------------------------------------------------------------
# Density functions
# ---------------------------------------------------------------------------

class TestDensityFunctions:
    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_density_fn_callable(self, species):
        cfg = SpeciesConfig.load(species)
        result = cfg.retinal.cone_density_fn(0.0)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_density_at_center_is_positive(self, species):
        cfg = SpeciesConfig.load(species)
        densities = cfg.retinal.cone_density_fn(0.0)
        assert all(v > 0 for v in densities.values())

    @pytest.mark.parametrize("species", SPECIES)
    def test_cone_density_decreases_with_eccentricity(self, species):
        cfg = SpeciesConfig.load(species)
        d_center = sum(cfg.retinal.cone_density_fn(0.0).values())
        d_periph = sum(cfg.retinal.cone_density_fn(5.0).values())
        assert d_periph < d_center

    @pytest.mark.parametrize("species", SPECIES)
    def test_rod_density_fn_callable(self, species):
        cfg = SpeciesConfig.load(species)
        result = cfg.retinal.rod_density_fn(1.0)
        assert isinstance(result, float)

    @pytest.mark.parametrize("species", SPECIES)
    def test_rod_density_positive_outside_center(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.retinal.rod_density_fn(1.0) > 0

    def test_human_rod_free_zone_at_fovea(self):
        cfg = SpeciesConfig.load("human")
        # Foveal center is rod-free for human
        assert cfg.retinal.rod_density_fn(0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("species", ["dog", "cat"])
    def test_no_rod_free_zone_for_non_human(self, species):
        cfg = SpeciesConfig.load(species)
        assert cfg.retinal.rod_density_fn(0.0) > 0
