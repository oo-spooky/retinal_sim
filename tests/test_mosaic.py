"""Phase 4 tests: photoreceptor mosaic generator.

Validation criteria (architecture §11c):
- Receptor count within 25% of density-model prediction per unit area
- Cone/rod ratio within 15% of expected
- Nyquist frequency: human ~60 cpd, dog ~12 cpd (generous ranges used here)
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import KDTree

from retinal_sim.retina.mosaic import MosaicGenerator, PhotoreceptorMosaic
from retinal_sim.species.config import SpeciesConfig

# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

WAVELENGTHS = np.arange(380, 721, 5, dtype=np.float32)
SEED = 42


def _make_generator(species_name: str) -> MosaicGenerator:
    cfg = SpeciesConfig.load(species_name)
    return MosaicGenerator(cfg.retinal, cfg.optical, wavelengths=WAVELENGTHS)


def _make_mosaic(species_name: str) -> PhotoreceptorMosaic:
    return _make_generator(species_name).generate(seed=SEED)


@pytest.fixture(scope="module")
def human_mosaic() -> PhotoreceptorMosaic:
    return _make_mosaic("human")


@pytest.fixture(scope="module")
def dog_mosaic() -> PhotoreceptorMosaic:
    return _make_mosaic("dog")


@pytest.fixture(scope="module")
def cat_mosaic() -> PhotoreceptorMosaic:
    return _make_mosaic("cat")


def _expected_counts(species_name: str, n_grid: int = 100):
    """Numerically integrate density model over the 2° patch.

    Returns (expected_cones, expected_rods) via a vectorised meshgrid sum.
    Coarse grid (100×100) is fine enough for a ±25% tolerance test.
    """
    cfg = SpeciesConfig.load(species_name)
    op = cfg.optical
    rp = cfg.retinal

    hw = op.focal_length_mm * np.tan(np.radians(rp.patch_extent_deg / 2.0))
    xs = np.linspace(-hw, hw, n_grid)
    ys = np.linspace(-hw, hw, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    flat_x = XX.ravel()
    flat_y = YY.ravel()
    ecc = np.sqrt(flat_x**2 + flat_y**2)
    angle = np.arctan2(flat_y, flat_x)
    cell_area = (2.0 * hw / n_grid) ** 2

    cone_types = list(rp.cone_peak_wavelengths.keys())

    cone_total = np.zeros(len(ecc))
    for ct in cone_types:
        cone_total += np.vectorize(
            lambda e, a, _ct=ct: rp.cone_density_fn(e, a).get(_ct, 0.0)
        )(ecc, angle)

    rod_total = np.vectorize(lambda e, a: rp.rod_density_fn(e, a))(ecc, angle)

    return float(cone_total.sum() * cell_area), float(rod_total.sum() * cell_area)


def _nyquist_cpd(mosaic: PhotoreceptorMosaic, focal_length_mm: float) -> float:
    """Estimate Nyquist frequency (cycles/deg) from mean nearest-cone spacing."""
    is_cone = mosaic.types != "rod"
    cone_pos = mosaic.positions[is_cone]
    if len(cone_pos) < 2:
        return 0.0
    tree = KDTree(cone_pos)
    dists, _ = tree.query(cone_pos, k=2)  # k=2: self + nearest neighbour
    mean_spacing_mm = float(dists[:, 1].mean())
    mm_per_deg = focal_length_mm * np.tan(np.radians(1.0))
    spacing_deg = mean_spacing_mm / mm_per_deg
    return 1.0 / (2.0 * spacing_deg)


# ---------------------------------------------------------------------------
# TestMosaicBasic
# ---------------------------------------------------------------------------

class TestMosaicBasic:
    def test_returns_photoreceptor_mosaic(self, human_mosaic):
        assert isinstance(human_mosaic, PhotoreceptorMosaic)

    def test_array_shapes_consistent(self, human_mosaic):
        n = human_mosaic.n_receptors
        assert human_mosaic.positions.shape == (n, 2)
        assert human_mosaic.types.shape == (n,)
        assert human_mosaic.apertures.shape == (n,)
        assert human_mosaic.sensitivities.shape == (n, len(WAVELENGTHS))

    def test_n_receptors_matches_positions(self, human_mosaic):
        assert human_mosaic.n_receptors == len(human_mosaic.positions)

    def test_dtypes(self, human_mosaic):
        assert human_mosaic.positions.dtype == np.float32
        assert human_mosaic.apertures.dtype == np.float32
        assert human_mosaic.sensitivities.dtype == np.float32
        assert human_mosaic.types.dtype.kind == "U"

    def test_all_species_generate_without_error(self):
        for sp in ("human", "dog", "cat"):
            m = _make_mosaic(sp)
            assert m.n_receptors > 0, f"{sp}: mosaic is empty"

    def test_positions_within_patch(self, human_mosaic):
        """All receptor positions should lie within the 2° patch bounds."""
        cfg = SpeciesConfig.load("human")
        hw = cfg.optical.focal_length_mm * np.tan(
            np.radians(cfg.retinal.patch_extent_deg / 2.0)
        )
        assert np.all(np.abs(human_mosaic.positions[:, 0]) <= hw * 1.01)
        assert np.all(np.abs(human_mosaic.positions[:, 1]) <= hw * 1.01)

    def test_apertures_positive(self, human_mosaic):
        assert np.all(human_mosaic.apertures > 0)


# ---------------------------------------------------------------------------
# TestReproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_identical_positions(self):
        gen = _make_generator("human")
        m1 = gen.generate(seed=0)
        m2 = gen.generate(seed=0)
        np.testing.assert_array_equal(m1.positions, m2.positions)
        np.testing.assert_array_equal(m1.types, m2.types)

    def test_different_seeds_produce_different_positions(self):
        gen = _make_generator("human")
        m1 = gen.generate(seed=0)
        m2 = gen.generate(seed=1)
        assert not np.array_equal(m1.positions, m2.positions)


# ---------------------------------------------------------------------------
# TestDensity  (architecture §11c validation)
# ---------------------------------------------------------------------------

class TestDensity:
    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_receptor_count_within_25_percent(self, species):
        """Total receptor count within 25% of density-model integral."""
        mosaic = _make_mosaic(species)
        exp_cones, exp_rods = _expected_counts(species)
        expected_total = exp_cones + exp_rods
        actual = mosaic.n_receptors
        rel_err = abs(actual - expected_total) / expected_total
        assert rel_err < 0.25, (
            f"{species}: actual {actual}, expected {expected_total:.0f}, "
            f"relative error {rel_err:.2%}"
        )

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_cone_fraction_within_15_percent(self, species):
        """Cone fraction within 15 percentage points of expected."""
        mosaic = _make_mosaic(species)
        exp_cones, exp_rods = _expected_counts(species)
        expected_cone_frac = exp_cones / max(exp_cones + exp_rods, 1.0)

        n_cones = int((mosaic.types != "rod").sum())
        actual_cone_frac = n_cones / max(mosaic.n_receptors, 1)

        err = abs(actual_cone_frac - expected_cone_frac)
        assert err < 0.15, (
            f"{species}: actual cone frac {actual_cone_frac:.3f}, "
            f"expected {expected_cone_frac:.3f}, diff {err:.3f}"
        )

    def test_human_more_cones_than_dog_in_central_patch(
        self, human_mosaic, dog_mosaic
    ):
        """Human foveal cone density > dog at same patch extent."""
        h_cones = int((human_mosaic.types != "rod").sum())
        d_cones = int((dog_mosaic.types != "rod").sum())
        assert h_cones > d_cones


# ---------------------------------------------------------------------------
# TestNyquist  (architecture §11c)
# ---------------------------------------------------------------------------

class TestNyquist:
    def test_human_nyquist_in_range(self, human_mosaic):
        """Human cone Nyquist: expect roughly 40–150 cpd at fovea."""
        cfg = SpeciesConfig.load("human")
        nyq = _nyquist_cpd(human_mosaic, cfg.optical.focal_length_mm)
        assert 30.0 < nyq < 200.0, f"Human Nyquist {nyq:.1f} cpd outside expected range"

    def test_dog_nyquist_in_range(self, dog_mosaic):
        """Dog cone Nyquist: expect roughly 5–30 cpd."""
        cfg = SpeciesConfig.load("dog")
        nyq = _nyquist_cpd(dog_mosaic, cfg.optical.focal_length_mm)
        assert 3.0 < nyq < 50.0, f"Dog Nyquist {nyq:.1f} cpd outside expected range"

    def test_human_nyquist_greater_than_dog(self, human_mosaic, dog_mosaic):
        """Human cone sampling finer than dog (higher Nyquist)."""
        h_nyq = _nyquist_cpd(human_mosaic, SpeciesConfig.load("human").optical.focal_length_mm)
        d_nyq = _nyquist_cpd(dog_mosaic, SpeciesConfig.load("dog").optical.focal_length_mm)
        assert h_nyq > d_nyq


# ---------------------------------------------------------------------------
# TestSensitivity
# ---------------------------------------------------------------------------

class TestSensitivity:
    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_all_sensitivities_peak_at_unity(self, species):
        """Every receptor's sensitivity curve is peak-normalised to 1.0."""
        mosaic = _make_mosaic(species)
        if mosaic.n_receptors == 0:
            pytest.skip("empty mosaic")
        peaks = mosaic.sensitivities.max(axis=1)
        assert np.allclose(peaks, 1.0, atol=1e-5), \
            f"Some sensitivities not normalised: min peak = {peaks.min():.6f}"

    def test_human_cone_types_have_distinct_peak_wavelengths(self, human_mosaic):
        """S, M, L cones differ in peak wavelength."""
        for ctype in ("S_cone", "M_cone", "L_cone"):
            mask = human_mosaic.types == ctype
            assert mask.any(), f"No {ctype} found"
            peak_wl = WAVELENGTHS[human_mosaic.sensitivities[mask][0].argmax()]
            if ctype == "S_cone":
                assert peak_wl < 450
            elif ctype == "M_cone":
                assert 510 < peak_wl < 560
            elif ctype == "L_cone":
                assert peak_wl > 540

    @pytest.mark.parametrize("species", ["human", "dog", "cat"])
    def test_no_negative_sensitivities(self, species):
        mosaic = _make_mosaic(species)
        assert np.all(mosaic.sensitivities >= 0)


# ---------------------------------------------------------------------------
# TestTypes
# ---------------------------------------------------------------------------

class TestTypes:
    def test_human_has_three_cone_types(self, human_mosaic):
        cone_types = set(human_mosaic.types[human_mosaic.types != "rod"])
        assert {"S_cone", "M_cone", "L_cone"} == cone_types

    def test_dog_is_dichromat(self, dog_mosaic):
        """Dog has only S_cone and L_cone (no M_cone)."""
        cone_types = set(dog_mosaic.types[dog_mosaic.types != "rod"])
        assert cone_types == {"S_cone", "L_cone"}

    def test_cat_is_dichromat(self, cat_mosaic):
        cone_types = set(cat_mosaic.types[cat_mosaic.types != "rod"])
        assert cone_types == {"S_cone", "L_cone"}

    def test_human_rod_free_zone(self, human_mosaic):
        """No rods within the rod-free foveal pit (radius ≈ 0.175 mm)."""
        rod_mask = human_mosaic.types == "rod"
        rod_pos = human_mosaic.positions[rod_mask]
        if rod_pos.shape[0] == 0:
            return  # no rods at all — also valid
        rod_ecc = np.sqrt(rod_pos[:, 0] ** 2 + rod_pos[:, 1] ** 2)
        assert np.all(rod_ecc >= 0.15), \
            f"Rod found inside rod-free zone: min ecc = {rod_ecc.min():.4f} mm"

    def test_dog_has_rods_near_center(self, dog_mosaic):
        """Dog has no rod-free zone — rods present close to area centralis."""
        rod_mask = dog_mosaic.types == "rod"
        assert rod_mask.any(), "Dog mosaic contains no rods"
        rod_ecc = np.sqrt(
            dog_mosaic.positions[rod_mask, 0] ** 2
            + dog_mosaic.positions[rod_mask, 1] ** 2
        )
        # At least some rods within 0.1 mm of center
        assert np.any(rod_ecc < 0.1), \
            f"No rods near dog area centralis; min ecc = {rod_ecc.min():.4f} mm"

    def test_only_known_types_present(self, human_mosaic):
        known = {"S_cone", "M_cone", "L_cone", "rod"}
        actual = set(human_mosaic.types)
        assert actual <= known, f"Unknown types: {actual - known}"

    def test_visual_streak_hook_disabled_by_default_for_dog_and_cat(self):
        dog_cfg = SpeciesConfig.load("dog")
        cat_cfg = SpeciesConfig.load("cat")
        assert dog_cfg.retinal.visual_streak.supported is True
        assert cat_cfg.retinal.visual_streak.supported is True
        assert dog_cfg.retinal.visual_streak.enabled is False
        assert cat_cfg.retinal.visual_streak.enabled is False
        assert dog_cfg.retinal.visual_streak.status == "deferred"
        assert cat_cfg.retinal.visual_streak.status == "deferred"
