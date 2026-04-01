"""Photoreceptor mosaic generator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from retinal_sim.constants import WAVELENGTHS as _CANONICAL_WAVELENGTHS


@dataclass
class Photoreceptor:
    position: Tuple[float, float]   # (x_mm, y_mm)
    type: str                        # 'S_cone' | 'M_cone' | 'L_cone' | 'rod'
    aperture_um: float
    sensitivity: np.ndarray          # S(λ), shape (N_λ,)


@dataclass
class PhotoreceptorMosaic:
    positions: np.ndarray    # (N, 2) float32
    types: np.ndarray        # (N,) dtype 'U10'
    apertures: np.ndarray    # (N,) float32
    sensitivities: np.ndarray  # (N, N_λ) float32
    voronoi: Optional[object] = field(default=None, repr=False)

    @property
    def n_receptors(self) -> int:
        return len(self.positions)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_types_vectorized(
    rng: np.random.Generator, ratios: Dict[str, float], n: int
) -> np.ndarray:
    """Draw *n* receptor type labels according to *ratios* probabilities."""
    ctypes = list(ratios.keys())
    probs = np.array([ratios[t] for t in ctypes], dtype=float)
    probs /= probs.sum()
    indices = rng.choice(len(ctypes), size=n, p=probs)
    return np.array(ctypes, dtype="U10")[indices]


def _cone_aperture_um(ecc_mm: np.ndarray) -> np.ndarray:
    """Inner segment diameter (μm) as a simple linear function of eccentricity.

    Foveal cones: ~2 μm.  Peripheral cones grow to ~10 μm at ~2 mm ecc.
    NOTE: This is a human-centric approximation.  Dog/cat inner segment
    diameters differ; for later phases, move aperture parameters into species
    YAML and parameterise per species.  Rod aperture (_ROD_APERTURE_UM) is
    also a constant — in reality rods grow from ~1 μm foveal to ~3 μm
    peripheral, but rod aperture is not used in PoC spectral integration.
    """
    return np.clip(2.0 + 4.0 * np.asarray(ecc_mm, dtype=float), 2.0, 10.0)


def _build_sensitivity_curves(retinal_params: object, wavelengths: np.ndarray) -> dict:
    """Build normalised spectral sensitivity curves directly from RetinalParams.

    Uses the Govardovskii A1 nomogram for each receptor type.  Avoids
    requiring a species-name look-up by reading λ_max values straight from
    *retinal_params*.

    **Important (CR-15):** The Govardovskii nomogram returns *in-vitro*
    absorption spectra.  Architecture §3b calls for applying lens/media
    pre-filtering (``OpticalParams.media_transmission``) to convert these to
    in-vivo sensitivities.  This conversion is deferred to a later phase; for
    the PoC, ``OpticalStage`` applies media transmission to the irradiance
    before it reaches the retinal stage, which is mathematically equivalent
    *only* when ``OpticalStage`` precedes ``RetinalStage`` in the pipeline.
    Calling ``RetinalStage.compute_response()`` directly on raw hyperspectral
    data (without a preceding ``OpticalStage``) will over-estimate S-cone and
    rod sensitivity at short wavelengths.
    """
    from retinal_sim.retina.opsin import govardovskii_a1

    wl = np.asarray(wavelengths, dtype=float)
    curves: dict = {
        ctype: govardovskii_a1(lam_max, wl).astype(np.float32)
        for ctype, lam_max in retinal_params.cone_peak_wavelengths.items()
    }
    curves["rod"] = govardovskii_a1(
        retinal_params.rod_peak_wavelength, wl
    ).astype(np.float32)
    return curves


class MosaicGenerator:
    """Generates species-appropriate photoreceptor mosaics.

    PoC implementation: jittered grid (10x faster than Poisson disk).

    Each axis-aligned cell in a regular grid receives at most one receptor.
    Cell size is set by the peak receptor density at the area centralis.
    Whether a cell is occupied is determined by Bernoulli sampling with
    probability = local_total_density × cell_area, so the expected density
    matches the species density model everywhere in the patch.

    Upgrade path: full Poisson disk sampling with spatially varying radius.
    """

    _ROD_APERTURE_UM: float = 2.0

    def __init__(
        self,
        retinal_params: object,
        optical_params: object,
        wavelengths: Optional[np.ndarray] = None,
    ) -> None:
        self._rp = retinal_params
        self._op = optical_params

        if wavelengths is None:
            wavelengths = _CANONICAL_WAVELENGTHS.astype(np.float32)
        self._wavelengths = np.asarray(wavelengths, dtype=np.float32)

        # Patch half-extent in mm (square patch)
        half_deg = float(retinal_params.patch_extent_deg) / 2.0
        self._patch_half_mm: float = float(
            optical_params.focal_length_mm * np.tan(np.radians(half_deg))
        )
        self._cx: float = float(retinal_params.patch_center_mm[0])
        self._cy: float = float(retinal_params.patch_center_mm[1])

        # Pre-build spectral sensitivities (shared across all generated mosaics)
        self._sensitivities: dict = _build_sensitivity_curves(
            retinal_params, self._wavelengths
        )

    def generate(self, seed: int = 0) -> PhotoreceptorMosaic:
        """Generate a photoreceptor mosaic via stochastic jittered grid.

        Args:
            seed: Integer random seed for reproducibility.

        Returns:
            PhotoreceptorMosaic populated with positions, types, apertures,
            and spectral sensitivities.
        """
        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # 1. Regular grid — cell size from peak density at area centralis
        # ------------------------------------------------------------------
        peak_density = self._peak_total_density()
        # 1.1 safety margin: ensures cell_area × peak_density < 1.0 everywhere,
        # so Bernoulli acceptance probability is never capped at or above 1.0.
        cell_size = 1.0 / np.sqrt(max(peak_density, 1.0) * 1.1)

        x0 = self._cx - self._patch_half_mm
        x1 = self._cx + self._patch_half_mm
        y0 = self._cy - self._patch_half_mm
        y1 = self._cy + self._patch_half_mm

        xs = np.arange(x0 + cell_size / 2.0, x1, cell_size)
        ys = np.arange(y0 + cell_size / 2.0, y1, cell_size)
        XX, YY = np.meshgrid(xs, ys)
        flat_x = XX.ravel()
        flat_y = YY.ravel()

        # ------------------------------------------------------------------
        # 2. Per-cell densities (vectorised via np.vectorize)
        # ------------------------------------------------------------------
        ecc = np.sqrt((flat_x - self._cx) ** 2 + (flat_y - self._cy) ** 2)
        angle = np.arctan2(flat_y - self._cy, flat_x - self._cx)

        cone_total, rod_total = self._vectorized_densities(ecc, angle)
        total_density = cone_total + rod_total

        # ------------------------------------------------------------------
        # 3. Bernoulli acceptance — keeps expected density = density model
        # ------------------------------------------------------------------
        cell_area = cell_size ** 2
        prob = np.minimum(1.0, total_density * cell_area)
        keep = rng.random(len(flat_x)) < prob
        if not keep.any():
            return self._empty_mosaic()

        n_kept = int(keep.sum())

        # Jitter within cell
        kx = flat_x[keep] + rng.uniform(-0.5, 0.5, n_kept) * cell_size
        ky = flat_y[keep] + rng.uniform(-0.5, 0.5, n_kept) * cell_size
        k_ecc = np.sqrt((kx - self._cx) ** 2 + (ky - self._cy) ** 2)

        k_cone = cone_total[keep]
        k_rod = rod_total[keep]
        k_total = k_cone + k_rod

        # ------------------------------------------------------------------
        # 4. Stochastic type assignment
        # ------------------------------------------------------------------
        is_cone = rng.random(n_kept) < (k_cone / np.maximum(k_total, 1e-30))

        types_arr = np.empty(n_kept, dtype="U10")
        apertures_arr = np.zeros(n_kept, dtype=np.float32)
        n_wl = len(self._wavelengths)
        sens_arr = np.zeros((n_kept, n_wl), dtype=np.float32)

        # Rods (batch assignment)
        rod_mask = ~is_cone
        if rod_mask.any():
            types_arr[rod_mask] = "rod"
            apertures_arr[rod_mask] = self._ROD_APERTURE_UM
            sens_arr[rod_mask] = self._sensitivities["rod"]

        # Cones: draw specific type from constant ratio distribution, then
        # batch-assign per type to avoid per-receptor Python loops
        cone_mask = is_cone
        if cone_mask.any():
            n_cones = int(cone_mask.sum())
            ratios = self._rp.cone_ratio_fn(0.0)
            cone_type_labels = _sample_types_vectorized(rng, ratios, n_cones)

            apertures_arr[cone_mask] = _cone_aperture_um(
                k_ecc[cone_mask]
            ).astype(np.float32)

            cone_global_idx = np.where(cone_mask)[0]
            for ctype in self._rp.cone_peak_wavelengths:
                ct_sel = cone_type_labels == ctype
                if ct_sel.any():
                    gidx = cone_global_idx[ct_sel]
                    types_arr[gidx] = ctype
                    sens_arr[gidx] = self._sensitivities[ctype]

        return PhotoreceptorMosaic(
            positions=np.stack([kx, ky], axis=1).astype(np.float32),
            types=types_arr,
            apertures=apertures_arr,
            sensitivities=sens_arr,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _peak_total_density(self) -> float:
        """Maximum total density (cone + rod) anywhere in the patch.

        Scans a radial profile from centre to patch edge so that the cell
        size accommodates the densest location, preventing the Bernoulli
        acceptance probability from being capped (and therefore losing
        receptors) anywhere in the patch.
        """
        eccs = np.linspace(1e-9, self._patch_half_mm, 40)
        max_density = 0.0
        for ecc in eccs:
            cone_d = float(sum(self._rp.cone_density_fn(float(ecc)).values()))
            rod_d = float(self._rp.rod_density_fn(float(ecc)))
            max_density = max(max_density, cone_d + rod_d)
        return max_density

    def _vectorized_densities(
        self, ecc: np.ndarray, angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-cell (cone_total, rod_total) density arrays.

        Uses np.vectorize so that density callables from SpeciesConfig (which
        operate on scalars) are applied element-wise without explicit Python
        loops over individual cells.
        """
        cone_types: List[str] = list(self._rp.cone_peak_wavelengths.keys())

        cone_total = np.zeros_like(ecc)
        for ct in cone_types:
            def _cone_fn(e: float, a: float, _ct: str = ct) -> float:
                return self._rp.cone_density_fn(e, a).get(_ct, 0.0)

            cone_total += np.vectorize(_cone_fn)(ecc, angle)

        rod_total = np.vectorize(
            lambda e, a: self._rp.rod_density_fn(e, a)
        )(ecc, angle)

        return cone_total, rod_total

    def _empty_mosaic(self) -> PhotoreceptorMosaic:
        """Return a zero-receptor mosaic (edge case guard)."""
        return PhotoreceptorMosaic(
            positions=np.zeros((0, 2), dtype=np.float32),
            types=np.array([], dtype="U10"),
            apertures=np.zeros(0, dtype=np.float32),
            sensitivities=np.zeros(
                (0, len(self._wavelengths)), dtype=np.float32
            ),
        )
