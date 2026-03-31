# Retinal Sim — Implementation Progress

_Last updated: 2026-03-30 (Phase 5 complete)_

---

## How to use this file

- Check this file at the start of every session to orient on current status.
- Update the status column and add notes after completing work.
- Run `pytest tests/test_retina.py -v` to re-verify Phase 1; run `pytest` for all phases.

---

## Phase Status Summary

| Phase | Component                          | Status       | Tests                        | Notes |
|-------|------------------------------------|--------------|------------------------------|-------|
| 1     | Govardovskii nomogram              | **COMPLETE** | 51/51 pass (`test_retina.py`)| A1/A2 nomogram, `build_sensitivity_curves()`, all λ_max for human/dog/cat |
| 2     | Species config loader (YAML)       | **COMPLETE** | 58/58 pass (`test_species.py`)| `species/config.py::SpeciesConfig.load()`, three species YAMLs, density function closures |
| 3     | Scene geometry module              | **COMPLETE** | 24/24 pass (`test_scene.py`) | Angular subtense, retinal scaling, accommodation, patch clipping |
| 4     | Mosaic generator (jittered grid)   | **COMPLETE** | 32/32 pass (`test_mosaic.py`)| Bernoulli jittered grid; all three species; rod-free zone; Nyquist validated |
| 5     | Simplified optical PSF (Gaussian)  | **COMPLETE** | 28/28 pass (`test_optical.py`)| `PSFGenerator.gaussian_psf`, `OpticalStage.apply`; §11b energy conserved |
| 6     | Smits spectral upsampler           | stub         | —                            | `spectral/upsampler.py` — raises `NotImplementedError` |
| 7     | Spectral integration + Naka-Rushton| partial      | —                            | `naka_rushton()` implemented; spectral integration stub |
| 8     | Voronoi visualization              | stub         | —                            | `output/` — raises `NotImplementedError` |
| 9     | Snellen acuity validation          | not started  | —                            | Requires phases 2–7 |
| 10    | Dichromat confusion validation     | not started  | —                            | Requires phases 2–7 |
| 11    | Distance-dependent resolution test | not started  | —                            | Requires phases 2–7 |
| 12    | Species comparison pipeline        | not started  | —                            | Requires phases 2–11 |
| 13    | Full validation report generator   | not started  | —                            | HTML report with all figures |

---

## Phase 1 — Govardovskii Nomogram ✓ COMPLETE

**Validated:** 2026-03-30
**Test command:** `pytest tests/test_retina.py -v`
**Result:** 51 passed in 0.08s

Covered by tests:
- `TestPeakWavelength` — A1/A2 peak location for all species/receptor types
- `TestBetaBand` — beta band increases short-wavelength absorption, does not shift main peak
- `TestShape` — long-wavelength flank decay, relative peak ordering (human), half-width plausible
- `TestBuildSensitivityCurves` — returns dict per species, all curves normalized, correct receptor types, output shape, unknown species raises, human trichromat, dog/cat dichromat
- `TestCrossSpecies` — dog L-cone peak in range, cat S-cone red-shifted vs human, rod peaks clustered

Visual validation: `python examples/plot_nomogram.py`

---

## Phase 2 — Species Config Loader ✓ COMPLETE

**Validated:** 2026-03-30
**Test command:** `pytest tests/test_species.py -v`
**Result:** 58 passed in 0.20s

Covered by tests:
- `TestLoading` — all three species load; unknown species raises ValueError
- `TestOpticalParams` — focal length, axial length, pupil shape match architecture doc; positive values
- `TestRetinalStructure` — cone types match LAMBDA_MAX; peak wavelengths match; ratios sum to 1.0; naka-rushton keys present for all receptor types
- `TestDensityFunctions` — callables return dicts/floats; cone density decreases with eccentricity; human rod-free zone at fovea; dog/cat have rods at center

**Files added:**
- `retinal_sim/data/species/human.yaml`
- `retinal_sim/data/species/dog.yaml`
- `retinal_sim/data/species/cat.yaml`
- `retinal_sim/species/config.py` (implemented)

---

## Phase 3 — Scene Geometry ✓ COMPLETE

**Validated:** 2026-03-30
**Test command:** `pytest tests/test_scene.py -v`
**Result:** 24 passed in 0.07s

Covered by tests:
- `TestAngularSubtense` — exact arctan formula, collimated = 0°, small-angle limit, aspect ratio inference
- `TestRetinalScaling` — human > cat > dog ordering, exact focal-length ratio, mm_per_pixel consistency
- `TestAccommodation` — collimated = 0D, 1m = 1D, human within range, dog beyond range, non-negative
- `TestPatchClipping` — small scene not clipped, large scene clipped, fraction ∈ [0,1]
- `TestValidation` — §11e criteria: 1m@57.3m → 1° (< 0.01%), retinal ordering, area ∝ 1/d²
- `TestInputValidation` — negative width raises, zero distance raises

**Key implementation notes:**
- Accommodation hard-cutoff in `_MAX_ACCOMMODATION` dict (human=10D, dog=2.5D, cat=3.0D)
- Species name attached to OpticalParams as `op._species_name` (no YAML change needed)
- Collimated (inf distance): angular and retinal extents are exactly 0

---

## Phase 4 — Mosaic Generator ✓ COMPLETE

**Validated:** 2026-03-30
**Test command:** `pytest tests/test_mosaic.py -v`
**Result:** 32 passed in 8.99s

Covered by tests:
- `TestMosaicBasic` — returns PhotoreceptorMosaic; shape consistency; all 3 species; positions within patch; apertures positive
- `TestReproducibility` — same seed → identical mosaic; different seeds → different mosaic
- `TestDensity` — receptor count within 25% of density-model integral (§11c); cone fraction within 15%; human > dog cone count
- `TestNyquist` — human 30–200 cpd, dog 3–50 cpd; human > dog (§11c)
- `TestSensitivity` — all sensitivities peak-normalised; S/M/L distinct peaks; no negatives
- `TestTypes` — human trichromat; dog/cat dichromat; human rod-free zone; dog rods at center; only known types

**Key implementation notes:**
- Cell size from MAX total density radially scanned across patch (not just center) — prevents Bernoulli prob capping where rods and cones overlap
- `_build_sensitivity_curves()` uses `govardovskii_a1` directly on `RetinalParams.cone_peak_wavelengths` — no species-name lookup needed
- `cone_ratio_fn` is called once at ecc=0 then sampled vectorially for all cones (PoC simplification; ratio is eccentricity-independent)
- np.vectorize on density callables for vectorised grid computation (acceptable < 15s total for all 3 species)

---

## Phase 5 — Gaussian PSF ✓ COMPLETE

**Validated:** 2026-03-30
**Test command:** `pytest tests/test_optical.py -v`
**Result:** 28 passed in 0.44s

Covered by tests:
- `TestPSFGeneratorShape` — (N_λ, K, K) shape, non-negative values, single/custom kernel sizes
- `TestPSFEnergyConservation` — §11b: |sum - 1.0| < 1e-6 per band for human, dog, defocus, single λ, fine pixel scale
- `TestPSFPhysics` — peak at centre, wavelength dependence (longer λ → wider), defocus widens PSF, symmetry
- `TestPSFPixelScale` — coarser pixel scale → fewer pixels per sigma
- `TestOpticalStageInit` — instantiation, compute_psf shape and energy
- `TestOpticalStageApply` — RetinalIrradiance output, shape/wavelength preservation, scene integration, defocus effect, media transmission

**Key implementation notes:**
- sigma = quadrature sum of diffraction component (1.22 * λ * f#) and defocus component (f * δD * D_p / 2000)
- Kernels stored as float64 for exact normalisation (meets 1e-6 criterion)
- `PSFGenerator` takes `pixel_scale_mm_per_px`; `OpticalStage.apply` infers this from `scene.mm_per_pixel` when available
- `convolve(..., mode="reflect")` avoids edge darkening
- `diffraction_psf` remains a stub (post-PoC)

---

## Phase 6 — Smits Spectral Upsampler

**Status:** Stub — `spectral/upsampler.py` raises `NotImplementedError`
**Validation criteria (§11a):**
- RGB roundtrip RMSE < 0.02 on [0,1] scale

---

## Phase 7 — Spectral Integration + Naka-Rushton

**Status:** Partial — `retina/transduction.py::naka_rushton()` implemented; spectral integration is a stub
**Validation criteria (§11c):**
- Nyquist sampling test (combined with phase 4)

---

## Phases 8–13 — Blocked on phases 2–7

See phase status summary above.

---

## Key Architecture References

- Pipeline stages: `README` / `retinal_sim_architecture.md` §0–4
- Implementation order: §12
- Validation framework: §11
- PoC decisions: §10 (2° patch, photopic only, jittered grid, Smits upsampler)
- Species constants: `retina/opsin.py::LAMBDA_MAX`, `optical/stage.py::OpticalParams`
