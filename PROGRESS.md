# Retinal Sim — Implementation Progress

_Last updated: 2026-03-30 (Phase 2 complete)_

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
| 3     | Scene geometry module              | stub         | —                            | Angular subtense + retinal scaling tests pending |
| 4     | Mosaic generator (jittered grid)   | stub         | —                            | `retina/mosaic.py` — raises `NotImplementedError` |
| 5     | Simplified optical PSF (Gaussian)  | stub         | —                            | `optical/psf.py::gaussian_psf` — raises `NotImplementedError` |
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

## Phase 3 — Scene Geometry

**Status:** Stub
**Validation criteria (§11e):**
- Angular subtense: 1m object at 57.3m → 1° (within 0.01%)
- Retinal scaling across species: proportional to focal lengths (human 22.3mm > cat 18.5mm > dog 17.0mm)
- Accommodation/defocus injection: correct residual per species
- Distance-dependent receptor sampling: count scales as ~1/d²

---

## Phase 4 — Mosaic Generator

**Status:** Stub — `retina/mosaic.py` raises `NotImplementedError`
**Validation criteria (§11c):**
- Receptor count within 25% of published histological counts per 1mm²
- Cone/rod ratio within 15%
- Nyquist frequency: human ~60 cpd, dog ~12 cpd, cat ~8–10 cpd

---

## Phase 5 — Gaussian PSF

**Status:** Stub — `optical/psf.py::gaussian_psf` raises `NotImplementedError`
**Validation criteria (§11b):**
- PSF energy conservation: |sum(PSF) - 1.0| < 1e-6 per wavelength band

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
