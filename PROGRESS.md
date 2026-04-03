# Retinal Sim — Implementation Progress

_Last updated: 2026-04-03 (Phase 13 complete)_

---

## How to use this file

- Check this file at the start of every session to orient on current status.
- Update the status column and add notes after completing work.
- Run `pytest` for all phases.

---

## Phase Status Summary

| Phase | Component                          | Status       | Tests                        | Notes |
|-------|------------------------------------|--------------|------------------------------|-------|
| 1     | Govardovskii nomogram              | **COMPLETE** | 51/51 pass (`test_retina.py`)| A1/A2 nomogram, `build_sensitivity_curves()`, all λ_max for human/dog/cat |
| 2     | Species config loader (YAML)       | **COMPLETE** | 58/58 pass (`test_species.py`)| `species/config.py::SpeciesConfig.load()`, three species YAMLs, density function closures |
| 3     | Scene geometry module              | **COMPLETE** | 24/24 pass (`test_scene.py`) | Angular subtense, retinal scaling, accommodation, patch clipping |
| 4     | Mosaic generator (jittered grid)   | **COMPLETE** | 32/32 pass (`test_mosaic.py`)| Bernoulli jittered grid; all three species; rod-free zone; Nyquist validated |
| 5     | Simplified optical PSF (Gaussian)  | **COMPLETE** | 28/28 pass (`test_optical.py`)| `PSFGenerator.gaussian_psf`, `OpticalStage.apply`; §11b energy conserved |
| 6     | Smits spectral upsampler           | **COMPLETE** | 32/32 pass (`test_spectral.py`) | D65-optimised basis via `lsq_linear`; roundtrip RMSE < 0.0002 |
| 7     | Spectral integration + Naka-Rushton| **COMPLETE** | 34/34 pass (`test_retinal_stage.py`) | `RetinalStage`: bilinear interp, spectral dot product, Naka-Rushton per type |
| 8     | Voronoi visualization              | **COMPLETE** | 28/28 pass (`test_output.py`)| `render_voronoi`, `render_reconstructed`, `render_comparison`, `render_mosaic_map` |
| 9     | Snellen acuity validation          | **COMPLETE** | 42/42 pass (`test_snellen.py`)| Snellen E generator; full pipeline discriminability; species ordering validated |
| 10    | Dichromat confusion validation     | **COMPLETE** | 36/36 pass (`test_dichromat.py`) | Ishihara-style dot pattern; `find_confusion_pair` (dog confusion axis); D_human > D_dog for confusion pair |
| 11    | Distance-dependent resolution test | **COMPLETE** | 75/75 pass (`test_distance.py`) | Geometry scaling, receptor count 1/d², cross-species, coverage, seed stability, edge distances |
| 12    | Species comparison pipeline        | **COMPLETE** | 42/42 pass (`test_pipeline.py`) | RetinalSimulator orchestrator, compare_species, end-to-end color deficit validation |
| 13    | Full validation report generator   | **COMPLETE** | 41/41 pass (`test_validation_report.py`) | ValidationSuite with 14 tests, figures for every test, self-contained HTML report |

---

## Phase 13 — Complete

`retinal_sim/validation/report.py` — `ValidationResult`, `ValidationReport`, `ValidationSuite` classes.
14 validation tests across 5 stages (scene, spectral, optical, retinal, e2e), each producing
a figure. `save_html()` generates a self-contained HTML report with embedded base64 figures,
summary table, pass/fail badges, and bonus reference figures (nomogram, mosaics).

---

_Detailed per-phase validation notes for Phases 1–9 are in git history. Key gotchas are in SCRATCHPAD.md._
