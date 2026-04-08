# Retinal Sim - Status Snapshot

_Last updated: 2026-04-07 (explorer workflow and milestone visibility)_

## Current State

- Phases 1-13 are complete.
- Current work is code review, architecture audit follow-through, reporting transparency, and documentation discipline.
- Latest notable shipped change: the 2026-04-07 perceptual rendering correction in `retinal_sim/output/perceptual.py` plus simulated-patch cropping in `scripts/render_scene.py`.

## Implementation Phases

| Phase | Component | Status | Tests | Notes |
|-------|-----------|--------|-------|-------|
| 1 | Govardovskii nomogram | **COMPLETE** | 51/51 pass (`test_retina.py`) | A1/A2 nomogram, `build_sensitivity_curves()`, all `lambda_max` values for human/dog/cat |
| 2 | Species config loader (YAML) | **COMPLETE** | 58/58 pass (`test_species.py`) | `species/config.py::SpeciesConfig.load()`, three species YAMLs, density function closures |
| 3 | Scene geometry module | **COMPLETE** | 24/24 pass (`test_scene.py`) | Angular subtense, retinal scaling, accommodation, patch clipping |
| 4 | Mosaic generator (jittered grid) | **COMPLETE** | 32/32 pass (`test_mosaic.py`) | Bernoulli jittered grid; all three species; rod-free zone; Nyquist validated |
| 5 | Simplified optical PSF (Gaussian) | **COMPLETE** | 28/28 pass (`test_optical.py`) | `PSFGenerator.gaussian_psf`, `OpticalStage.apply`; energy conserved |
| 6 | Smits spectral upsampler | **COMPLETE** | 32/32 pass (`test_spectral.py`) | D65-optimized basis via `lsq_linear`; roundtrip RMSE < 0.0002 |
| 7 | Spectral integration + Naka-Rushton | **COMPLETE** | 34/34 pass (`test_retinal_stage.py`) | `RetinalStage`: bilinear interp, spectral dot product, Naka-Rushton per type |
| 8 | Voronoi visualization | **COMPLETE** | 28/28 pass (`test_output.py`) | `render_voronoi`, `render_reconstructed`, `render_comparison`, `render_mosaic_map` |
| 9 | Snellen acuity validation | **COMPLETE** | 42/42 pass (`test_snellen.py`) | Snellen E generator; full pipeline discriminability; species ordering validated |
| 10 | Dichromat confusion validation | **COMPLETE** | 36/36 pass (`test_dichromat.py`) | Ishihara-style dot pattern; dog confusion axis and control separation validated |
| 11 | Distance-dependent resolution test | **COMPLETE** | 75/75 pass (`test_distance.py`) | Geometry scaling, receptor count 1/d^2, cross-species, seed stability, edge distances |
| 12 | Species comparison pipeline | **COMPLETE** | 42/42 pass (`test_pipeline.py`) | `RetinalSimulator` orchestration, `compare_species`, end-to-end color-deficit validation |
| 13 | Full validation report generator | **COMPLETE** | 51/51 pass (`test_validation_report.py`) | ValidationSuite with staged audit output, figures, and transparency metadata |

## Remediation Milestones

| Milestone | Status | Current evidence |
|-----------|--------|------------------|
| M1 | implemented with caveats | Throughput scaling, cat slit anisotropy, wavelength-aware blur/LCA, media transmission diagnostics, and claim-calibrated outputs are implemented; remaining caveats are tracked in `CODEREVIEW.md`, the audit, and report wording. |
| M2 | in progress | Scene-input semantics, retinal-front-end provenance, and validation framing are implemented, but the repo still correctly stops short of claiming a fully grounded end-to-end biological simulator. |
| M3 | future | Publication-grade closure work such as fuller provenance, stronger empirical parameter grounding, and quantified simplification/error bounds remains ahead. |

## Testing And Validation References

- Fast local loop: `pytest -m "not slow"`
- Phase-specific loop: `pytest tests/test_<phase>.py -v`
- Full gate: `pytest`
- `reports/status_latest.html` is the current audit dashboard for automated test status, implementation snapshot, documentation drift, and review state.
- `reports/validation_report.html` and its JSON companion are the detailed validation artifacts.
- `docs/reporting_transparency.md` defines wording and transparency requirements for generated reports.

## Canonical References

- `retinal_sim_architecture.md` is the canonical software architecture.
- `reports/architecture_audit_2026-04-04.md` is the baseline external-methodology audit.
- `reports/remediation_roadmap_2026-04-04.md` explains the R1-R6 remediation decisions already implemented in the repo.
- `CODEREVIEW.md` tracks open review findings.
- `TOUCHLOG.md` and `docs/llm_coordination.md` define the current cross-LLM workflow.

## History

- Detailed phase-by-phase implementation notes are archived in `docs/archive/PROGRESS_legacy_2026-04.md`.
