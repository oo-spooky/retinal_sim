# AGENTS.md

Project instructions for Codex CLI when working in this repository.

## Start Here

- Read `docs/llm_coordination.md` for the shared Codex/Claude workflow, touchlog rules, doc roles, and archive policy.
- Read `TOUCHLOG.md` before starting work. Append one end-of-session entry in local time before leaving.
- Check `PROGRESS.md` for the current repo snapshot, `CODEREVIEW.md` for open findings, and `SCRATCHPAD.md` for unresolved short-lived notes.
- For structural, scientific-claim, or reporting work, also read `retinal_sim_architecture.md`, `docs/reporting_transparency.md`, `reports/architecture_audit_2026-04-04.md`, and `reports/remediation_roadmap_2026-04-04.md`.
- Treat any newer dated `reports/*.md` artifacts the same way: required reference material, not optional history.

## Project Overview

`retinal_sim` is a physically-parameterized, species-configurable simulation of image formation on the mammalian retina. It models how humans, dogs, and cats see the same scene differently based on their optical and retinal physiology.

## Architecture

Linear pipeline of five stages, each consuming the previous stage's output dataclass:

```text
RGB image
  -> SpectralImage (H, W, N_lambda)           spectral/upsampler.py
  -> RetinalIrradiance (H, W, N_lambda)       optical/stage.py
  -> PhotoreceptorMosaic + MosaicActivation   retina/stage.py
  -> rendered figures                         output/
```

Orchestrator: `pipeline.py::RetinalSimulator`
Species config: `species/config.py::SpeciesConfig` loads from `data/species/{human,dog,cat}.yaml`

### Source Layout

```text
retinal_sim/
  constants.py          # shared physical constants
  pipeline.py           # RetinalSimulator orchestrator
  data/species/         # human.yaml, dog.yaml, cat.yaml
  optical/
    stage.py            # OpticalStage, OpticalParams
    psf.py              # PSFGenerator (gaussian_psf, diffraction_psf)
  output/               # Voronoi rendering, comparison figures
  retina/
    opsin.py            # Govardovskii nomogram, LAMBDA_MAX
    mosaic.py           # MosaicGenerator (jittered grid)
    stage.py            # RetinalStage (spectral integration + Naka-Rushton)
  scene/                # SceneGeometry, SceneDescription
  species/
    config.py           # SpeciesConfig loader
  spectral/
    upsampler.py        # Smits spectral upsampling
  validation/
    acuity.py           # Snellen E acuity validation
    dichromat.py        # Dichromat confusion validation
```

### Test Layout

```text
tests/
  test_retina.py             # Phase 1: nomogram
  test_species.py            # Phase 2: species config
  test_scene.py              # Phase 3: scene geometry
  test_mosaic.py             # Phase 4: mosaic generator
  test_optical.py            # Phase 5: optical PSF
  test_spectral.py           # Phase 6: spectral upsampling
  test_retinal_stage.py      # Phase 7: retinal stage
  test_output.py             # Phase 8: output rendering
  test_snellen.py            # Phase 9: Snellen acuity
  test_dichromat.py          # Phase 10: dichromat validation
  test_distance.py           # Phase 11: distance-dependent resolution
  test_pipeline.py           # integration tests
  test_validation_report.py  # Phase 13: validation report
```

## Commands

```bash
pip install -e ".[dev]"          # install (editable + test deps)
pytest -m "not slow"             # fast local loop
pytest tests/test_<phase>.py -v  # phase-specific loop
pytest                           # full gate
```

Setup gotcha if editable install fails with `BackendUnavailable`:

```bash
pip install setuptools wheel
pip install --no-build-isolation -e ".[dev]"
```

## Repo State

- Phases 1-13 are complete.
- Current focus is code review, architecture audit follow-through, reporting transparency, and doc hygiene.
- `PROGRESS.md` is a live snapshot, not a historical log.
- `CODEREVIEW.md` is open-items only. Resolved review history lives in `docs/archive/CODEREVIEW_RESOLVED_2026-04.md`.
- `reports/status_latest.html` is the audit dashboard for current test status, doc drift, and review state.
- `reports/validation_report.html` is the detailed validation artifact; generated report content should expose assumptions, methods, thresholds, limitations, and code provenance.

## Critical Gotchas

These are the quickest ways to break tests or invalidate behavior:

1. PSF kernels must stay `float64`; `float32` exceeds the energy tolerance `|sum - 1.0| < 1e-6`.
2. Diffraction sigma is `0.42 * lambda * f_number`, not `1.22 * lambda * f_number`.
3. `RetinalIrradiance.metadata` uses `pixel_scale_mm`, not `mm_per_pixel`.
4. Unit-reflectance inputs saturate Naka-Rushton. Validation code uses `stimulus_scale = 0.01` to keep excitations near sigma.
5. Acuity computes discriminability only inside the letter bounding box, not across the whole patch.
6. Acuity uses Pearson correlation distance: `D = 1 - corr(r1, r2)`.
7. Mosaic cell size uses the maximum total density across the patch, not center density alone.
8. D65 is required for spectral roundtrip; equal-energy integration breaks saturated colors.
9. Accommodation is a hard cutoff: `defocus = max(0, demand - species_max)`.
10. `output/perceptual.py::cone_maps_to_srgb` must keep the current Naka-Rushton inverse -> shared normalization -> HPE LMS -> XYZ -> von Kries -> sRGB chain. Do not replace it with a near-identity LMS -> RGB mapping.
11. `scripts/render_scene.py` must crop the input to the simulated patch before displaying results.
12. The proof-of-concept patch size is 2 degrees.

## Code Review

When running `codex review`, check changes against:

- the gotchas listed above,
- architecture compliance in `retinal_sim_architecture.md`,
- test coverage for any new behavior,
- numerical stability, especially kernel normalization and dtype choices,
- and species-correctness across human, dog, and cat.

Write findings in this format:

```text
- **File:** path/to/file.py
- **Function:** function_name
- **Issue:** what's wrong
- **Fix:** what the fix should be
```

Move resolved findings out of `CODEREVIEW.md` into the resolved archive instead of leaving historical material in the active file.

## When Implementing New Code

- Each stage must be independently validatable with its own test file.
- Follow existing patterns; look at a completed phase test file for conventions.
- Species constants go in YAML files, not hardcoded in Python.
- Peak wavelengths live in `retina/opsin.py::LAMBDA_MAX`.
- Optical params (pupil, focal length, axial length) live in `optical/stage.py::OpticalParams`.
- Run `pytest` after changes to verify nothing breaks.
