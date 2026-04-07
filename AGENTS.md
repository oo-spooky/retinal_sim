# AGENTS.md

Project instructions for Codex CLI when working in this repository.

## Project overview

`retinal_sim` is a physically-parameterized, species-configurable simulation of image formation on the mammalian retina. It models how humans, dogs, and cats see the same scene differently based on their optical and retinal physiology.

## Architecture

Linear pipeline of five stages, each consuming the previous stage's output dataclass:

```
RGB image
  -> SpectralImage (H, W, N_lambda)          spectral/upsampler.py
  -> RetinalIrradiance (H, W, N_lambda)      optical/stage.py
  -> PhotoreceptorMosaic + MosaicActivation   retina/stage.py
  -> rendered figures                          output/
```

Orchestrator: `pipeline.py::RetinalSimulator`
Species config: `species/config.py::SpeciesConfig` loads from `data/species/{human,dog,cat}.yaml`

### Source layout

```
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

### Test layout

```
tests/
  test_retina.py        # Phase 1: nomogram (51 tests)
  test_species.py       # Phase 2: species config (58 tests)
  test_scene.py         # Phase 3: scene geometry (24 tests)
  test_mosaic.py        # Phase 4: mosaic generator (32 tests)
  test_optical.py       # Phase 5: optical PSF (28 tests)
  test_spectral.py      # Phase 6: spectral upsampling (32 tests)
  test_retinal_stage.py # Phase 7: retinal stage (34 tests)
  test_output.py        # Phase 8: output rendering (28 tests)
  test_snellen.py       # Phase 9: Snellen acuity (42 tests)
  test_dichromat.py     # Phase 10: dichromat validation (36 tests)
  test_distance.py      # Phase 11: distance-dependent resolution (75 tests)
  test_pipeline.py      # integration tests
```

## Commands

```bash
pip install -e ".[dev]"          # install (editable + test deps)
pytest                           # run full suite / full gate
pytest tests/test_retina.py -v   # single test file
```

Recommended loops:
```bash
pytest -m "not slow"                     # fast local loop
pytest tests/test_<phase>.py -v          # phase-specific loop
pytest                                   # full gate / release confidence
```

**Setup gotcha:** If `pip install -e ".[dev]"` fails with `BackendUnavailable`:
```bash
pip install setuptools wheel && pip install --no-build-isolation -e ".[dev]"
```

## Implementation status

Phases 1-13 are COMPLETE. The current repo focus is full code review, architecture audit, and transparent reporting.

Implementation progress, validation status, and architecture audit status are different things:
- `PROGRESS.md` tracks implementation completion and test counts.
- `CODEREVIEW.md` tracks review findings.
- `reports/status_latest.html` is the audit dashboard for current test status, doc drift, and review state.
- `reports/validation_report.html` is the detailed validation artifact; generated user content should expose assumptions, methods, thresholds, and code provenance.
- `reports/architecture_audit_2026-04-04.md` is the baseline architecture audit and the yardstick for the R-phase remediation effort.
- `reports/remediation_roadmap_2026-04-04.md` is the R1–R6 remediation plan (milestones M1/M2/M3, work packages A–D). R1–R6 are already implemented; treat the roadmap as the authoritative record of why the current optical/spectral/retinal/validation/output choices look the way they do, and check it before proposing structural changes. Any later dated `reports/*.md` artifacts from subsequent sessions should be read the same way.

## Key design decisions and gotchas

These are critical — violating them will break tests or produce wrong results:

1. **PSF kernels must be float64** — float32 accumulates rounding error exceeding the energy criterion |sum - 1.0| < 1e-6.

2. **Diffraction sigma = 0.42 * lambda * f#** — NOT 1.22 * lambda * f# (that's the Airy first-zero radius, not the Gaussian sigma).

3. **pixel_scale_mm** (not mm_per_pixel) is the key name in RetinalIrradiance.metadata.

4. **Naka-Rushton saturation** — With unit-reflectance inputs, excitations >> sigma (0.5), saturating all responses. Validation code uses `stimulus_scale = 0.01` to keep excitations near sigma.

5. **Acuity uses letter-region masking** — D is computed only for cones inside the letter bounding box, not the full patch. Without this, background saturation kills discriminability.

6. **Acuity metric is Pearson correlation distance** — D = 1 - corr(r1, r2). Absolute-difference metrics don't work.

7. **Mosaic cell size uses max total density across the patch**, not just center density. Otherwise Bernoulli probability > 1.0 in parafovea.

8. **D65 illuminant is required** for spectral roundtrip — equal-energy integration fails for saturated colors.

9. **Accommodation model is hard-cutoff** — defocus = max(0, demand - species_max). No lag/lead curves.

10. **Perceptual rendering uses HPE LMS→XYZ + von Kries, not a hand-rolled matrix.** `output/perceptual.py::cone_maps_to_srgb` (a) inverts Naka-Rushton with `x = sigma * (r/(R_max - r))**(1/n)` before any color matrix, (b) renormalizes all cones by a *shared* 99th-percentile scalar (per-cone normalization destroys color), (c) applies the inverse Hunt-Pointer-Estevez D65 matrix composed with `inv(_SRGB_TO_XYZ)` and a von Kries diagonal computed at import time from the actual Govardovskii curves under D65, (d) finishes with the proper sRGB EOTF (not `^1/2.2`). Do not replace this with a near-identity LMS→RGB mapping — it produces yellow output because human L (560 nm) and M (530 nm) overlap massively. Regression tests live in `tests/test_output.py::TestPerceptualHumanColor` (white-in≈neutral, red-in→red-dominant) and run the full pipeline end-to-end.

11. **`scripts/render_scene.py` crops input to the simulated patch.** The pipeline only models the central `--patch-deg` of the scene; rendering cone activations stretched across the full image makes high-acuity species look unnaturally blurry. The script computes the scene's angular subtense from `--scene-width-m` / `--distance-m` and crops the input before running the pipeline.

10. **PoC patch size is 2 degrees**. Scale to 5-10 degrees later.

## Code review findings

Open code review items are tracked in `CODEREVIEW.md`. Check it before making changes. Resolved items are in `CODEREVIEW_RESOLVED.md`.

## When doing code review

When running `codex review`, check changes against:
- The gotchas listed above
- Architecture compliance (see `retinal_sim_architecture.md` for full spec)
- Test coverage for any new functionality
- Numerical stability (float64 for kernels, proper normalization)
- Species-correctness (changes must work for human, dog, AND cat)

Write findings in this format:
```
- **File:** path/to/file.py
- **Function:** function_name
- **Issue:** what's wrong
- **Fix:** what the fix should be
```

## When implementing new code

- Each stage must be independently validatable with its own test file
- Follow existing patterns — look at a completed phase's test file for conventions
- Species constants go in YAML files, not hardcoded
- Peak wavelengths live in `retina/opsin.py::LAMBDA_MAX`
- Optical params (pupil, focal length, axial length) are in `optical/stage.py::OpticalParams`
- Run `pytest` after changes to verify nothing breaks
