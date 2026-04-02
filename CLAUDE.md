# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session workflow

Check `PROGRESS.md` for current implementation status before starting work.
Read `SCRATCHPAD.md` for non-obvious gotchas and architectural decisions before touching any phase.
Check `CODEREVIEW.md` for open review findings and address any open items before starting new phase work.
Update `PROGRESS.md` after completing any phase.
Update `SCRATCHPAD.md` whenever you discover something that would waste time if rediscovered.
**Before closing a session, run all tests and generate a status report at reports/status_latest.html. Include: phase completion status, test results summary, any validation figures generated this session, and open CODEREVIEW.md items. Keep it simple — one page, no frameworks.**
```bash
python scripts/status_report.py
```
**Before confirming a session is ready to close: commit all changes, then push to the remote.**
```bash
git push
```

## Code review sessions (Opus)

When running as Opus in a review session: audit the codebase against `retinal_sim_architecture.md`, run all tests, and write findings to `CODEREVIEW.md` using this format per item:

- **File:** path/to/file.py
- **Function:** function_name
- **Issue:** what's wrong
- **Fix:** what the fix should be

Move resolved items to the Resolved section rather than deleting them.

## Commands

```bash
# Install (editable, with test deps)
pip install -e ".[dev]"
# If that fails with "BackendUnavailable: Cannot import setuptools.build_meta":
#   pip install setuptools wheel && pip install --no-build-isolation -e ".[dev]"

# Run all tests
pytest

# Run Phase 1 tests only
pytest tests/test_retina.py -v

# Run a single test
pytest tests/test_retina.py::TestPeakWavelength::test_a1_peak_location -v

# Visual validation plot (requires matplotlib)
python examples/plot_nomogram.py
```

## Architecture

The pipeline is a linear chain of five stages, each consuming the previous stage's output dataclass:

```
RGB image
  → SpectralImage (H, W, N_λ)          spectral/upsampler.py
  → RetinalIrradiance (H, W, N_λ)      optical/stage.py
  → PhotoreceptorMosaic + MosaicActivation  retina/stage.py
  → rendered figures                    output/
```

`pipeline.py::RetinalSimulator` orchestrates these stages. `species/config.py::SpeciesConfig` loads all species-specific parameters (optical + retinal) from `data/species/{human,dog,cat}.yaml` and is the single place to change per-species constants.

**What's implemented vs stubbed:**
- `retina/opsin.py` — **fully implemented**: Govardovskii et al. (2000) A1/A2 nomogram, `build_sensitivity_curves()`, all λ_max values for human/dog/cat
- `retina/transduction.py` — **implemented**: `naka_rushton()` function
- `species/config.py` — **fully implemented**: `SpeciesConfig.load()` for human/dog/cat; YAML files at `data/species/`
- `scene/geometry.py` — **fully implemented**: `SceneGeometry.compute()`, angular subtense, retinal scaling, accommodation/defocus, patch clipping
- `retina/mosaic.py` — **fully implemented**: `MosaicGenerator`, Bernoulli jittered-grid, all three species, rod-free zone, Nyquist validated
- Everything else — stubs raising `NotImplementedError`, signatures match the architecture doc

**Implementation order** (from architecture §12): nomogram ✓ → species YAML loader ✓ → scene geometry ✓ → mosaic generator ✓ → Gaussian PSF → Smits spectral upsampler → spectral integration → Voronoi viz → Snellen validation → dichromat confusion → species comparison pipeline.

## Key design decisions

- **PoC patch size**: 2° centered on area centralis. Scale to 5–10° later.
- **Mosaic generation**: jittered grid (not Poisson disk) for speed. Upgrade path noted in `retina/mosaic.py`.
- **PSF**: Gaussian placeholder first (`optical/psf.py::gaussian_psf`), then diffraction-limited FFT (`diffraction_psf`). Chromatic aberration deferred.
- **Spectral upsampling**: Smits (1999) first, Mallett-Yuksel later. Either can be bypassed entirely if a hyperspectral input is provided.
- **Each stage is independently validatable** — see architecture §11 for the full validation matrix. `tests/test_retina.py` covers Phase 1 (nomogram); other test files are stubs.

## Species constants

Peak wavelengths (nm) live in `retina/opsin.py::LAMBDA_MAX`. Optical params (pupil, focal length, axial length) are in `optical/stage.py::OpticalParams`. Cat uses a vertical slit pupil → anisotropic PSF; human and dog use circular pupils.
