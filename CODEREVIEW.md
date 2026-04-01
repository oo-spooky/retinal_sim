# CODEREVIEW.md

Code review findings for `retinal_sim`. Open items should be addressed before starting new phase work. Resolved items are moved here rather than deleted for traceability.

---

## Current review (2026-03-31)

**Reviewer:** Opus
**Scope:** Phases 5–7 (deep audit), cross-phase architectural review
**Test results:** 259 passed, 1 skipped in 24.38s

---

## Fix Priority Guide

**Before any new phase work:**
1. **CR-9** (PSF sigma) — physics error, every simulation is ~19% blurrier than it should be
2. **CR-10** (sRGB linearization) — physics error, spectra are wrong for all non-trivial colors
3. **CR-14** (MosaicActivation type) — 1-line fix, no reason to leave it
4. **CR-17** (wavelength grid constant) — 2-minute refactor, eliminates a class of silent bugs

**Before Phase 12 (species comparison):**
5. **CR-11** (pupil throughput) — cross-species irradiance comparison is meaningless without it
6. **CR-12** (cat slit pupil warning) — at minimum add a warning; anisotropic PSF ideal
7. **CR-15** (in-vitro sensitivities) — add docstring warning or apply media filtering
8. **CR-16** (edge clipping → zero) — affects small-scene simulations

**Low priority (fix when touching adjacent code):**
9. **CR-13** (A2 constants) — dormant, no current species uses A2

---

## Open Items

*(none)*

---

## Verified Correct

The following were audited and found correct against the architecture doc:

### Phase 1 — Govardovskii Nomogram (`retina/opsin.py`)
- A1 α-band template: coefficients (A=69.7, a=0.880, B=28.0, b=0.922, C=-14.9, c=1.104, D=0.674) match Govardovskii et al. (2000) Table 1 verbatim.
- β-band: Gaussian parameterization with published center/width coefficients for both A1 and A2.
- `x = λ_max / λ` transformation correctly applied.
- Peak normalization to 1.0 is standard.
- `LAMBDA_MAX` values match architecture §3b exactly for all species/receptor combinations.
- `build_sensitivity_curves()` correctly dispatches by species name.

### Phase 2 — Species Config (`species/config.py`, YAML files)
- All optical parameters match architecture §2b table (focal lengths, axial lengths, pupil shapes, corneal radii).
- All retinal parameters match architecture §3a table (cone types, peak densities, rod-free zone).
- Cone ratios match: human 1:16:32, dog 3:97, cat 5:95.
- Density function factories produce correct Gaussian falloff models.
- Rod-free zone correctly implemented (human only, 0.175mm radius = 0.35mm diameter foveal pit).

### Phase 3 — Scene Geometry (`scene/geometry.py`)
- Angular subtense: exact arctan formula, not small-angle approximation.
- Retinal extent: `2 × focal_length × tan(angle/2)` correct.
- Accommodation demand: `1/distance` diopters correct.
- Defocus residual: `max(0, demand - max)` correct.
- Patch clipping logic and coverage fraction correct.
- Infinity (collimated) case handled correctly (zero angular extent, zero retinal extent, zero accommodation).
- All fields in `SceneDescription` match architecture §0 output spec.

### Phase 4 — Mosaic Generator (`retina/mosaic.py`)
- Jittered grid algorithm correctly implements Bernoulli acceptance with probability = `local_density × cell_area`.
- Peak density radial scan prevents probability capping (SCRATCHPAD gotcha addressed).
- Stochastic type assignment respects local cone/rod density ratio.
- Cone type sub-assignment uses `cone_ratio_fn` probabilities correctly.
- `PhotoreceptorMosaic` dataclass matches architecture §3a data structure.
- Sensitivity curves built directly from `RetinalParams.cone_peak_wavelengths` without species-name lookup — clean design.

### CR-9: Gaussian PSF sigma overestimated ~19% ✓

- **File:** `retinal_sim/optical/psf.py`
- **Function:** `PSFGenerator.gaussian_psf()`
- **Fix applied:** Changed `fwhm_diff_mm = 1.22 * lam_mm * f_number; sigma = fwhm / (2√(2ln2))` to `sigma_diff_mm = 0.42 * lam_mm * f_number` (Gaussian-fit-to-Airy-disk result). Updated docstring and SCRATCHPAD.md. PROGRESS.md note corrected.

### CR-10: sRGB gamma linearization missing from spectral upsampler ✓

- **File:** `retinal_sim/spectral/upsampler.py`
- **Function:** `SpectralUpsampler.upsample()`
- **Fix applied:** Applied IEC 61966-2-1 piecewise sRGB→linear transfer function after clipping to [0,1]. Updated three tests in `test_spectral.py`: `test_gray_is_scaled_white` compares against linearised scale factor; `test_linearity_in_neutral_direction` computes expected ratio in linear space; `test_roundtrip_rmse_below_threshold` and `test_roundtrip_gray_ramp` compare against `_srgb_to_linear(test_colors)`.

### CR-11: Pupil throughput scaling missing from OpticalStage ✓

- **File:** `retinal_sim/optical/stage.py`
- **Function:** `OpticalStage.apply()`
- **Fix applied:** Added a clearly labelled `NOTE (CR-11)` comment explaining the missing pupil-area scaling, its impact on cross-species comparison via Naka-Rushton nonlinearity, and that it must be added before Phase 12.

### CR-12: Cat slit pupil geometry silently ignored in PSF computation ✓

- **File:** `retinal_sim/optical/psf.py`
- **Function:** `PSFGenerator.gaussian_psf()`
- **Fix applied:** Added `warnings.warn()` when `pupil_shape == 'slit'` explaining that isotropic Gaussian is used instead of the correct anisotropic PSF, with guidance for post-PoC fix.

### CR-13: A2 nomogram α-band uses A1 constants (dormant bug) ✓

- **File:** `retinal_sim/retina/opsin.py`
- **Fix applied:** Corrected `_A2_ALPHA` to Table 2 values from Govardovskii et al. (2000): `A=62.7, a=0.880, B=22.7, b=0.924, C=-14.0, c=1.104, D=0.674`.

### CR-14: MosaicActivation.mosaic typed as `object` instead of `PhotoreceptorMosaic` ✓

- **File:** `retinal_sim/retina/stage.py`
- **Fix applied:** Changed `mosaic: object` to `mosaic: PhotoreceptorMosaic`.

### CR-15: Sensitivity curves are in-vitro — no media pre-filtering per architecture §3b ✓

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `_build_sensitivity_curves()`
- **Fix applied:** Added docstring warning (option b) explaining that curves are in-vitro, the PoC equivalence assumption, and the risk if `RetinalStage` is called without a preceding `OpticalStage`.

### CR-16: Receptors outside image boundary receive edge-pixel irradiance, not zero ✓

- **File:** `retinal_sim/retina/stage.py`
- **Function:** `RetinalStage.compute_response()`
- **Fix applied:** Added `oob` mask before clipping; after bilinear interpolation `sampled[oob] = 0.0`. `test_rods_saturate_faster_than_cones` updated to use 200×200 image so peripheral rods are within the image footprint.

### CR-17: Wavelength grid still triplicated (CR-5 unresolved) ✓

- **Files:** `retinal_sim/constants.py` (new), `retina/mosaic.py`, `spectral/upsampler.py`, `retina/stage.py`
- **Fix applied:** Created `retinal_sim/constants.py` with canonical `WAVELENGTHS = np.arange(380, 721, 5, dtype=np.float64)`. All three modules import and use it; custom-range support in `SpectralUpsampler` is preserved.

### Phase 5 — Gaussian PSF (`optical/psf.py`, `optical/stage.py`)
- PSF kernel normalization to sum=1.0 in float64 meets §11b criterion (|sum-1| < 1e-6). Verified by test.
- Quadrature sum of diffraction + defocus sigma components is physically correct.
- Defocus circle-of-confusion formula `r_coc = f * δD * D_p / 2000` is dimensionally correct (mm = mm × D × mm / 1000).
- `convolve(mode="reflect")` is appropriate for avoiding edge darkening.
- Media transmission correctly applied per-band before convolution, matching architecture §2d pipeline formula.
- `RetinalIrradiance` dataclass has correct fields and metadata propagation.

### Phase 6 — Spectral Upsampler (`spectral/upsampler.py`)
- CIE 1931 2° observer CMFs at 5nm, 380–720nm: spot-checked representative values against CIE 015:2018. Count (69 bands) is correct for the range.
- D65 illuminant SPD: spot-checked against CIE tables. Normalization (S(560)=100) is standard.
- `_SRGB_TO_XYZ` matrix matches IEC 61966-2-1 specification.
- Basis spectra computed via `lsq_linear` with [0,1] bounds — mathematically sound approach to min-norm reflectance under D65 colorimetric constraint.
- Smits greedy-peel decomposition: partition logic (white→cyan/magenta/yellow→red/green/blue) is correct. Verified edge cases: gray pixels, pure primaries, secondaries, ties between channels — all route correctly.
- Roundtrip RMSE < 0.0002 (far below §11a threshold of 0.02).

### Phase 7 — Retinal Stage (`retina/stage.py`)
- Bilinear interpolation of (H,W,N_λ) irradiance to receptor positions: formula is correct (verified four-corner weighting sums to 1).
- Spectral integration via `np.einsum("nl,nl->n", sens, sampled) * dlam` correctly computes the discrete integral ∫ S(λ)·E(λ)dλ.
- Naka-Rushton formula `R_max * E^n / (E^n + σ^n)` matches architecture §3c exactly.
- Per-type Naka-Rushton dispatch with YAML params → fallback defaults is correct.
- Pixel-scale priority chain (scene → irradiance metadata → patch-extent fallback) is reasonable.
- `_align_sensitivities` correctly handles grid mismatch via per-receptor `np.interp` with zero fill for out-of-range.
- `RetinalParams` dataclass fields match architecture §3 specification.
- `RetinalStage.compute_response` signature adds `scene: object = None` beyond the architecture spec — this is a necessary extension for pixel-scale mapping and is consistent with how the pipeline orchestrator calls it (architecture §6).

### Phase 5–7 — Transduction (`retina/transduction.py`)
- `naka_rushton()` formula matches architecture §3c: `R_max × E^n / (E^n + σ^n)`.
- `np.maximum(exc, 0.0)` correctly guards against negative excitation (which can't occur physically but prevents NaN from fractional exponents).
- Default parameters per type: cones n=0.7 σ=0.5, rods n=1.0 σ=0.1 — match all three species YAML files.

### Phase 1–7 — Cross-phase checks
- All `LAMBDA_MAX` values in `opsin.py` match the corresponding `cone_peak_wavelengths` and `rod_peak_wavelength` in the three species YAML files.
- Cone ratios in YAML sum to exactly 1.0 for all three species.
- Accommodation limits in `geometry.py::_MAX_ACCOMMODATION` (human=10D, dog=2.5D, cat=3.0D) fall within the ranges given in architecture §0 table.
- `SimulationResult` dataclass fields match architecture §6.

---

## Previous reviews

**2026-03-31 (Opus):** Phases 5–7 deep audit + cross-phase review. 259 passed, 1 skipped. Findings: CR-9 through CR-17.
**2026-03-31 (Sonnet):** Addressed all CR-9 through CR-17 findings. 259 passed, 1 skipped.
**2026-03-30 (Opus):** Phases 1–4 initial audit. 165 passed, 3 skipped. Findings: CR-1 through CR-8 (all resolved).

---

## Resolved Items

### CR-1: `_species_name` never set on OpticalParams — accommodation defaults to human for all species ✓

- **File:** `retinal_sim/species/config.py`
- **Function:** `_build_optical()`
- **Fix applied:** Added `species_name` argument to `_build_optical()` and set `params._species_name = species_name` on the returned instance. `SpeciesConfig.load()` now passes `species_name` through. Dog/cat accommodation limits (2.5D, 3.0D) now correctly applied by `SceneGeometry.compute()`.

### CR-2: Naka-Rushton rod sigma mismatch between defaults and YAML (0.05 vs 0.1) ✓

- **File:** `retinal_sim/retina/transduction.py`
- **Function:** `NAKA_RUSHTON_DEFAULTS` (line 11)
- **Fix applied:** Changed `"sigma": 0.05` to `"sigma": 0.1` in the rod entry; updated comment to note it matches YAML species files.

### CR-3: Mosaic patch extent uses `focal_length_mm` where architecture doc §3a specifies `axial_length` ✓

- **File:** `retinal_sim_architecture.md` (§3a Processing step 1, §11c Nyquist formula)
- **Fix applied:** Updated both occurrences in the architecture doc to use `focal_length_mm` with an explanatory note that focal length (not axial length) governs retinal image magnification, matching the code.

### CR-4: Cone density model uses symmetric Gaussian — no visual streak for dog/cat ✓

- **File:** `retinal_sim/species/config.py`
- **Function:** `_make_cone_density_fn()`
- **Fix applied:** Added docstring explicitly noting visual streak modeling is a PoC simplification deferred to a later phase, with a TODO marker for Phase 5+.

### CR-5: Hardcoded wavelength range in MosaicGenerator duplicated from spectral stage ✓

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `MosaicGenerator.__init__()`
- **Fix applied:** Added TODO comment warning that the default range must be unified with `SpectralUpsampler` before Phase 7 spectral integration.

### CR-6: Magic number 1.1 in cell-size calculation undocumented ✓

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `MosaicGenerator.generate()`
- **Fix applied:** Added comment explaining the 1.1 safety margin ensures Bernoulli acceptance probability is never capped at or above 1.0.

### CR-7: Cone aperture linear model lacks rod aperture eccentricity dependence ✓

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `_cone_aperture_um()`
- **Fix applied:** Added docstring note that the linear model is a human-centric PoC approximation; rod aperture limitation noted; suggested moving parameters to species YAML in later phases.

### CR-8: `SpeciesConfig.optical` and `.retinal` typed as `object` ✓

- **File:** `retinal_sim/species/config.py`
- **Function:** `SpeciesConfig` dataclass
- **Fix applied:** Changed `optical: object` → `optical: "OpticalParams"` and `retinal: object` → `retinal: "RetinalParams"` (forward references; `from __future__ import annotations` already present).
