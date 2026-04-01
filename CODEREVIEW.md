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

### CR-9: Gaussian PSF sigma overestimated ~19% — Airy first-zero radius mislabeled as FWHM

- **File:** `retinal_sim/optical/psf.py`
- **Function:** `PSFGenerator.gaussian_psf()` (line 81)
- **Issue:** The code computes `fwhm_diff_mm = 1.22 * lam_mm * f_number`. The value `1.22 * λ * f#` is the **radius of the first dark ring** of the Airy diffraction pattern, not the FWHM. The actual FWHM of the Airy intensity pattern is `≈ 1.029 * λ * f#`. By treating the first-zero radius as FWHM, the code computes `σ = 1.22/(2√(2ln2)) * λ * f# ≈ 0.518 * λ * f#`, whereas a Gaussian properly fitted to the Airy disk has `σ ≈ 0.42–0.45 * λ * f#`. The PSF is ~19% wider than it should be, meaning the simulation underestimates optical resolution for all species.
- **Fix:** Change the formula to `fwhm_diff_mm = 1.029 * lam_mm * f_number` (or equivalently, compute `sigma_diff_mm = 0.42 * lam_mm * f_number` directly, which is the standard Gaussian-fit-to-Airy-disk result, bypassing the FWHM intermediate). Update the docstring comment accordingly. Also update the SCRATCHPAD entry that repeats this formula.

### CR-10: sRGB gamma linearization missing from spectral upsampler

- **File:** `retinal_sim/spectral/upsampler.py`
- **Function:** `SpectralUpsampler.upsample()` (line 136–139)
- **Issue:** Input RGB values are divided by 255 (for uint8) and clipped to [0,1], but no sRGB-to-linear conversion is applied. The `_SRGB_TO_XYZ` matrix used to compute basis spectra assumes **linear** sRGB input, and the Smits decomposition operates in linear space. Feeding gamma-compressed sRGB values produces spectra with incorrect spectral shapes. Example: sRGB (128,128,128) has linear value ~0.216, not 0.502 — the decomposition overweights the white basis by ~2.3x. The roundtrip test passes because the error is self-consistent (gamma-compressed in, same gamma-compressed out), but the intermediate spectra are physically wrong. This affects downstream spectral integration — photoreceptor excitations will have incorrect relative magnitudes across wavelengths.
- **Fix:** Apply sRGB linearization before decomposition: `linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)`. The roundtrip test should then compare against linearized values (or re-apply gamma to the output).

### CR-11: Pupil throughput scaling missing from OpticalStage

- **File:** `retinal_sim/optical/stage.py`
- **Function:** `OpticalStage.apply()` (line 93–96)
- **Issue:** Architecture §2a specifies "Scale input irradiance by pupil area (light gathering)." The implementation applies media transmission and PSF convolution but does not scale by pupil area. For cross-species comparison, this matters: a dog (4mm pupil, area=12.6mm²) gathers ~1.78× more light per unit solid angle than a human (3mm pupil, area=7.1mm²). Because Naka-Rushton is nonlinear, this missing scaling doesn't just shift responses — it changes their shape differently per species. Rods (σ=0.1) are especially sensitive: 1.78× more light significantly increases saturation.
- **Fix:** Multiply irradiance by `π * (pupil_diameter_mm / 2)² / reference_area` before convolution (or after — it's a scalar per species). Alternatively, normalize to a reference pupil area and document the convention. Deferred to the pipeline integration phase is acceptable, but must be present before Phase 12 (species comparison).

### CR-12: Cat slit pupil geometry silently ignored in PSF computation

- **File:** `retinal_sim/optical/psf.py`
- **Function:** `PSFGenerator.gaussian_psf()` (entire method)
- **Issue:** The cat YAML specifies `pupil_shape: slit`, and the architecture §2a explicitly states the slit pupil produces an anisotropic PSF "wider horizontally than vertically." The `gaussian_psf` method computes an isotropic circular Gaussian regardless of `pupil_shape` — it reads `pupil_diameter_mm` but never checks `pupil_shape`. The architecture acknowledges this as post-PoC (§11b), but the code contains **no warning, guard, or docstring note** that slit geometry is being ignored. A user running the cat pipeline would see isotropic blur and assume it's correct.
- **Fix:** Add a warning or docstring note in `gaussian_psf` that slit pupil geometry is not yet modeled. Optionally, log a `warnings.warn()` when `pupil_shape == 'slit'` so cat users are alerted. For a more complete fix (post-PoC): use an elliptical Gaussian with separate σ_x and σ_y for slit pupils.

### CR-13: A2 nomogram α-band uses A1 constants (dormant bug)

- **File:** `retinal_sim/retina/opsin.py`
- **Function:** `govardovskii_a2()` / `_A2_ALPHA` (line 26)
- **Issue:** `_A2_ALPHA` is identical to `_A1_ALPHA`: `dict(A=69.7, a=0.880, B=28.0, b=0.922, C=-14.9, c=1.104, D=0.674)`. Govardovskii et al. (2000) Table 2 gives **different** A2 α-band constants: A=62.7, a=0.880, B=22.7, b=0.924, C=-14.0, c=1.104, D=0.674. (A, B, C differ; a, b are close but not identical.) The A2 nomogram currently produces A1-shaped curves, missing the characteristic broader absorption of A2 pigments. This is dormant — no current species uses A2 — but will produce incorrect results when A2 species (fish, amphibians) are added.
- **Fix:** Correct `_A2_ALPHA` to: `dict(A=62.7, a=0.880, B=22.7, b=0.924, C=-14.0, c=1.104, D=0.674)`. Verify against Table 2 of the paper.

### CR-14: MosaicActivation.mosaic typed as `object` instead of `PhotoreceptorMosaic`

- **File:** `retinal_sim/retina/stage.py`
- **Function:** `MosaicActivation` dataclass (line 30)
- **Issue:** Architecture §3c specifies `mosaic: PhotoreceptorMosaic`. The implementation uses `mosaic: object`. `PhotoreceptorMosaic` is already imported at the top of the file (line 9), so there is no circular import obstacle.
- **Fix:** Change `mosaic: object` to `mosaic: PhotoreceptorMosaic`.

### CR-15: Sensitivity curves are in-vitro — no media pre-filtering per architecture §3b

- **File:** `retinal_sim/retina/mosaic.py`
- **Function:** `_build_sensitivity_curves()` (line 59)
- **Issue:** Architecture §3b states: "Apply lens/media pre-filtering (from Optical Stage `media_transmission`) to convert from in-vitro absorption to in-vivo spectral sensitivity." The implementation builds raw Govardovskii nomogram curves without any media filtering. While mathematically equivalent to applying transmission to irradiance (which `OpticalStage.apply()` does), this breaks the architecture's stage-independence principle — each stage should be independently validatable (§11). If `RetinalStage.compute_response()` is called without a preceding `OpticalStage` (e.g., with raw hyperspectral data, which the architecture explicitly allows in §1), the sensitivities will be in-vitro, missing lens absorption of UV/blue light. This primarily affects S-cones and rods, whose sensitivity would be overestimated at short wavelengths.
- **Fix:** Either (a) apply media transmission to sensitivity curves in `_build_sensitivity_curves()` (requires `OpticalParams.media_transmission` access), or (b) add a docstring warning that sensitivities are in-vitro and `OpticalStage` must precede `RetinalStage` in the pipeline. Option (b) is simpler for PoC; option (a) is correct for stage independence.

### CR-16: Receptors outside image boundary receive edge-pixel irradiance, not zero

- **File:** `retinal_sim/retina/stage.py`
- **Function:** `RetinalStage.compute_response()` (lines 131–132)
- **Issue:** `col_f` and `row_f` are clipped to `[0, W-1]` and `[0, H-1]` respectively. Receptors whose retinal positions fall outside the image see the nearest edge pixel value. Architecture §0 states: "If smaller, the image occupies a subregion of the mosaic and surrounding receptors receive no stimulation (background luminance only)." The current behavior produces a "stretched edge" artifact where peripheral receptors see the border of the image instead of darkness. This most affects small-scene simulations (e.g., a distant sign that subtends 0.3° in a 2° patch).
- **Fix:** Instead of clipping, set an out-of-bounds mask: `oob = (col_f < 0) | (col_f > W-1) | (row_f < 0) | (row_f > H-1)`. After bilinear interpolation, zero out `sampled[oob] = 0.0`. Then clip coordinates only for the in-bounds interpolation.

### CR-17: Wavelength grid still triplicated (CR-5 unresolved)

- **File:** `retinal_sim/retina/mosaic.py`, `retinal_sim/spectral/upsampler.py`, `retinal_sim/retina/stage.py`
- **Issue:** CR-5 was marked resolved by adding a TODO comment, but the underlying problem persists. Three modules independently define `np.arange(380, 721, 5)`: `MosaicGenerator.__init__` (mosaic.py:108), `SpectralUpsampler.__init__` (upsampler.py:113–118), and `RetinalStage._WAVELENGTHS` (stage.py:44). If any one is changed, the others silently diverge. Now that Phase 7 connects all three, this is no longer a future risk — it's a current fragility.
- **Fix:** Define a single canonical constant (e.g., `retinal_sim/constants.py::WAVELENGTHS`) and import it in all three modules. This is a two-minute refactor that eliminates an entire class of silent bugs.

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
