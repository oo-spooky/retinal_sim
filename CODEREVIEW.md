# CODEREVIEW.md

Code review findings for `retinal_sim`. Open items should be addressed before starting new phase work. Resolved items are moved here rather than deleted for traceability.

---

## Context for next Opus review session

**Prepared by:** Sonnet (2026-03-31, end of Phase 7 session)

The last Opus review covered Phases 1–4. Since then, Phases 5, 6, and 7 have been implemented and all tests pass. This section gives Opus the orientation needed to review those three phases efficiently.

### What to review this session

- **Phase 5** — `retinal_sim/optical/psf.py`, `retinal_sim/optical/stage.py`
- **Phase 6** — `retinal_sim/spectral/upsampler.py`
- **Phase 7** — `retinal_sim/retina/stage.py` (the newly written `RetinalStage`; `retina/transduction.py` was already implemented but worth a quick pass)

Run `pytest` — expected: **259 passed, 1 skipped** in ~35 s.

### Key architectural decisions made since last review

**Phase 5 (Gaussian PSF / OpticalStage):**
- PSF kernels are stored as float64 (not float32) to satisfy the §11b energy conservation criterion `|sum - 1.0| < 1e-6`. float32 accumulates ~5e-5 rounding error over a 31×31 kernel.
- `PSFGenerator` takes `pixel_scale_mm_per_px` in its constructor (baked in). `OpticalStage.apply()` constructs a fresh `PSFGenerator` per call with the actual scene pixel scale; `OpticalStage.compute_psf()` uses a 1 µm/px default for standalone PSF inspection.
- Gaussian sigma = quadrature sum of diffraction component and defocus component (thin-lens approximation). `diffraction_psf` remains a `NotImplementedError` stub (post-PoC).
- `convolve(..., mode="reflect")` chosen over "constant" to avoid edge darkening.
- Media transmission applied per-band before convolution (not after), so the blurred result correctly represents attenuated irradiance.

**Phase 6 (Smits spectral upsampler):**
- Abandoned Smits (1999) hardcoded table — the MAGENTA and YELLOW values recalled from memory were wrong (RMSE > 0.18). Instead: each basis spectrum is computed at init via `scipy.optimize.lsq_linear` as the min-norm [0,1]-bounded reflectance satisfying the D65 XYZ colorimetric constraint.
- Roundtrip integration must use the D65 illuminant (not equal-energy). Per-channel white normalisation fails for saturated colours.
- The BLUE basis has non-zero values above 530 nm (~0.02–0.04); MAGENTA exploits the CIE x̄ secondary lobe at ~450 nm. Tests reflect this with a relaxed threshold (0.05, not 0.01).

**Phase 7 (RetinalStage / spectral integration):**
- `RetinalStage` wraps `MosaicGenerator` (Phase 4) and adds `compute_response()`.
- Bilinear interpolation maps receptor mm positions to the (H, W, N_λ) irradiance grid. Aperture Gaussian weighting (§3c) is deferred — at PoC pixel scales (~5 µm/px) and PSF widths (>20 µm), a 2–10 µm receptor aperture contributes negligible additional blur.
- Spectral integration: `np.einsum("nl,nl->n", sensitivities, sampled) * dlam`. Sensitivities come directly from `PhotoreceptorMosaic.sensitivities` (pre-built in Phase 4 via Govardovskii nomogram, no media filtering needed here because `OpticalStage.apply()` already bakes media transmission into the irradiance).
- `_align_sensitivities()` handles wavelength grid mismatches (e.g., coarser irradiance grid) by linear interpolation per receptor. The common path (both grids are the canonical 380–720 nm 5 nm grid) is a no-op.
- Naka-Rushton applied per unique receptor type. Parameters from `RetinalParams.naka_rushton_params` (YAML), falling back to `NAKA_RUSHTON_DEFAULTS`. Rod σ=0.1 (vs cone σ=0.5) means rods saturate at lower irradiance — verified in tests.
- Pixel scale priority: scene.mm_per_pixel → irradiance.metadata["pixel_scale_mm"] → patch-extent fallback.

### Specific things worth scrutinising

1. **`RetinalStage._align_sensitivities`** — per-receptor `np.interp` loop is O(N_receptors). For a human 2° patch with ~40K receptors, this is fine, but flag if it becomes a bottleneck at larger patches.
2. **Wavelength grid fragmentation** — `MosaicGenerator`, `SpectralUpsampler`, and `RetinalStage` all independently hardcode 380–720 nm at 5 nm. CR-5 from the last review noted this risk. A canonical constant shared across all three would be cleaner.
3. **`OpticalStage.apply()` rebuilds PSFGenerator on every call** — this is intentional (the pixel scale comes from the scene), but if the same stage is called in a loop with identical scenes it wastes recomputation. Not a bug, worth noting.
4. **No media transmission in `RetinalStage`** — the irradiance already has transmission applied by `OpticalStage`. If someone calls `RetinalStage.compute_response()` with a raw `SpectralImage` (skipping `OpticalStage`), there is no guard. This is acceptable for PoC but worth a docstring warning.
5. **`MosaicActivation.mosaic` typed as `object`** — it should be `PhotoreceptorMosaic`. The dataclass in `retina/stage.py` is not importing `PhotoreceptorMosaic` for the type annotation (to avoid circular imports, or just oversight). Check and tighten if clean.

---

## Previous review (2026-03-30)

**Reviewer:** Opus
**Scope:** Phases 1–4
**Test results:** 165 passed, 3 skipped in 9.12s

---

## Open Items

_No open items._

---

## Verified Correct

The following were audited and found correct against the architecture doc:

### Phase 1 — Govardovskii Nomogram (`retina/opsin.py`)
- A1 α-band template: coefficients (A=69.7, a=0.880, B=28.0, b=0.922, C=-14.9, c=1.104, D=0.674) match Govardovskii et al. (2000) Table 1 verbatim.
- A2 α-band: separate coefficient set correctly implemented.
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

### Stubs (Phases 5–8, pipeline)
- All stub signatures and dataclass definitions match architecture doc.
- `RetinalIrradiance`, `SpectralImage`, `MosaicActivation`, `SimulationResult` all have correct fields.
- Method signatures on `OpticalStage`, `RetinalStage`, `SpectralUpsampler`, `RetinalSimulator` match architecture.

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
