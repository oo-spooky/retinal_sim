# CODEREVIEW.md

Code review findings for `retinal_sim`. Open items should be addressed before starting new phase work. Resolved items are moved here rather than deleted for traceability.

**Review date:** 2026-03-30
**Reviewer:** Opus (code review session)
**Scope:** Phases 1–4 audited against `retinal_sim_architecture.md`
**Test results:** 165 passed, 3 skipped (stubs) in 9.12s

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
